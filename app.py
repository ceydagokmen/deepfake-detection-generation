
import gradio as gr
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
import numpy as np
import sys
sys.path.insert(0, './stylegan2-ada-pytorch')
import pickle

# Cihaz
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Algılama modeli
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load("deepfake_detector_v2.pth", map_location=device))
model = model.to(device)
model.eval()

transform_single = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# StyleGAN2
with open("ffhq.pkl", "rb") as f:
    G = pickle.load(f)["G_ema"].to(device)

def detect(image):
    img_tensor = transform_single(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)[0]
    fake_prob = probs[0].item()
    real_prob = probs[1].item()
    if fake_prob > real_prob:
        label = "🔴 SAHTE (Deepfake)"
        confidence = fake_prob
    else:
        label = "🟢 GERÇEK"
        confidence = real_prob
    return {f"{label} — Güven: {confidence*100:.1f}%": confidence, "Karşı ihtimal": 1 - confidence}

def generate(_):
    seed = np.random.randint(0, 99999)
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device).float()
    c = torch.zeros([1, G.c_dim]).to(device)
    with torch.no_grad():
        img = G(z, c, truncation_psi=0.7)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return Image.fromarray(img[0].cpu().numpy())

with gr.Blocks(title="Deepfake Sistemi") as demo:
    gr.Markdown("# 🎭 Deepfake Algılama ve Üretme Sistemi")
    with gr.Tab("🔍 Deepfake Algılama"):
        gr.Markdown("Bir yüz fotoğrafı yükle — gerçek mi sahte mi olduğunu söyleyelim.")
        with gr.Row():
            inp = gr.Image(type="pil", label="Fotoğraf Yükle")
            out = gr.Label(label="Sonuç")
        btn1 = gr.Button("Analiz Et", variant="primary")
        btn1.click(fn=detect, inputs=inp, outputs=out)
    with gr.Tab("🎨 Sahte Yüz Üret (StyleGAN2)"):
        gr.Markdown("Butona bas — hiç var olmamış bir yüz üretilsin.")
        out_img = gr.Image(label="Üretilen Sahte Yüz")
        btn2 = gr.Button("Yeni Yüz Üret 🎲", variant="primary")
        btn2.click(fn=generate, inputs=btn2, outputs=out_img)

demo.launch()
