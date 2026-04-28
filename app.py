
import sys
import torch
import numpy as np
import gradio as gr
from PIL import Image

sys.path.insert(0, './stylegan2-ada-pytorch')

from model import DeepfakeDetector
from dataset import get_transform
from generate import FaceGenerator


class DeepfakeApp:
    def __init__(self, model_path, generator_path, device):
        self.device = device
        self.transform = get_transform()

        self.detector = DeepfakeDetector()
        self.detector.load(model_path, device)
        self.detector = self.detector.to(device)

        self.generator = FaceGenerator(generator_path, device)

    def detect(self, image):
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.detector(img_tensor)
            probs = torch.softmax(output, dim=1)[0]
        fake_prob = probs[0].item()
        real_prob = probs[1].item()
        if fake_prob > real_prob:
            label = "🔴 SAHTE (Deepfake)"
            confidence = fake_prob
        else:
            label = "🟢 GERÇEK"
            confidence = real_prob
        return {
            f"{label} — Güven: {confidence*100:.1f}%": confidence,
            "Karşı ihtimal": 1 - confidence
        }

    def generate(self, _):
        img, seed = self.generator.generate()
        return img

    def launch(self):
        with gr.Blocks(title="Deepfake Sistemi") as demo:
            gr.Markdown("# 🎭 Deepfake Algılama ve Üretme Sistemi")

            with gr.Tab("🔍 Deepfake Algılama"):
                gr.Markdown("Bir yüz fotoğrafı yükle — gerçek mi sahte mi?")
                with gr.Row():
                    inp = gr.Image(type="pil", label="Fotoğraf Yükle")
                    out = gr.Label(label="Sonuç")
                btn1 = gr.Button("Analiz Et", variant="primary")
                btn1.click(fn=self.detect, inputs=inp, outputs=out)

            with gr.Tab("🎨 Sahte Yüz Üret (StyleGAN2)"):
                gr.Markdown("Butona bas — hiç var olmamış bir yüz üretilsin.")
                out_img = gr.Image(label="Üretilen Sahte Yüz")
                btn2 = gr.Button("Yeni Yüz Üret 🎲", variant="primary")
                btn2.click(fn=self.generate, inputs=btn2, outputs=out_img)

        demo.launch(share=True)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app = DeepfakeApp(
        model_path="./deepfake_detector_v2.pth",
        generator_path="./ffhq.pkl",
        device=device
    )
    app.launch()
