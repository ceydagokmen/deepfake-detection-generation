
import torch
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt


class FaceGenerator:
    def __init__(self, model_path, device):
        self.device = device
        with open(model_path, "rb") as f:
            self.G = pickle.load(f)["G_ema"].to(device)
        print("StyleGAN2 generator yüklendi!")

    def generate(self, seed=None, truncation_psi=0.7):
        if seed is None:
            seed = np.random.randint(0, 99999)
        z = torch.from_numpy(
            np.random.RandomState(seed).randn(1, self.G.z_dim)
        ).to(self.device).float()
        c = torch.zeros([1, self.G.c_dim]).to(self.device)
        with torch.no_grad():
            img = self.G(z, c, truncation_psi=truncation_psi)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return Image.fromarray(img[0].cpu().numpy()), seed

    def generate_grid(self, n=6, save_path="./results/generated_faces.png"):
        rows = n // 3
        fig, axes = plt.subplots(rows, 3, figsize=(12, rows * 4))
        fig.suptitle("StyleGAN2 ile Üretilen Sahte Yüzler", fontsize=14)
        for i, ax in enumerate(axes.flat):
            img, seed = self.generate()
            ax.imshow(img)
            ax.set_title(f"Seed: {seed}", fontsize=8)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.show()
        print(f"Grid kaydedildi: {save_path}")
