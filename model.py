
import torch
import torch.nn as nn
from torchvision import models


class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepfakeDetector, self).__init__()
        self.backbone = models.efficientnet_b0(weights="IMAGENET1K_V1")
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

    def save(self, path):
        torch.save(self.state_dict(), path)
        print(f"Model kaydedildi: {path}")

    def load(self, path, device):
        self.load_state_dict(torch.load(path, map_location=device))
        self.eval()
        print(f"Model yüklendi: {path}")
