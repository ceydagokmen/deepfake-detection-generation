
import torch
import os
from dataset import get_dataloaders
from model import DeepfakeDetector
from train import Trainer
from utils import plot_training_history, plot_confusion_matrix, evaluate
from generate import FaceGenerator

# Ayarlar

DATA_DIR  = "./data/real_vs_fake/real-vs-fake"
MODEL_PATH = "./models/deepfake_detector_v2.pth"
GAN_PATH   = "./ffhq.pkl"
EPOCHS     = 5
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("./models",  exist_ok=True)
os.makedirs("./results", exist_ok=True)

print(f"Cihaz: {DEVICE}")

# Veri

train_loader, valid_loader, test_loader = get_dataloaders(DATA_DIR)

# Model Eğitimi

model = DeepfakeDetector().to(DEVICE)
trainer = Trainer(model, DEVICE, lr=0.0001)
history = trainer.fit(train_loader, valid_loader, epochs=EPOCHS)
model.save(MODEL_PATH)

# Sonuçlar

plot_training_history(history)
all_labels, all_preds = evaluate(model, test_loader, DEVICE)
plot_confusion_matrix(all_labels, all_preds)

# Üretim Örneği

generator = FaceGenerator(GAN_PATH, DEVICE)
generator.generate_grid(n=6)

print("Tüm işlemler tamamlandı!")
