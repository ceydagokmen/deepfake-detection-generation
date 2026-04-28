
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


class Trainer:
    def __init__(self, model, device, lr=0.0001):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=3, gamma=0.5)
        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": []
        }

    def train_epoch(self, loader):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        return total_loss / len(loader), correct / total

    def validate(self, loader):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
        return total_loss / len(loader), correct / total

    def fit(self, train_loader, valid_loader, epochs=5):
        for epoch in range(epochs):
            tl, ta = self.train_epoch(train_loader)
            vl, va = self.validate(valid_loader)
            self.scheduler.step()
            self.history["train_loss"].append(tl)
            self.history["train_acc"].append(ta)
            self.history["val_loss"].append(vl)
            self.history["val_acc"].append(va)
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {tl:.4f} | Train Acc: {ta*100:.2f}% | "
                  f"Val Loss: {vl:.4f} | Val Acc: {va*100:.2f}%")
        return self.history
