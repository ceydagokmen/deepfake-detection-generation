
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch


def plot_training_history(history, save_path="./results/training_results.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-o", label="Eğitim")
    ax1.plot(epochs, history["val_loss"],   "r-o", label="Validasyon")
    ax1.set_title("Loss (Kayıp)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, [x*100 for x in history["train_acc"]], "b-o", label="Eğitim")
    ax2.plot(epochs, [x*100 for x in history["val_acc"]],   "r-o", label="Validasyon")
    ax2.set_title("Accuracy (Doğruluk %)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Doğruluk (%)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Grafik kaydedildi: {save_path}")


def plot_confusion_matrix(all_labels, all_preds,
                          save_path="./results/confusion_matrix.png"):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fake", "Real"],
                yticklabels=["Fake", "Real"])
    plt.title("Confusion Matrix")
    plt.ylabel("Gerçek Etiket")
    plt.xlabel("Tahmin")
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(classification_report(all_labels, all_preds,
                                target_names=["Fake", "Real"]))


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return all_labels, all_preds
