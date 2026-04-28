
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


def get_subset(dataset, per_class=5000):
    indices = []
    for class_idx in range(len(dataset.classes)):
        class_indices = [
            i for i, (_, label) in enumerate(dataset.samples)
            if label == class_idx
        ]
        indices += class_indices[:per_class]
    return Subset(dataset, indices)


def get_dataloaders(data_dir, batch_size=32, num_workers=2):
    transform = get_transform()

    train_data = ImageFolder(f"{data_dir}/train", transform=transform)
    valid_data = ImageFolder(f"{data_dir}/valid", transform=transform)
    test_data  = ImageFolder(f"{data_dir}/test",  transform=transform)

    train_loader = DataLoader(get_subset(train_data, 5000),
                              batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    valid_loader = DataLoader(get_subset(valid_data, 2000),
                              batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    test_loader  = DataLoader(get_subset(test_data, 1000),
                              batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    print(f"Sınıflar: {train_data.classes}")
    print(f"Eğitim: {len(train_loader.dataset)} | "
          f"Validasyon: {len(valid_loader.dataset)} | "
          f"Test: {len(test_loader.dataset)}")

    return train_loader, valid_loader, test_loader
