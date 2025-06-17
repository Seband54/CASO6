import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import os
from PIL import Image


class AddGaussianNoise:
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0., 1.)


def get_dataloaders(batch_size=64, val_split=0.1):
    transform_clean = transforms.ToTensor()
    transform_noisy = transforms.Compose([
        transforms.ToTensor(),
        AddGaussianNoise(mean=0., std=0.2)
    ])

    # Descargar dataset CIFAR-10
    clean_dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform_clean)
    noisy_dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform_noisy)

    # Dataset combinado: (input_noisy, target_clean)
    class DenoisingDataset(Dataset):
        def __init__(self, noisy, clean):
            self.noisy = noisy
            self.clean = clean

        def __len__(self):
            return len(self.clean)

        def __getitem__(self, idx):
            noisy_img, _ = self.noisy[idx]
            clean_img, _ = self.clean[idx]
            return noisy_img, clean_img

    full_dataset = DenoisingDataset(noisy_dataset, clean_dataset)

    # División entrenamiento/validación
    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
