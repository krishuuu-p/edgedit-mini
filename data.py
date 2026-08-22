import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def get_dataloaders(data_dir="./data", batch_size=128, num_workers=2):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # -> [-1, 1]
    ])
    train_set = datasets.FashionMNIST(data_dir, train=True, download=True, transform=tfm)
    test_set = datasets.FashionMNIST(data_dir, train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True, pin_memory=True)
    return train_loader, test_loader


def get_calibration_batch(loader, device):
    """One real batch, used repeatedly as calibration data for feature-wise KD."""
    x, y = next(iter(loader))
    return x.to(device), y.to(device)
