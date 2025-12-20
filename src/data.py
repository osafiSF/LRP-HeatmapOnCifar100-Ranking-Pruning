"""Script holds data loader methods.
"""
from torchvision.datasets import CIFAR100
import argparse
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


def get_data_loader(cfg=None, batch_size: int = 128, resize: int = None, train: bool = False):
    if cfg is not None:
        batch_size = cfg.batch_size if hasattr(cfg, 'batch_size') else batch_size
        resize = cfg.resize if hasattr(cfg, 'resize') else resize

    # ... بقیه کد بدون تغییر    # ... بقیه کد بدون تغییر
    mean = (0.5071, 0.4865, 0.4409)
    std = (0.2673, 0.2564, 0.2762)

    if train:
        transform_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        transform_list = []

    if resize:
        transform_list.append(transforms.Resize((resize, resize)))

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform = transforms.Compose(transform_list)

    dataset = torchvision.datasets.CIFAR100(
        root="./data",
        train=train,
        download=True,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=0,  # مهم برای ویندوز
        pin_memory=torch.cuda.is_available()
    )
