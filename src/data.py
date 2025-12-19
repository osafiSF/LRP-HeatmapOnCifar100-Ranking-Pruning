"""Script holds data loader methods.
"""
import argparse
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


def get_data_loader(config: argparse.Namespace) -> torch.utils.data.DataLoader:
    """
    Creates dataloader for CIFAR-100 dataset.

    Args:
        config: Argparse namespace object.

    Returns:
        Data loader object.
    """

    batch_size = config.batch_size

    # CIFAR-100 mean / std
    mean = (0.5071, 0.4865, 0.4409)
    std = (0.2673, 0.2564, 0.2762)

    transform_list = []

    # Resize برای سازگاری با VGG
    if config.resize:
        transform_list.append(
            transforms.Resize((config.resize, config.resize))
        )

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform = transforms.Compose(transform_list)

    dataset = torchvision.datasets.CIFAR100(
        root="./data",
        train=False,          # برای LRP بهتره test باشه
        download=True,
        transform=transform,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return data_loader
