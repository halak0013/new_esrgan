import torch
from torch.utils.data import DataLoader

import src.utils.config as cfg
from src.utils.dataset import NatureDataset
from src.utils.dataset_from_folder import NatureFromFolder

from_folder = cfg.DATASET_FROM_FOLDER
train_dataset = (
    NatureDataset(type_="train") if not from_folder else NatureFromFolder(type_="train")
)
valid_dataset = (
    NatureDataset(type_="val") if not from_folder else NatureFromFolder(type_="val")
)
test_dataset = (
    NatureDataset(type_="test") if not from_folder else NatureFromFolder(type_="test")
)


train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=cfg.NUM_WORKERS,
    persistent_workers=True,
    prefetch_factor=2,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=cfg.BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
    num_workers=cfg.NUM_WORKERS,
    persistent_workers=True,
    prefetch_factor=2,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=cfg.BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
    num_workers=cfg.NUM_WORKERS,
    persistent_workers=True,
    prefetch_factor=2,
)


def get_loaders():
    return train_loader, valid_loader, test_loader
