"""
dataset.py — NIH Chest X-Ray14 veri yükleme ve ön-işleme.

Kullanım:
    from dataset import get_dataloaders
    train_loader, val_loader, test_loader, train_ds = get_dataloaders()
"""
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import config as cfg


# ── Transforms ───────────────────────────────

def train_transforms():
    return T.Compose([
        T.Resize((cfg.IMAGE_SIZE + 32, cfg.IMAGE_SIZE + 32)),
        T.RandomCrop(cfg.IMAGE_SIZE),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])


def val_transforms():
    return T.Compose([
        T.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])


# ── Dataset ──────────────────────────────────

class ChestXRayDataset(Dataset):
    """
    NIH Chest X-Ray14 binary sınıflandırma dataset'i.
    Label: 0 = Normal, 1 = Patolojik (herhangi bir bulgu var)
    """

    def __init__(self, csv_file, images_dir, transform=None, subset: Optional[List[str]] = None):
        self.images_dir = Path(images_dir)
        self.transform  = transform or val_transforms()

        df = pd.read_csv(csv_file)
        df = df[df["Image Index"].apply(lambda x: (self.images_dir / x).exists())].reset_index(drop=True)
        df["label"] = (df["Finding Labels"] != "No Finding").astype(int)

        if subset is not None:
            df = df[df["Image Index"].isin(subset)].reset_index(drop=True)

        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int, str]:
        row   = self.df.iloc[idx]
        image = Image.open(self.images_dir / row["Image Index"]).convert("RGB")
        label = int(row["label"])
        return self.transform(image), label, row["Image Index"]

    def class_weights(self) -> torch.Tensor:
        """Dengesiz sınıflar için ters-frekans ağırlıkları."""
        counts  = self.df["label"].value_counts().sort_index()
        total   = len(self.df)
        weights = torch.tensor([total / (len(counts) * counts[i]) for i in range(len(counts))],
                               dtype=torch.float32)
        return weights


# ── DataLoader Factory ───────────────────────

def get_dataloaders(batch_size: int = cfg.BATCH_SIZE):
    """
    Train / Val / Test DataLoader'larını döndürür.
    Returns: train_loader, val_loader, test_loader, train_dataset
    """
    df = pd.read_csv(cfg.CSV_FILE)
    df = df[df["Image Index"].apply(lambda x: (cfg.IMAGES_DIR / x).exists())].reset_index(drop=True)
    names = df["Image Index"].tolist()
    n     = len(names)

    rng     = np.random.default_rng(cfg.SEED)
    idx     = rng.permutation(n)
    n_train = int(n * cfg.TRAIN_RATIO)
    n_val   = int(n * cfg.VAL_RATIO)

    train_names = [names[i] for i in idx[:n_train]]
    val_names   = [names[i] for i in idx[n_train:n_train + n_val]]
    test_names  = [names[i] for i in idx[n_train + n_val:]]

    train_ds = ChestXRayDataset(cfg.CSV_FILE, cfg.IMAGES_DIR, train_transforms(), train_names)
    val_ds   = ChestXRayDataset(cfg.CSV_FILE, cfg.IMAGES_DIR, val_transforms(),   val_names)
    test_ds  = ChestXRayDataset(cfg.CSV_FILE, cfg.IMAGES_DIR, val_transforms(),   test_names)

    kw = dict(batch_size=batch_size, num_workers=cfg.NUM_WORKERS, pin_memory=(cfg.DEVICE == "cuda"))
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw)

    print(f"Dataset -> Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    return train_loader, val_loader, test_loader, train_ds
