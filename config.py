"""
config.py — Tüm hyperparameter ve path ayarları.
DATA_ROOT'u kendi dataset klasörünüze göre düzenleyin.
"""
import os
from pathlib import Path
import torch

# ── Paths ───────────────────────────────────
ROOT        = Path(__file__).resolve().parent
DATA_DIR    = ROOT / "data" / "raw"
CKPT_DIR    = ROOT / "results" / "checkpoints"
FIGURES_DIR = ROOT / "results" / "figures"
TABLES_DIR  = ROOT / "results" / "tables"
LOG_DIR     = ROOT / "logs"

for d in [DATA_DIR, CKPT_DIR, FIGURES_DIR, TABLES_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Dataset ──────────────────────────────────
CSV_FILE    = DATA_DIR / "Data_Entry_2017.csv"
IMAGES_DIR  = DATA_DIR / "images"
IMAGE_SIZE  = 224
NUM_CLASSES = 2
CLASS_NAMES = ["Normal", "Pathological"]

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
SEED        = 42

# ── Training ─────────────────────────────────
BATCH_SIZE    = 32          # RTX 4060 can handle 32+ easily for 224px
NUM_EPOCHS    = 15          # Faster iteration
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
PATIENCE      = 5           # early stopping

# ── Model ────────────────────────────────────
BACKBONE    = "resnet50"    # resnet18 | resnet50 | densenet121
PRETRAINED  = True          # Use ImageNet weights for real training
DROPOUT_P   = 0.25
MC_SAMPLES  = 30            # Standard MC samples for reliability

# ── Hardware ─────────────────────────────────
DEVICE      = "cuda"        # Explicitly use RTX 4060
NUM_WORKERS = 4             # Higher workers for faster data loading


def set_seed(s: int = SEED):
    import random, numpy as np
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
