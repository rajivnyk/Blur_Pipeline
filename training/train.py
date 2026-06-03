"""Train the DamageDetectorUNet on synthetic (clean + augmented) face data.

Designed to run on Kaggle (T4 GPU, ~4-6 hours for 30 epochs on FFHQ-256).

Usage on Kaggle
---------------
1. Add this dataset as input:
     FFHQ-256: https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq
   or CelebA:  https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
2. Upload this file + augment.py + dataset.py + model_arch.py to Kaggle
3. Set DATASET_ROOT to the input path (e.g. /kaggle/input/ffhq/thumbnails128x128)
4. Run. The final weights are saved to OUTPUT_PATH.
5. Download damage_detector.pth and copy it to backend/weights/

Loss: BCE + Dice (handles class imbalance between damaged vs clean pixels).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_ROOT = os.environ.get("DATASET_ROOT", "/kaggle/input/ffhq/thumbnails128x128")
OUTPUT_PATH  = os.environ.get("OUTPUT_PATH",  "/kaggle/working/damage_detector.pth")
IMAGE_SIZE   = int(os.environ.get("IMAGE_SIZE",  "256"))
BATCH_SIZE   = int(os.environ.get("BATCH_SIZE",  "16"))
EPOCHS       = int(os.environ.get("EPOCHS",      "30"))
LR           = float(os.environ.get("LR",        "1e-4"))
VAL_SPLIT    = float(os.environ.get("VAL_SPLIT", "0.1"))
NUM_WORKERS  = int(os.environ.get("NUM_WORKERS", "2"))
# ──────────────────────────────────────────────────────────────────────────────

# Allow importing sibling modules when run as a script or in a Kaggle notebook.
# __file__ is undefined in Jupyter kernels, so fall back to cwd.
try:
    _HERE = Path(__file__).parent
except NameError:
    _HERE = Path.cwd()

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "backend"))

from dataset import DamageDetectionDataset
from model_arch import DamageDetectorUNet


# ── Loss ──────────────────────────────────────────────────────────────────────

def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1.0 - (2.0 * intersection + eps) / (pred_flat.sum() + target_flat.sum() + eps)


def combined_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice


# ── Training loop ─────────────────────────────────────────────────────────────

def train() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    full_ds = DamageDetectionDataset(
        root_dir=DATASET_ROOT,
        image_size=IMAGE_SIZE,
        min_ops=2,
        max_ops=4,
        fading_prob=0.3,
    )
    n_val = max(1, int(len(full_ds) * VAL_SPLIT))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device == "cuda"),
    )

    print(f"Train: {n_train} samples | Val: {n_val} samples")

    model = DamageDetectorUNet(in_channels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # ── train ──
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = combined_loss(preds, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= n_train

        # ── validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                val_loss += combined_loss(preds, masks).item() * imgs.size(0)
        val_loss /= n_val

        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{EPOCHS}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), OUTPUT_PATH)
            print(f"  ✓ Saved best model (val_loss={best_val_loss:.4f})")

    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
    print(f"Weights saved to: {OUTPUT_PATH}")
    print("Copy this file to backend/weights/damage_detector.pth to enable damage detection.")


if __name__ == "__main__":
    train()
