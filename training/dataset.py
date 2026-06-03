"""PyTorch Dataset for damage-detection U-Net training.

Expects a directory of clean face images (FFHQ or CelebA).
On each __getitem__ call it applies random synthetic damage on-the-fly so
every epoch sees fresh augmentation — no need to pre-generate pairs on disk.

Dataset sources (download on Kaggle before running train.py):
  FFHQ 256x256: https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq
  CelebA:       https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from augment import apply_random_damage

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

_to_tensor = transforms.ToTensor()
_normalise = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


def _load_rgb(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


class DamageDetectionDataset(Dataset):
    """Paired (damaged image, binary mask) dataset generated on-the-fly.

    Parameters
    ----------
    root_dir : str | Path
        Directory containing clean face images (recursive search).
    image_size : int
        Spatial size for training (both H and W, e.g. 256 or 512).
    min_ops, max_ops : int
        Min / max number of damage operations applied per sample.
    fading_prob : float
        Probability of adding global fading on top of other damage.
    """

    def __init__(
        self,
        root_dir: str | Path,
        image_size: int = 256,
        min_ops: int = 2,
        max_ops: int = 4,
        fading_prob: float = 0.3,
    ) -> None:
        root = Path(root_dir)
        self.paths = [
            p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS
        ]
        if not self.paths:
            raise FileNotFoundError(f"No images found under {root}")

        self.image_size = image_size
        self.min_ops = min_ops
        self.max_ops = max_ops
        self.fading_prob = fading_prob

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        clean_rgb = _load_rgb(self.paths[idx], self.image_size)
        damaged_rgb, mask_uint8 = apply_random_damage(
            clean_rgb,
            min_ops=self.min_ops,
            max_ops=self.max_ops,
            fading_prob=self.fading_prob,
        )

        # image: float32 tensor [3, H, W] in [-1, 1]
        img_tensor = _normalise(_to_tensor(damaged_rgb))

        # mask: float32 tensor [1, H, W] in [0, 1]
        mask_tensor = torch.from_numpy(mask_uint8.astype(np.float32) / 255.0).unsqueeze(0)

        return img_tensor, mask_tensor
