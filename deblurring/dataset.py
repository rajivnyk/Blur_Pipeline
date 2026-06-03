"""
GoPro Deblurring Dataset + Heavy Augmentation
===============================================
Supports the Kaggle GoPro dataset layout:

  /kaggle/input/gopro-image-deblurring-dataset/
    train/
      GOPR0372_07_00/
        blur/  *.png
        sharp/ *.png
      ...
    test/
      ...

Also handles flat layouts (train/blur/*.png, train/sharp/*.png) by searching
recursively and pairing by relative path stem.

Augmentation pipeline (training only)
--------------------------------------
Geometric (applied identically to blur + sharp):
  - RandomCrop 256×256
  - RandomHorizontalFlip
  - RandomVerticalFlip
  - RandomRotation ±30°

Degradation (applied to blur image only, simulating additional damage):
  - Gaussian noise (σ 0–25, p=0.15)
  - JPEG compression (quality 30–95, p=0.15)
  - Synthetic motion blur (kernel 5–25 px, p=0.12)
  - Synthetic defocus blur (σ 1–15, p=0.12)
  - Brightness / contrast jitter (p=0.3)
  - Saturation jitter (p=0.2)

Mixing:
  - MixUp between two blur images (α=0.4, p=0.1)
  - CutMix (p=0.1)
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


# ── pair discovery ─────────────────────────────────────────────────────────────

def _diagnose(root: Path, split: str) -> None:
    """Print the directory tree to help debug structure mismatches."""
    print(f"\n[GoproDataset] Could not find pairs for split='{split}'")
    print(f"  root={root}  exists={root.exists()}")
    if root.exists():
        top = sorted(root.iterdir())
        print(f"  root/: {[p.name for p in top[:12]]}")
        split_dir = root / split
        if split_dir.exists():
            children = sorted(split_dir.iterdir())
            print(f"  {split}/: {[p.name for p in children[:12]]}")
            for sub in children[:4]:
                if sub.is_dir():
                    print(f"    {sub.name}/: {[p.name for p in sorted(sub.iterdir())[:6]]}")
        else:
            print(f"  '{split}/' not found. Top-level dirs: {[p.name for p in top if p.is_dir()]}")
    blur_hits = list(root.rglob("blur"))[:6]
    print(f"  'blur' dirs found anywhere under root: {blur_hits}")


def _pairs_from_search_root(search_root: Path) -> list[tuple[Path, Path]]:
    """Find (blur, sharp) pairs by recursively searching under search_root."""
    blur_files = sorted(
        p for p in search_root.rglob("*.*")
        if p.suffix.lower() in IMG_EXTS and "blur" in p.parts
        and "sharp" not in p.parts
    )
    if not blur_files:
        return []
    pairs = []
    for bp in blur_files:
        parts = list(bp.parts)
        # Replace the first 'blur' component with 'sharp'
        for i, part in enumerate(parts):
            if part == "blur":
                parts[i] = "sharp"
                break
        sp = Path(*parts)
        if sp.exists():
            pairs.append((bp, sp))
    return pairs


def _find_pairs(root: Path, split: str) -> list[tuple[Path, Path]]:
    """
    Return list of (blur_path, sharp_path) tuples for the given split.

    Tries multiple layout strategies so the function works regardless of
    whether Kaggle adds an extra /datasets/<user>/ prefix or the dataset
    uses sequences vs flat organisation:

      Layout A  <split>/<seq>/blur/*.png  +  <split>/<seq>/sharp/*.png
      Layout B  <split>/blur/*.png        +  <split>/sharp/*.png
      Layout C  root is already the split dir (caller passed wrong root)
      Layout D  split folder found anywhere under root (deep nesting)
    """
    # Build a list of candidate search roots in priority order
    candidates: list[Path] = []

    split_dir = root / split
    if split_dir.exists():
        candidates.append(split_dir)         # normal case

    # Maybe root itself is the split directory
    candidates.append(root)

    # Maybe the split dir is nested deeper (e.g. root/gopro/train/)
    for d in root.rglob(split):
        if d.is_dir() and d not in candidates:
            candidates.append(d)

    for search_root in candidates:
        pairs = _pairs_from_search_root(search_root)
        if pairs:
            return pairs

    _diagnose(root, split)
    raise FileNotFoundError(
        f"Could not find blur/sharp pairs for split='{split}' "
        f"under {root}. Check the diagnostic output above."
    )


# ── augmentation helpers ───────────────────────────────────────────────────────

def _add_gaussian_noise(img: np.ndarray, sigma_max: float = 25.0) -> np.ndarray:
    sigma = random.uniform(0, sigma_max) / 255.0
    return np.clip(img + np.random.randn(*img.shape).astype(np.float32) * sigma, 0, 1)


def _add_jpeg(img: np.ndarray, q_min: int = 30, q_max: int = 95) -> np.ndarray:
    q   = random.randint(q_min, q_max)
    u8  = (img * 255).clip(0, 255).astype(np.uint8)
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(u8, cv2.COLOR_RGB2BGR),
                          [cv2.IMWRITE_JPEG_QUALITY, q])
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _add_motion_blur(img: np.ndarray, k_min: int = 5, k_max: int = 25) -> np.ndarray:
    size = random.randrange(k_min, k_max + 1, 2)
    kernel = np.zeros((size, size), dtype=np.float32)
    angle  = random.uniform(0, 180)
    cv2.ellipse(kernel, (size // 2, size // 2), (size // 2, 0), angle, 0, 360, 1, -1)
    kernel /= kernel.sum() + 1e-8
    u8 = (img * 255).clip(0, 255).astype(np.uint8)
    blurred = cv2.filter2D(u8, -1, kernel)
    return blurred.astype(np.float32) / 255.0


def _add_defocus_blur(img: np.ndarray, r_min: float = 1.0, r_max: float = 15.0) -> np.ndarray:
    sigma = random.uniform(r_min, r_max)
    ksize = int(2 * math.ceil(2 * sigma) + 1)
    k1d   = cv2.getGaussianKernel(ksize, sigma)
    kernel = k1d @ k1d.T
    u8 = (img * 255).clip(0, 255).astype(np.uint8)
    blurred = cv2.filter2D(u8, -1, kernel)
    return blurred.astype(np.float32) / 255.0


def _color_jitter(img: np.ndarray) -> np.ndarray:
    # Brightness + contrast
    alpha = random.uniform(0.75, 1.25)   # contrast
    beta  = random.uniform(-0.1, 0.1)    # brightness
    img = np.clip(alpha * img + beta, 0, 1)
    # Saturation (convert to HSV momentarily)
    if random.random() < 0.5:
        u8  = (img * 255).clip(0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(cv2.cvtColor(u8, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= random.uniform(0.7, 1.3)
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img


def _random_crop(a: np.ndarray, b: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = a.shape[:2]
    if h < size:
        a = cv2.resize(a, (w, size), interpolation=cv2.INTER_LINEAR)
        b = cv2.resize(b, (w, size), interpolation=cv2.INTER_LINEAR)
        h = size
    if w < size:
        a = cv2.resize(a, (size, h), interpolation=cv2.INTER_LINEAR)
        b = cv2.resize(b, (size, h), interpolation=cv2.INTER_LINEAR)
        w = size
    top  = random.randint(0, h - size)
    left = random.randint(0, w - size)
    return a[top:top+size, left:left+size], b[top:top+size, left:left+size]


def _rotate(a: np.ndarray, b: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = a.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    kw = {"flags": cv2.INTER_LINEAR, "borderMode": cv2.BORDER_REFLECT_101}
    return cv2.warpAffine(a, M, (w, h), **kw), cv2.warpAffine(b, M, (w, h), **kw)


# ── math import (needed by _add_defocus_blur) ─────────────────────────────────
import math


# ── dataset ───────────────────────────────────────────────────────────────────

class GoproDataset(Dataset):
    """
    GoPro paired deblurring dataset with on-the-fly augmentation.

    Parameters
    ----------
    root           : path to the dataset root (contains train/ and test/)
    split          : "train" or "test"
    patch_size     : random-crop size used during training
    val_patch_size : if > 0 and split == "test", apply a centre-crop of this
                     size for validation instead of returning full-resolution
                     images.  Full-res evaluation is accurate but slow
                     (~1.4 s / 1280×720 image on T4).  A 512×512 centre-crop
                     is ~8× faster and still gives a reliable PSNR proxy
                     during training.  Set to 0 to disable (default = 512).
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        patch_size: int = 256,
        val_patch_size: int = 512,
    ) -> None:
        self.pairs          = _find_pairs(Path(root), split)
        self.patch_size     = patch_size
        self.val_patch_size = val_patch_size
        self.is_train       = (split == "train")
        print(f"[GoproDataset] {split}: {len(self.pairs)} pairs found")

    def __len__(self) -> int:
        return len(self.pairs)

    def _load(self, p: Path) -> np.ndarray:
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    def _augment(self, blur: np.ndarray, sharp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # ── geometric (both images) ──
        blur, sharp = _random_crop(blur, sharp, self.patch_size)

        if random.random() > 0.5:
            blur, sharp = np.fliplr(blur).copy(), np.fliplr(sharp).copy()
        if random.random() > 0.5:
            blur, sharp = np.flipud(blur).copy(), np.flipud(sharp).copy()
        if random.random() > 0.5:
            angle = random.uniform(-30, 30)
            blur, sharp = _rotate(blur, sharp, angle)

        # ── additional degradation (blur image only) ──
        if random.random() < 0.15:
            blur = _add_gaussian_noise(blur)
        if random.random() < 0.15:
            blur = _add_jpeg(blur)
        if random.random() < 0.12:
            blur = _add_motion_blur(blur)
        if random.random() < 0.12:
            blur = _add_defocus_blur(blur)
        if random.random() < 0.30:
            blur = _color_jitter(blur)

        return blur, sharp

    def _mixup(self, blur1: np.ndarray, sharp1: np.ndarray,
               blur2: np.ndarray, sharp2: np.ndarray, alpha: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
        lam  = np.random.beta(alpha, alpha)
        return (lam * blur1  + (1 - lam) * blur2,
                lam * sharp1 + (1 - lam) * sharp2)

    def _cutmix(self, blur1: np.ndarray, sharp1: np.ndarray,
                blur2: np.ndarray, sharp2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w  = blur1.shape[:2]
        lam   = np.random.beta(1.0, 1.0)
        cut_h = int(h * math.sqrt(1 - lam))
        cut_w = int(w * math.sqrt(1 - lam))
        cx    = random.randint(0, w)
        cy    = random.randint(0, h)
        x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)
        y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
        out_b, out_s = blur1.copy(), sharp1.copy()
        out_b[y1:y2, x1:x2] = blur2[y1:y2, x1:x2]
        out_s[y1:y2, x1:x2] = sharp2[y1:y2, x1:x2]
        return out_b, out_s

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        blur_p, sharp_p = self.pairs[idx]
        blur  = self._load(blur_p)
        sharp = self._load(sharp_p)

        if self.is_train:
            blur, sharp = self._augment(blur, sharp)

            # MixUp (10%)
            if random.random() < 0.10:
                idx2 = random.randint(0, len(self.pairs) - 1)
                b2   = self._load(self.pairs[idx2][0])
                s2   = self._load(self.pairs[idx2][1])
                b2, s2 = _random_crop(b2, s2, self.patch_size)
                blur, sharp = self._mixup(blur, sharp, b2, s2)

            # CutMix (10%)
            elif random.random() < 0.10:
                idx2 = random.randint(0, len(self.pairs) - 1)
                b2   = self._load(self.pairs[idx2][0])
                s2   = self._load(self.pairs[idx2][1])
                b2, s2 = _random_crop(b2, s2, self.patch_size)
                blur, sharp = self._cutmix(blur, sharp, b2, s2)
        else:
            # Validation: optional centre-crop for faster evaluation.
            # val_patch_size=0 keeps full resolution (accurate but slow).
            vp = self.val_patch_size
            if vp and vp > 0:
                h, w = blur.shape[:2]
                if h >= vp and w >= vp:
                    top  = (h - vp) // 2
                    left = (w - vp) // 2
                    blur  = blur[top:top+vp, left:left+vp]
                    sharp = sharp[top:top+vp, left:left+vp]

        # HWC float32 → CHW float32 tensor
        to_t = lambda x: torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        return to_t(blur.clip(0, 1)), to_t(sharp.clip(0, 1))
