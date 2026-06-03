"""
Utilities: metrics, EMA, checkpoints, visualisation
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn


class AverageMeter:
    def __init__(self) -> None: self.reset()
    def reset(self) -> None:    self.val = self.avg = self.sum = self.count = 0.0
    def update(self, val: float, n: int = 1) -> None:
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor,
                   max_val: float = 1.0) -> float:
    with torch.no_grad():
        pred   = pred.float().clamp(0, max_val)
        target = target.float().clamp(0, max_val)
        mse    = ((pred - target) ** 2).mean()
        if mse == 0:
            return float("inf")
        return float(20 * math.log10(max_val) - 10 * torch.log10(mse).item())


# Per-device SSIMLoss singleton — avoids GPU alloc on every validation call
_ssim_cache: dict = {}


def calculate_ssim_batch(pred: torch.Tensor, target: torch.Tensor) -> float:
    from losses import SSIMLoss
    dev = pred.device
    if dev not in _ssim_cache:
        _ssim_cache[dev] = SSIMLoss().to(dev)
    with torch.no_grad():
        return float(1.0 - _ssim_cache[dev](pred.clamp(0, 1),
                                              target.clamp(0, 1)).item())


def laplacian_sharpness(img: np.ndarray) -> float:
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class EMA:
    """Exponential Moving Average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay  = decay
        self.shadow = deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(m.data, alpha=1.0 - self.decay)

    def state_dict(self) -> dict:        return self.shadow.state_dict()
    def load_state_dict(self, sd: dict) -> None: self.shadow.load_state_dict(sd)


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    """Atomic save: write to .tmp then rename so file is never half-written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pth.tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(
    path: str | Path,
    model:       nn.Module,
    optimizer_G: torch.optim.Optimizer | None = None,
    disc:        nn.Module | None = None,
    optimizer_D: torch.optim.Optimizer | None = None,
    ema:         EMA | None = None,
) -> tuple[int, float]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    if optimizer_G and "optimizer_G" in ck: optimizer_G.load_state_dict(ck["optimizer_G"])
    if disc        and "disc"        in ck: disc.load_state_dict(ck["disc"])
    if optimizer_D and "optimizer_D" in ck: optimizer_D.load_state_dict(ck["optimizer_D"])
    if ema         and "ema"         in ck: ema.load_state_dict(ck["ema"])
    return ck.get("epoch", 0), ck.get("best_psnr", 0.0)


class TrainingLog:
    def __init__(self, path: str | Path) -> None:
        self.path    = Path(path)
        self.records: list[dict] = []

    def append(self, record: dict) -> None:
        self.records.append(record)
        with open(self.path, "w") as f:
            json.dump(self.records, f, indent=2)

    def best_psnr(self) -> float:
        return max((r.get("ema_psnr", 0.0) for r in self.records), default=0.0)


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    return (t.float().clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def save_comparison(blur: torch.Tensor, pred: torch.Tensor,
                    sharp: torch.Tensor, path: str | Path,
                    label: str = "") -> None:
    def _t(x): return tensor_to_uint8(x)
    imgs  = [_t(blur), _t(pred), _t(sharp)]
    texts = ["Blurry", "Restored", "Ground Truth"]
    h     = imgs[0].shape[0]
    strip = np.full((h + 30, sum(i.shape[1] for i in imgs) + 4, 3), 20, np.uint8)
    x = 0
    for img, txt in zip(imgs, texts):
        strip[30:30+h, x:x+img.shape[1]] = img
        cv2.putText(strip, txt, (x + 4, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        x += img.shape[1] + 2
    if label:
        cv2.putText(strip, label, (strip.shape[1] - 280, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 220, 100), 1)
    cv2.imwrite(str(path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
