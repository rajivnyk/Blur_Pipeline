"""
Evaluation Script — PSNR / SSIM / LPIPS / Edge Sharpness
===========================================================
Evaluates a trained checkpoint on the GoPro test split and prints a
full metric table. Results are saved to evaluate_results.json.

Usage
-----
  python evaluate.py --checkpoint /kaggle/working/deblur_out/checkpoints/best.pth
  python evaluate.py --checkpoint best.pth --data /kaggle/input/gopro-...

Dependencies
------------
  pip install lpips          # for LPIPS metric
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset   import GoproDataset
from inference import load_model, restore_image
from utils     import AverageMeter, calculate_psnr, laplacian_sharpness, tensor_to_uint8


# ── PSNR ──────────────────────────────────────────────────────────────────────

def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    return calculate_psnr(pred, target, max_val)


# ── SSIM (full image, not patched) ────────────────────────────────────────────

def ssim_image(pred: np.ndarray, target: np.ndarray) -> float:
    """
    OpenCV-based SSIM for evaluation (more accurate for full-res images).
    Both inputs: uint8 (H, W, 3) RGB.
    """
    scores = []
    for c in range(3):
        p = pred[:, :, c].astype(np.float64)
        t = target[:, :, c].astype(np.float64)

        mu_p  = cv2.GaussianBlur(p, (11, 11), 1.5)
        mu_t  = cv2.GaussianBlur(t, (11, 11), 1.5)
        mu_pp = cv2.GaussianBlur(p * p, (11, 11), 1.5) - mu_p ** 2
        mu_tt = cv2.GaussianBlur(t * t, (11, 11), 1.5) - mu_t ** 2
        mu_pt = cv2.GaussianBlur(p * t, (11, 11), 1.5) - mu_p * mu_t

        C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        num = (2 * mu_p * mu_t + C1) * (2 * mu_pt + C2)
        den = (mu_p ** 2 + mu_t ** 2 + C1) * (mu_pp + mu_tt + C2)
        scores.append((num / (den + 1e-8)).mean())

    return float(np.mean(scores))


# ── LPIPS ─────────────────────────────────────────────────────────────────────

class LPIPSMetric:
    def __init__(self, device: torch.device) -> None:
        try:
            import lpips
            self.fn     = lpips.LPIPS(net="alex").to(device)
            self.device = device
            self.ok     = True
        except ImportError:
            print("lpips not installed — LPIPS metric skipped. pip install lpips")
            self.ok = False

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        if not self.ok:
            return float("nan")
        p = (pred.clamp(0, 1) * 2 - 1).to(self.device)
        t = (target.clamp(0, 1) * 2 - 1).to(self.device)
        with torch.no_grad():
            return float(self.fn(p, t).mean().item())


# ── Edge Sharpness ────────────────────────────────────────────────────────────

def edge_sharpness(t: torch.Tensor) -> float:
    """Laplacian variance of the restored image (higher = sharper)."""
    return laplacian_sharpness(tensor_to_uint8(t[0]).astype(np.float32) / 255.0)


# ── main evaluation loop ──────────────────────────────────────────────────────

def evaluate(
    ckpt_path: str,
    data_root: str,
    tile_size: int = 512,
    overlap:   int = 32,
    save_imgs: bool = False,
    out_dir:   str  = "./eval_results",
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model  = load_model(ckpt_path, device)
    lpips_fn = LPIPSMetric(device)

    test_ds = GoproDataset(data_root, split="test", patch_size=256)
    # Use batch_size=1 to process full-resolution images
    loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    out_path = Path(out_dir)
    if save_imgs:
        out_path.mkdir(parents=True, exist_ok=True)

    psnr_m   = AverageMeter()
    ssim_m   = AverageMeter()
    lpips_m  = AverageMeter()
    sharp_m  = AverageMeter()   # restored image sharpness
    sharp_gt = AverageMeter()   # ground-truth sharpness (reference)

    per_image: list[dict] = []

    print(f"\nEvaluating {len(test_ds)} test pairs…\n")
    for i, (blur, sharp) in enumerate(loader):
        blur, sharp = blur.to(device), sharp.to(device)

        restored = restore_image(model, blur, device, tile_size, overlap)
        restored = restored.to(device)

        # Crop/align to same size (val images may be full-res → odd sizes)
        h = min(restored.shape[2], sharp.shape[2])
        w = min(restored.shape[3], sharp.shape[3])
        restored = restored[:, :, :h, :w]
        sharp    = sharp[:, :, :h, :w]

        p  = psnr(restored, sharp)
        s  = ssim_image(tensor_to_uint8(restored[0]), tensor_to_uint8(sharp[0]))
        lp = lpips_fn(restored, sharp)
        es = edge_sharpness(restored)
        eg = edge_sharpness(sharp)

        psnr_m.update(p)
        ssim_m.update(s)
        lpips_m.update(lp)
        sharp_m.update(es)
        sharp_gt.update(eg)

        per_image.append({"idx": i, "psnr": p, "ssim": s, "lpips": lp, "sharpness": es})

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1:4d}/{len(test_ds)}]  PSNR={psnr_m.avg:.2f}  "
                  f"SSIM={ssim_m.avg:.4f}  LPIPS={lpips_m.avg:.4f}")

        if save_imgs and i < 20:
            from utils import save_comparison
            save_comparison(blur[0].cpu(), restored[0].cpu(), sharp[0].cpu(),
                            out_path / f"sample_{i:04d}.png",
                            label=f"PSNR={p:.2f}dB SSIM={s:.4f}")

    # ── print results table ────────────────────────────────────────────────────
    targets = {"psnr": 33.0, "ssim": 0.95, "lpips": 0.07}
    results = {
        "psnr":       round(psnr_m.avg,  4),
        "ssim":       round(ssim_m.avg,  4),
        "lpips":      round(lpips_m.avg, 4),
        "sharpness":  round(sharp_m.avg, 2),
        "gt_sharpness": round(sharp_gt.avg, 2),
    }

    print("\n" + "═" * 55)
    print("  EVALUATION RESULTS — GoPro Test Set")
    print("═" * 55)
    rows = [
        ("PSNR (dB)",    results["psnr"],     targets["psnr"],  "≥"),
        ("SSIM",         results["ssim"],     targets["ssim"],  "≥"),
        ("LPIPS",        results["lpips"],    targets["lpips"], "≤"),
        ("Sharpness",    results["sharpness"], None,            ""),
        ("GT Sharpness", results["gt_sharpness"], None,         ""),
    ]
    for name, val, tgt, op in rows:
        if tgt is not None:
            ok = (val >= tgt) if op == "≥" else (val <= tgt)
            status = "✓ PASS" if ok else "✗ FAIL"
            print(f"  {name:<18} {val:>8.4f}  target {op}{tgt:.2f}  {status}")
        else:
            print(f"  {name:<18} {val:>8.2f}")
    print("═" * 55)

    # Save JSON
    output = {"summary": results, "per_image": per_image}
    json_path = Path(out_dir) / "evaluate_results.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {json_path}")

    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data",       default="/kaggle/input/gopro-image-deblurring-dataset")
    p.add_argument("--tile_size",  type=int, default=512)
    p.add_argument("--overlap",    type=int, default=32)
    p.add_argument("--save_imgs",  action="store_true")
    p.add_argument("--out_dir",    default="./eval_results")
    args = p.parse_args()
    evaluate(args.checkpoint, args.data, args.tile_size,
             args.overlap, args.save_imgs, args.out_dir)


if __name__ == "__main__":
    main()
