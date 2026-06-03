"""
Inference Script — NAFNet Deblurring
======================================
Supports:
  - Single image inference
  - Batch folder processing
  - Tile-based processing for large / arbitrary-resolution images
  - Side-by-side comparison save (blurry | restored)
  - Optional GFPGAN face enhancement pass on restored image
  - < 2 s per image on T4 GPU (tile overlap=32, tile_size=512)

Usage
-----
  # Single image
  python inference.py --input photo.jpg --checkpoint best.pth

  # Folder
  python inference.py --input /path/to/folder --checkpoint best.pth --output /results

  # With GFPGAN face pass
  python inference.py --input photo.jpg --checkpoint best.pth --gfpgan
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from model import NAFNetDeblur
from utils import tensor_to_uint8

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device) -> NAFNetDeblur:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt.get("cfg", {})

    model = NAFNetDeblur(
        width          = cfg.get("width",          32),
        middle_blk_num = cfg.get("middle_blk_num",  4),
        enc_blk_nums   = cfg.get("enc_blk_nums",   [2, 2, 2, 4]),
        dec_blk_nums   = cfg.get("dec_blk_nums",   [2, 2, 2, 2]),
    )
    # Prefer EMA weights if available
    sd = ckpt.get("ema") or ckpt.get("model")
    model.load_state_dict(sd)
    model.eval().to(device)
    return model


# ── image I/O ─────────────────────────────────────────────────────────────────

def read_image(path: str | Path) -> torch.Tensor:
    """Load image → (1, 3, H, W) float32 tensor in [0, 1]."""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)


def write_image(tensor: torch.Tensor, path: str | Path) -> None:
    """(1, 3, H, W) or (3, H, W) → saved PNG."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    img = tensor_to_uint8(tensor)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# ── tile-based inference ──────────────────────────────────────────────────────

def infer_tile(
    model:     NAFNetDeblur,
    img:       torch.Tensor,   # (1, 3, H, W) float32
    tile_size: int = 512,
    overlap:   int = 32,
    device:    torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Process large images tile by tile to fit within GPU memory.
    Tiles overlap by `overlap` pixels to avoid border seams; outputs are
    blended with a linear-fade weight mask in the overlap zone.
    """
    _, _, H, W = img.shape
    stride = tile_size - overlap
    out    = torch.zeros_like(img)
    weight = torch.zeros(1, 1, H, W, device=device)

    # Build a 2-D linear weight kernel for smooth blending
    ramp  = torch.linspace(0, 1, overlap, device=device)
    ones  = torch.ones(tile_size - 2 * overlap, device=device)
    row   = torch.cat([ramp, ones, ramp.flip(0)])[:tile_size]
    kern  = (row.unsqueeze(0) * row.unsqueeze(1))   # (tile_size, tile_size)

    ys = list(range(0, H - tile_size + 1, stride)) + [max(H - tile_size, 0)]
    xs = list(range(0, W - tile_size + 1, stride)) + [max(W - tile_size, 0)]
    ys, xs = sorted(set(ys)), sorted(set(xs))

    img_d = img.to(device)

    with torch.no_grad():
        for y in ys:
            for x in xs:
                y2, x2 = min(y + tile_size, H), min(x + tile_size, W)
                patch  = img_d[:, :, y:y2, x:x2]

                restored = model(patch)
                if isinstance(restored, tuple):
                    restored = restored[0]

                ky = kern[:y2-y, :x2-x].to(device)
                out   [:, :, y:y2, x:x2] += restored.cpu() * ky
                weight[:, :, y:y2, x:x2] += ky

    return (out / weight.clamp(min=1e-8)).clamp(0, 1)


def restore_image(
    model:     NAFNetDeblur,
    img:       torch.Tensor,
    device:    torch.device,
    tile_size: int = 512,
    overlap:   int = 32,
) -> torch.Tensor:
    """Auto-choose direct vs tile inference based on image size."""
    _, _, H, W = img.shape
    if H <= tile_size and W <= tile_size:
        img = img.to(device)
        with torch.no_grad():
            out = model(img)
        if isinstance(out, tuple):
            out = out[0]
        return out.clamp(0, 1).cpu()
    return infer_tile(model, img, tile_size=tile_size, overlap=overlap, device=device)


# ── optional GFPGAN face pass ─────────────────────────────────────────────────

def apply_gfpgan(
    img_rgb: np.ndarray,   # uint8 (H, W, 3)
    weight:  float = 0.7,
) -> np.ndarray:
    """Enhance faces with GFPGAN on top of deblurred result."""
    try:
        from gfpgan import GFPGANer
        restorer = GFPGANer(
            model_path="GFPGANv1.4.pth",
            upscale=1,
            arch="clean",
            channel_multiplier=2,
        )
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        _, _, output = restorer.enhance(bgr, has_aligned=False,
                                        only_center_face=False,
                                        paste_back=True, weight=weight)
        return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    except ImportError:
        print("GFPGAN not installed — skipping face pass. pip install gfpgan")
        return img_rgb


# ── side-by-side comparison ───────────────────────────────────────────────────

def save_side_by_side(
    original:   np.ndarray,   # uint8 (H, W, 3)
    restored:   np.ndarray,
    out_path:   str | Path,
    psnr:       float | None = None,
) -> None:
    h = max(original.shape[0], restored.shape[0])
    if original.shape[0] != h:
        original = cv2.resize(original, (original.shape[1], h))
    if restored.shape[0] != h:
        restored = cv2.resize(restored, (restored.shape[1], h))

    divider = np.full((h, 4, 3), 200, dtype=np.uint8)
    strip   = np.concatenate([original, divider, restored], axis=1)

    # Labels
    label_h = 32
    canvas  = np.zeros((h + label_h, strip.shape[1], 3), dtype=np.uint8)
    canvas[label_h:] = strip
    cv2.putText(canvas, "Blurry Input",     (4, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    x_r = original.shape[1] + 4
    txt  = f"Restored (PSNR={psnr:.2f}dB)" if psnr else "Restored"
    cv2.putText(canvas, txt, (x_r + 4, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 220, 100), 2)

    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


# ── main entry points ──────────────────────────────────────────────────────────

def process_single(
    input_path:  str | Path,
    output_path: str | Path,
    model:       NAFNetDeblur,
    device:      torch.device,
    tile_size:   int  = 512,
    overlap:     int  = 32,
    gfpgan:      bool = False,
    compare:     bool = True,
) -> float:
    """Process one image. Returns inference time in seconds."""
    img_t = read_image(input_path)

    t0       = time.perf_counter()
    rest_t   = restore_image(model, img_t, device, tile_size, overlap)
    elapsed  = time.perf_counter() - t0

    rest_np = tensor_to_uint8(rest_t[0])
    if gfpgan:
        rest_np = apply_gfpgan(rest_np)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_image(rest_t, out_path)

    if compare:
        orig_np = tensor_to_uint8(img_t[0])
        save_side_by_side(orig_np, rest_np, out_path.with_name(out_path.stem + "_compare.png"))

    print(f"  {Path(input_path).name}  →  {out_path.name}  ({elapsed:.2f}s)")
    return elapsed


def process_folder(
    input_dir:  str | Path,
    output_dir: str | Path,
    model:      NAFNetDeblur,
    device:     torch.device,
    tile_size:  int  = 512,
    overlap:    int  = 32,
    gfpgan:     bool = False,
) -> None:
    in_dir  = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = [p for p in in_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    print(f"Found {len(paths)} images in {in_dir}")

    total = 0.0
    for p in paths:
        rel    = p.relative_to(in_dir)
        out_p  = out_dir / rel.with_suffix(".png")
        total += process_single(p, out_p, model, device, tile_size, overlap, gfpgan)

    print(f"\nDone. Total time: {total:.1f}s  |  Avg: {total/max(len(paths),1):.2f}s/img")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="NAFNet deblurring inference")
    parser.add_argument("--input",      required=True, help="Image file or folder")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pth")
    parser.add_argument("--output",     default="./restored", help="Output path/dir")
    parser.add_argument("--tile_size",  type=int, default=512)
    parser.add_argument("--overlap",    type=int, default=32)
    parser.add_argument("--gfpgan",     action="store_true",
                        help="Run GFPGAN face enhancement after deblurring")
    parser.add_argument("--no_compare", action="store_true",
                        help="Skip side-by-side comparison image")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.checkpoint} on {device}…")
    model  = load_model(args.checkpoint, device)

    inp = Path(args.input)
    if inp.is_dir():
        process_folder(inp, args.output, model, device,
                       args.tile_size, args.overlap, args.gfpgan)
    elif inp.suffix.lower() in IMG_EXTS:
        out = Path(args.output)
        if out.is_dir():
            out = out / (inp.stem + "_restored.png")
        process_single(inp, out, model, device,
                       args.tile_size, args.overlap,
                       args.gfpgan, not args.no_compare)
    else:
        print(f"Unknown input: {inp}")


if __name__ == "__main__":
    main()
