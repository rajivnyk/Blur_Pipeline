"""NeuroVision Fog Removal Pipeline (Pretrained-Only)

Pipeline stages:
  1. DCP Dehazing   — Dark Channel Prior (He et al., CVPR 2009).
                      Classical, fast (~1-2 s), zero download, excellent results.
  2. BSRGAN 4× SR   — weights already in weights/; recovers detail haze destroyed.
  3. Post-processing — CLAHE, bilateral filter, adaptive unsharp, saturation boost.

Usage (CLI):
    python fog.py  foggy_image.jpg  [--output_dir outputs]

Exports FogRemovalPipeline, used by main.py.
"""

from __future__ import annotations

import argparse
import logging
import sys
import types
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image

# ── torchvision shim ─────────────────────────────────────────────────────────
if "torchvision.transforms.functional_tensor" not in sys.modules:
    import torchvision.transforms.functional as _tvf
    _shim = types.ModuleType("torchvision.transforms.functional_tensor")
    _shim.rgb_to_grayscale = _tvf.rgb_to_grayscale          # type: ignore
    sys.modules["torchvision.transforms.functional_tensor"] = _shim

logger = logging.getLogger(__name__)

WEIGHTS_DIR = Path(__file__).parent / "weights"
ESRGAN_URL  = "https://github.com/cszn/KAIR/releases/download/v1.0/BSRGAN.pth"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MAX_DIM     = 1024


# =============================================================================
# HELPER UTILITIES
# =============================================================================

def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def _bgr_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

def _fit(img: Image.Image, max_dim: int = MAX_DIM) -> Image.Image:
    w, h  = img.size
    scale = min(max_dim / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img

def _resize_to_match(img: Image.Image, target: Image.Image) -> Image.Image:
    if img.size != target.size:
        return img.resize(target.size, Image.LANCZOS)
    return img


# =============================================================================
# STAGE 1 — DARK CHANNEL PRIOR DEHAZING
# =============================================================================

def _guided_filter(guide: np.ndarray, src: np.ndarray, r: int = 40, eps: float = 1e-3) -> np.ndarray:
    """Guided filter using cv2.blur (normalized box filter) — no contrib needed.

    cv2.blur is a built-in mean filter that always preserves the input shape,
    so there are no shape-mismatch issues.
    """
    ksize = 2 * r + 1

    def box(x: np.ndarray) -> np.ndarray:
        return cv2.blur(x.astype(np.float32), (ksize, ksize))

    mean_I  = box(guide)
    mean_p  = box(src)
    mean_Ip = box(guide * src)
    cov_Ip  = mean_Ip - mean_I * mean_p
    var_I   = box(guide * guide) - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    return np.clip(box(a) * guide + box(b), 0.0, 1.0).astype(np.float32)


def dcp_dehaze(img: Image.Image, patch: int = 15, omega: float = 0.95) -> Image.Image:
    """Dark Channel Prior dehazing — fast, no model download required.

    He et al., "Single Image Haze Removal Using Dark Channel Prior", CVPR 2009.
    Returns a PIL Image with haze removed.
    """
    bgr   = _pil_to_bgr(img)
    bgr_f = bgr.astype(np.float32) / 255.0

    # ── 1. Dark channel ──────────────────────────────────────────────────────
    dark = cv2.erode(bgr_f.min(axis=2),
                     np.ones((patch, patch), np.uint8))

    # ── 2. Atmospheric light A ───────────────────────────────────────────────
    flat = dark.flatten()
    n    = max(1, flat.size // 1000)          # top 0.1 %
    idx  = np.argpartition(flat, -n)[-n:]    # fast partial sort
    top_pixels = bgr_f.reshape(-1, 3)[idx]
    A    = np.mean(top_pixels, axis=0)       # 3-channel atmospheric light [B, G, R]
    A    = np.clip(A, 0.3, 1.0)              # Avoid div-by-zero or extreme values

    # ── 3. Raw transmission t = 1 - ω · dark / A ────────────────────────────
    norm_bgr = bgr_f / A
    dark_norm = cv2.erode(norm_bgr.min(axis=2),
                          np.ones((patch, patch), np.uint8))
    t_raw = np.clip(1.0 - omega * dark_norm, 0.1, 1.0).astype(np.float32)

    # ── 4. Guided-filter refine (pure NumPy, no ximgproc) ───────────────────
    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    t_ref = _guided_filter(gray, t_raw, r=40, eps=1e-3)

    # ── 5. Recover radiance J = (I - A) / t + A ─────────────────────────────
    t3 = t_ref[:, :, np.newaxis]
    J  = np.clip((bgr_f - A) / t3 + A, 0.0, 1.0)

    return _bgr_to_pil((J * 255).astype(np.uint8))


# =============================================================================
# FOG REMOVAL PIPELINE
# =============================================================================

class FogRemovalPipeline:
    """Load-once, call-many fog/haze removal pipeline.

    Stage 1 — DCP dehazing  (fast classical, no download)
    Stage 2 — BSRGAN 4× SR  (weights already present in weights/)
    Stage 3 — Post-processing (CLAHE + bilateral + unsharp + saturation)
    """

    def __init__(self) -> None:
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        self._esrgan = None
        self._load_esrgan()

    # ── model loader ─────────────────────────────────────────────────────────

    def _load_esrgan(self) -> None:
        """Load BSRGAN for 4× SR — reuses the weight files from deblur.py."""
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            bsrgan_raw    = WEIGHTS_DIR / "BSRGAN.pth"
            bsrgan_mapped = WEIGHTS_DIR / "BSRGAN_mapped.pth"

            if not bsrgan_raw.exists():
                from basicsr.utils.download_util import load_file_from_url
                load_file_from_url(
                    url=ESRGAN_URL, model_dir=str(WEIGHTS_DIR),
                    progress=True, file_name="BSRGAN.pth",
                )

            if not bsrgan_mapped.exists():
                ckpt = torch.load(str(bsrgan_raw), map_location="cpu", weights_only=False)
                new_ckpt = {}
                for k, v in ckpt.items():
                    nk = k.replace("RRDB_trunk.", "body.")
                    nk = nk.replace("RDB",         "rdb")
                    nk = nk.replace("trunk_conv.", "conv_body.")
                    nk = nk.replace("upconv1.",    "conv_up1.")
                    nk = nk.replace("upconv2.",    "conv_up2.")
                    nk = nk.replace("HRconv.",     "conv_hr.")
                    new_ckpt[nk] = v
                torch.save({"params": new_ckpt}, str(bsrgan_mapped))

            rrdb = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                           num_block=23, num_grow_ch=32, scale=4)
            self._esrgan = RealESRGANer(
                scale=4, model_path=str(bsrgan_mapped), model=rrdb,
                tile=512, tile_pad=16, pre_pad=0,
                half=(DEVICE == "cuda"), device=DEVICE,
            )
            logger.info("BSRGAN (4× SR) loaded on %s", DEVICE)
        except Exception as exc:
            logger.warning("BSRGAN could not be loaded (%s) — SR step skipped.", exc)
            self._esrgan = None

    # ── inference stages ─────────────────────────────────────────────────────

    def _run_esrgan(self, img: Image.Image, outscale: float = 4.0) -> Image.Image:
        if self._esrgan is None:
            return img
        bgr = _pil_to_bgr(img)
        try:
            enhanced, _ = self._esrgan.enhance(bgr, outscale=outscale)
            return _bgr_to_pil(enhanced)
        except Exception as exc:
            logger.warning("BSRGAN failed (%s) — SR skipped.", exc)
            return _bgr_to_pil(bgr)

    @staticmethod
    def _post_process(img: Image.Image) -> Image.Image:
        """Fog-specific post-processing:
          bilateral → CLAHE → adaptive unsharp → saturation boost → natural blend.
        """
        original_bgr = _pil_to_bgr(img)
        bgr = original_bgr.copy()

        # 1. Bilateral denoise
        bgr = cv2.bilateralFilter(bgr, d=9, sigmaColor=55, sigmaSpace=55)

        # 2. CLAHE on L channel
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # Reduced clipLimit for a more natural, less dramatic contrast
        l   = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(l)
        bgr = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        # 3. Adaptive unsharp masking
        gray    = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Softer sharpening for a more natural look
        strength = 1.30 if lap_var < 80 else (1.15 if lap_var < 200 else 1.05)
        blurred  = cv2.GaussianBlur(bgr, (0, 0), 1.2)
        bgr      = cv2.addWeighted(bgr, strength, blurred, -(strength - 1.0), 0)
        bgr      = np.clip(bgr, 0, 255).astype(np.uint8)

        # 4. Saturation restore (fog washes colours out)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        # Gentle 5% saturation boost instead of 18% to avoid unnatural neon colors
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.05, 0, 255)
        bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # 5. Blend: 60 % processed + 40 % original (natural look)
        bgr = cv2.addWeighted(original_bgr, 0.40, bgr, 0.60, 0)

        return _bgr_to_pil(bgr)

    # ── public API ────────────────────────────────────────────────────────────

    def remove_fog(self, img: Image.Image) -> Tuple[Image.Image, Image.Image, Image.Image]:
        """Run the full fog-removal pipeline.

        Returns:
            dehazed  — after DCP dehazing (stage 1)
            sr_out   — after BSRGAN 4× SR (stage 2)
            final    — after fog-aware post-processing (stage 3)
        """
        img = _fit(img)

        logger.info("Stage 1: DCP dehazing…")
        dehazed = dcp_dehaze(img)

        logger.info("Stage 2: BSRGAN 4× super-resolution…")
        sr_out = self._run_esrgan(dehazed, outscale=4.0)

        logger.info("Stage 3: Post-processing…")
        final = self._post_process(sr_out)

        dehazed = _resize_to_match(dehazed, final)
        sr_out  = _resize_to_match(sr_out,  final)

        return dehazed, sr_out, final


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="NeuroVision Fog Removal Pipeline."
    )
    parser.add_argument("input_image", help="Path to the foggy input image.")
    parser.add_argument("--output_dir", default="outputs",
                        help="Directory to save outputs (default: outputs/).")
    args = parser.parse_args()

    input_path = Path(args.input_image)
    if not input_path.exists():
        print(f"❌  Input file '{input_path}' not found.")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading fog-removal models…")
    try:
        pipeline = FogRemovalPipeline()
    except Exception as exc:
        print(f"❌  Failed to load models: {exc}")
        sys.exit(1)

    print(f"\nReading image: {input_path}")
    try:
        img = Image.open(input_path).convert("RGB")
    except Exception as exc:
        print(f"❌  Could not open image: {exc}")
        sys.exit(1)

    print("Running fog removal pipeline… Please wait.")
    try:
        dehazed, sr_out, final = pipeline.remove_fog(img)
    except Exception as exc:
        print(f"❌  Pipeline error: {exc}")
        sys.exit(1)

    base = input_path.stem
    p1   = out_dir / f"{base}_1_dehazed.png"
    p2   = out_dir / f"{base}_2_sr.png"
    p3   = out_dir / f"{base}_3_final.png"

    dehazed.save(p1)
    sr_out.save(p2)
    final.save(p3)

    print(f"\n✅  Outputs saved to '{out_dir}':")
    print(f"   Stage 1 – DCP dehazed  : {p1}")
    print(f"   Stage 2 – BSRGAN 4× SR : {p2}")
    print(f"   Stage 3 – Final         : {p3}")
    print("\n" + "=" * 60)
    print("             MODELS USED IN FOG PIPELINE")
    print("=" * 60)
    print("  1. DCP   — Dark Channel Prior (classical, no download)")
    print("  2. BSRGAN — RRDB-23 4× Super-Resolution")
    print("  3. Post-proc — CLAHE + bilateral + unsharp + saturation")
    print("=" * 60)


if __name__ == "__main__":
    main()
