"""NeuroVision — Pipeline Selector

Just run:
    python run.py

You will be asked:
  Step 1 → Choose 1 (Fog) or 2 (Blur)
  Step 2 → Type / paste the path to your image
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║              NeuroVision  Image Restoration                  ║
╠══════════════════════════════════════════════════════════════╣
║   Select the pipeline that matches your image degradation:   ║
║                                                              ║
║    1  →  Fog / Haze Removal  (AOD-Net + BSRGAN)             ║
║    2  →  Blur / Scratch Removal  (NAFNet + LaMa + BSRGAN)   ║
╚══════════════════════════════════════════════════════════════╝
"""


def _ask_pipeline() -> int:
    print(BANNER)
    while True:
        choice = input("Enter your choice [1 or 2]: ").strip()
        if choice in ("1", "2"):
            return int(choice)
        print("  ⚠️  Invalid input — please enter 1 or 2.")


def _ask_image_path(pipeline_name: str) -> Path:
    print(f"\n  You selected: {pipeline_name}")
    while True:
        raw = input("  Enter the path to your image: ").strip().strip('"').strip("'")
        p   = Path(raw)
        if p.exists() and p.is_file():
            return p
        print(f"  ⚠️  File not found: '{raw}' — please try again.")


def run_fog(input_path: Path, out_dir: Path) -> None:
    """Run the fog / haze removal pipeline."""
    from fog import FogRemovalPipeline

    print("\n🌫️  Loading Fog Removal pipeline…")
    pipeline = FogRemovalPipeline()

    img = Image.open(input_path).convert("RGB")

    print("🌫️  Removing fog… Please wait.")
    dehazed, sr_out, final = pipeline.remove_fog(img)

    base = input_path.stem
    p1   = out_dir / f"{base}_fog_1_dehazed.png"
    p2   = out_dir / f"{base}_fog_2_sr.png"
    p3   = out_dir / f"{base}_fog_3_final.png"

    dehazed.save(p1)
    sr_out.save(p2)
    final.save(p3)

    print(f"\n✅  Fog removal complete — outputs saved to '{out_dir}':")
    print(f"   [1] AOD-Net dehazed  : {p1}")
    print(f"   [2] BSRGAN 4× SR     : {p2}")
    print(f"   [3] Final (polished) : {p3}")


def run_blur(input_path: Path, out_dir: Path) -> None:
    """Run the blur / scratch removal pipeline."""
    from deblur import RestorePipeline

    print("\n🔍  Loading Blur Removal pipeline…")
    pipeline = RestorePipeline()

    img = Image.open(input_path).convert("RGB")

    print("🔍  Restoring image… Please wait.")
    trained_out, pretrained_out, final, mask = pipeline.restore(img)

    base = input_path.stem
    p1   = out_dir / f"{base}_blur_1_trained.png"
    p2   = out_dir / f"{base}_blur_2_pretrained.png"
    p3   = out_dir / f"{base}_blur_3_final.png"
    pm   = out_dir / f"{base}_blur_mask.png"

    trained_out.save(p1)
    pretrained_out.save(p2)
    final.save(p3)

    print(f"\n✅  Blur removal complete — outputs saved to '{out_dir}':")
    print(f"   [1] Trained models output    : {p1}")
    print(f"   [2] Pretrained models output : {p2}")
    print(f"   [3] Final (polished)         : {p3}")

    if mask is not None:
        import cv2
        import numpy as np
        from PIL import Image as _PIL

        heat        = (mask * 255).astype(np.uint8)
        colored_bgr = cv2.applyColorMap(heat, cv2.COLORMAP_HOT)
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
        _PIL.fromarray(colored_rgb).save(pm)
        print(f"   [4] Damage mask              : {pm}")


def main() -> None:
    # ── Step 1: choose pipeline ───────────────────────────────────────────────
    choice = _ask_pipeline()

    if choice == 1:
        label = "Fog / Haze Removal"
    else:
        label = "Blur / Scratch Removal"

    # ── Step 2: ask for the image path ───────────────────────────────────────
    input_path = _ask_image_path(label)

    # ── Output directory (sibling 'outputs/' folder next to the image) ───────
    out_dir = input_path.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 3: run ───────────────────────────────────────────────────────────
    try:
        if choice == 1:
            run_fog(input_path, out_dir)
        else:
            run_blur(input_path, out_dir)
    except Exception as exc:
        logger.exception("Pipeline failed")
        print(f"\n❌  Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
