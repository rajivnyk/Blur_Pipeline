"""FastAPI backend for the NeuroVision photo restoration system.

POST /predict
  - Blur / scratch removal (U-Net + LaMa + NAFNet + BSRGAN)
  - Returns JSON: { trained, pretrained, final, mask }

POST /predict-fog
  - Fog / haze removal (DCP + BSRGAN + post-proc)
  - Returns JSON: { dehazed, sr, final }

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import base64
import gc
import io
import logging
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from deblur import RestorePipeline
from fog import FogRemovalPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── globals — loaded once at startup, reused on every request ─────────────────
_blur_pipeline: RestorePipeline | None  = None
_fog_pipeline:  FogRemovalPipeline | None = None

ALLOWED_TYPES    = {"image/jpeg", "image/png", "image/webp"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


# ── encoding helpers ──────────────────────────────────────────────────────────

def _encode_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _mask_to_data_url(mask: np.ndarray) -> str:
    heat        = (mask * 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(heat, cv2.COLORMAP_HOT)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    return _encode_png(Image.fromarray(colored_rgb))


# ── lifespan — load ALL models once at startup ────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _blur_pipeline, _fog_pipeline

    logger.info("Loading Blur Restoration pipeline…")
    _blur_pipeline = RestorePipeline()
    logger.info("Blur pipeline ready.")

    logger.info("Loading Fog Removal pipeline…")
    _fog_pipeline = FogRemovalPipeline()
    logger.info("Fog pipeline ready — server accepting requests.")

    yield  # ── server running ──────────────────────────────────────────────

    _blur_pipeline = None
    _fog_pipeline  = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="NeuroVision Restoration API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ── request helper ────────────────────────────────────────────────────────────

async def _read_image(file: UploadFile) -> Image.Image:
    """Validate + decode an uploaded image file."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            400,
            detail=f"Unsupported file type '{file.content_type}'. Use JPEG, PNG, or WebP.",
        )
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, detail="Image exceeds 20 MB limit")
    try:
        img = Image.open(io.BytesIO(raw))
        img.verify()
        return Image.open(io.BytesIO(raw))
    except Exception:
        raise HTTPException(400, detail="Could not decode image — file may be corrupt")


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":         "ok",
        "blur_pipeline":  _blur_pipeline is not None,
        "fog_pipeline":   _fog_pipeline  is not None,
        "damage_detector": (
            _blur_pipeline is not None
            and _blur_pipeline._damage_detector is not None
        ),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Blur / scratch removal pipeline (U-Net + LaMa + NAFNet + BSRGAN)."""
    if _blur_pipeline is None:
        raise HTTPException(503, detail="Blur pipeline not loaded yet — try again shortly")

    img = await _read_image(file)
    try:
        trained_out, pretrained_out, final, mask = _blur_pipeline.restore(img)
    except Exception as exc:
        logger.exception("Blur restoration failed")
        raise HTTPException(500, detail=f"Blur pipeline error: {exc}")

    return JSONResponse({
        "trained":    _encode_png(trained_out),
        "pretrained": _encode_png(pretrained_out),
        "final":      _encode_png(final),
        "mask":       _mask_to_data_url(mask) if mask is not None else None,
    })


@app.post("/predict-fog")
async def predict_fog(file: UploadFile = File(...)):
    """Fog / haze removal pipeline (DCP + BSRGAN + post-processing)."""
    if _fog_pipeline is None:
        raise HTTPException(503, detail="Fog pipeline not loaded yet — try again shortly")

    img = await _read_image(file)
    try:
        dehazed, sr_out, final = _fog_pipeline.remove_fog(img)
    except Exception as exc:
        logger.exception("Fog removal failed")
        raise HTTPException(500, detail=f"Fog pipeline error: {exc}")

    return JSONResponse({
        "dehazed": _encode_png(dehazed),
        "sr":      _encode_png(sr_out),
        "final":   _encode_png(final),
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
