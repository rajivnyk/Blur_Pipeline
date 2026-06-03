"""FastAPI backend for the face-aware photo restoration system.

POST /predict
  - Accepts a multipart image upload (JPEG, PNG, WebP)
  - Returns JSON:
    {
      trained:    "data:image/png;base64,...",  # after trained models (U-Net + NAFNet)
      pretrained: "data:image/png;base64,...",  # after RealESRGAN pretrained SR
      final:      "data:image/png;base64,...",  # fully processed (all stages)
      mask:       "data:image/png;base64,..." | null
    }

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_pipeline: RestorePipeline | None = None

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


def _encode_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _mask_to_data_url(mask: np.ndarray) -> str:
    """Convert float32 [0,1] damage map to a HOT-colormap PNG data URL."""
    heat = (mask * 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(heat, cv2.COLORMAP_HOT)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    return _encode_png(Image.fromarray(colored_rgb))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _pipeline
    logger.info("Loading restoration models…")
    _pipeline = RestorePipeline()
    logger.info("Models ready — server accepting requests")
    yield
    _pipeline = None
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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "damage_detector": _pipeline is not None and _pipeline._damage_detector is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if _pipeline is None:
        raise HTTPException(503, detail="Models not loaded yet — try again shortly")

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
        img = Image.open(io.BytesIO(raw))
    except Exception:
        raise HTTPException(400, detail="Could not decode image — file may be corrupt")

    try:
        trained_out, pretrained_out, final, mask = _pipeline.restore(img)
    except Exception as exc:
        logger.exception("Restoration failed")
        raise HTTPException(500, detail=f"Restoration pipeline error: {exc}")

    return JSONResponse({
        "trained":    _encode_png(trained_out),
        "pretrained": _encode_png(pretrained_out),
        "final":      _encode_png(final),
        "mask":       _mask_to_data_url(mask) if mask is not None else None,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
