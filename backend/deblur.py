"""NeuroVision Restoration Engine - Standalone Inference Script

This single file contains the complete inference pipeline, including model
architectures for the trained networks (U-Net and NAFNet) and the logic
to load and execute both the trained and pretrained models.

Stage order:
1. Damage detection  — U-Net (trained) OR heuristic fallback
2. LaMa inpainting   — fill scratches/damage with surrounding context
3. NAFNet deblur     — blind deblur on the whole image
4. RealESRGAN-x4plus — 4× super-resolution & hallucinate textures
5. Post-processing   — bilateral → detail enhance → 3-pass unsharp → CLAHE → vibrance
"""

from __future__ import annotations

import argparse
import logging
import sys
import types
from pathlib import Path
from typing import Optional, Tuple

# Add parent directory to path so we can import RAG
sys.path.append(str(Path(__file__).resolve().parent.parent))
from RAG.similarity_search import get_image_context

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# -----------------------------------------------------------------------------
# SHIM FOR TORCHVISION / BASICSR COMPATIBILITY
# basicsr references torchvision.transforms.functional_tensor removed in 0.16.
# -----------------------------------------------------------------------------
if "torchvision.transforms.functional_tensor" not in sys.modules:
    import torchvision.transforms.functional as _tvf
    _shim = types.ModuleType("torchvision.transforms.functional_tensor")
    _shim.rgb_to_grayscale = _tvf.rgb_to_grayscale          # type: ignore
    sys.modules["torchvision.transforms.functional_tensor"] = _shim

logger = logging.getLogger(__name__)

# Constants
WEIGHTS_DIR        = Path(__file__).parent / "weights"
DAMAGE_DET_WEIGHTS = WEIGHTS_DIR / "damage_detector.pth"
NAFNET_WEIGHTS     = WEIGHTS_DIR / "deblur_nafnet.pth"

ESRGAN_URL  = "https://github.com/cszn/KAIR/releases/download/v1.0/BSRGAN.pth"

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
MAX_DIM = 1024


# =============================================================================
# MODEL ARCHITECTURES (TRAINED MODELS)
# =============================================================================

# --- 1. U-Net (Damage Detector) ---

class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DamageDetectorUNet(nn.Module):
    """Lightweight U-Net for binary damage segmentation."""
    FEATURES = [32, 64, 128, 256]

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        features = self.FEATURES

        self.encoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        ch = in_channels
        for f in features:
            self.encoders.append(_DoubleConv(ch, f))
            ch = f

        self.bottleneck = _DoubleConv(features[-1], features[-1] * 2)

        self.upconvs: nn.ModuleList = nn.ModuleList()
        self.decoders: nn.ModuleList = nn.ModuleList()
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.decoders.append(_DoubleConv(f * 2, f))

        self.head = nn.Conv2d(features[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i, (up, dec) in enumerate(zip(self.upconvs, self.decoders)):
            x = up(x)
            skip = skips[i]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return torch.sigmoid(self.head(x))


# --- 2. NAFNet (Blind Deblurring) ---

class LayerNorm2d(nn.Module):
    def __init__(self, c: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias   = nn.Parameter(torch.zeros(c))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = x.mean(1, keepdim=True)
        v = (x - m).pow(2).mean(1, keepdim=True)
        x = (x - m) / (v + self.eps).sqrt()
        return self.weight[:, None, None] * x + self.bias[:, None, None]

class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * b

class SimpleChannelAttn(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(c, c, 1, bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sca(x)

class NAFBlock(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        dw = c * 2
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.sg1   = SimpleGate()
        self.sca   = SimpleChannelAttn(dw // 2)
        self.conv3 = nn.Conv2d(dw // 2, c, 1)
        fn         = c * 2
        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, fn, 1)
        self.sg2   = SimpleGate()
        self.conv5 = nn.Conv2d(fn // 2, c, 1)
        self.beta  = nn.Parameter(torch.ones(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1, c, 1, 1))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.conv3(self.sca(self.sg1(self.conv2(self.conv1(self.norm1(inp))))))
        y = inp + x * self.beta
        x = self.conv5(self.sg2(self.conv4(self.norm2(y))))
        return y + x * self.gamma

class NAFNetDeblur(nn.Module):
    def __init__(
        self,
        img_channel:    int       = 3,
        width:          int       = 48,
        middle_blk_num: int       = 6,
        enc_blk_nums:   list[int] = None,
        dec_blk_nums:   list[int] = None,
    ) -> None:
        super().__init__()
        if enc_blk_nums is None: enc_blk_nums = [2, 2, 4, 8]
        if dec_blk_nums is None: dec_blk_nums = [2, 2, 4, 4]

        self.intro  = nn.Conv2d(img_channel, width, 3, padding=1)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs    = nn.ModuleList()
        self.ups      = nn.ModuleList()
        self.decoders = nn.ModuleList()

        ch = width
        for n in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))
            self.downs.append(nn.Conv2d(ch, ch * 2, 2, stride=2))
            ch *= 2

        self.middle_blks = nn.Sequential(*[NAFBlock(ch) for _ in range(middle_blk_num)])

        for n in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(ch, ch * 2, 1, bias=False),
                                           nn.PixelShuffle(2)))
            ch //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))

        self.padder_size = 2 ** len(enc_blk_nums)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        _, _, H, W = inp.shape
        inp_pad = self._pad(inp)
        x = self.intro(inp_pad)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape[1] != skip.shape[1]:
                c = min(x.shape[1], skip.shape[1])
                x, skip = x[:, :c], skip[:, :c]
            x = dec(x + skip)
        return (self.ending(x) + inp_pad)[:, :, :H, :W]

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        ph = (-h) % self.padder_size
        pw = (-w) % self.padder_size
        return F.pad(x, (0, pw, 0, ph))


# =============================================================================
# PIPELINE UTILS & CLASS
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

def _heuristic_scratch_mask(img: Image.Image, diff_thresh: int = 22) -> np.ndarray:
    bgr  = _pil_to_bgr(img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    m11  = cv2.medianBlur(gray, 11)
    m21  = cv2.medianBlur(gray, 21)
    diff = cv2.max(cv2.absdiff(gray, m11), cv2.absdiff(gray, m21))
    _, binary = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(binary, kernel, iterations=2).astype(np.float32) / 255.0

def _mask_to_lama(prob: np.ndarray, threshold: float = 0.4) -> Image.Image:
    binary = (prob > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return Image.fromarray(cv2.dilate(binary, kernel, iterations=2), mode="L")


class RestorePipeline:
    """Load-once, call-many restoration pipeline."""

    def __init__(self) -> None:
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        self._esrgan = None
        self._lama   = None
        self._nafnet = None
        self._damage_detector: Optional[DamageDetectorUNet] = None

        self._load_esrgan()
        self._load_lama()
        self._load_nafnet()
        self._load_damage_detector()

    @staticmethod
    def _fetch(url: str, name: str, min_bytes: int) -> str:
        from basicsr.utils.download_util import load_file_from_url
        dest = WEIGHTS_DIR / name
        if dest.exists() and dest.stat().st_size < min_bytes:
            logger.warning("%s looks incomplete — re-downloading.", name)
            dest.unlink()
        return load_file_from_url(url=url, model_dir=str(WEIGHTS_DIR),
                                  progress=True, file_name=name)

    def _load_esrgan(self) -> None:
        """Load BSRGAN for 4× super-resolution."""
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            ep = self._fetch(ESRGAN_URL, "BSRGAN.pth", 60 * 1024 * 1024)
            
            # BSRGAN uses old architecture key names. Convert to new basicsr names.
            ep_patched = str(Path(ep).with_name("BSRGAN_mapped.pth"))
            if not Path(ep_patched).exists():
                ckpt = torch.load(ep, map_location="cpu", weights_only=False)
                new_ckpt = {}
                for k, v in ckpt.items():
                    nk = k.replace("RRDB_trunk.", "body.")
                    nk = nk.replace("RDB", "rdb")
                    nk = nk.replace("trunk_conv.", "conv_body.")
                    nk = nk.replace("upconv1.", "conv_up1.")
                    nk = nk.replace("upconv2.", "conv_up2.")
                    nk = nk.replace("HRconv.", "conv_hr.")
                    new_ckpt[nk] = v
                # RealESRGANer expects 'params' key
                torch.save({"params": new_ckpt}, ep_patched)

            rrdb = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                           num_block=23, num_grow_ch=32, scale=4)
            self._esrgan = RealESRGANer(
                scale=4, model_path=ep_patched, model=rrdb,
                tile=512, tile_pad=16, pre_pad=0,
                half=(DEVICE == "cuda"), device=DEVICE,
            )
            logger.info("BSRGAN (4x SR) loaded on %s", DEVICE)
        except Exception as exc:
            logger.error("Could not load ESRGAN: %s", exc)
            raise RuntimeError(f"Restoration model init failed: {exc}") from exc

    def _load_lama(self) -> None:
        try:
            from simple_lama_inpainting import SimpleLama
            self._lama = SimpleLama()
            logger.info("LaMa inpainting loaded")
        except Exception as exc:
            logger.warning("LaMa unavailable (%s) — inpainting skipped", exc)

    def _load_nafnet(self) -> None:
        if not NAFNET_WEIGHTS.exists():
            logger.info("NAFNet weights absent — deblur step skipped.")
            return
        try:
            ckpt = torch.load(str(NAFNET_WEIGHTS), map_location="cpu", weights_only=False)
            cfg  = ckpt.get("cfg", {})
            model = NAFNetDeblur(
                width          = cfg.get("width",          32),
                middle_blk_num = cfg.get("middle_blk_num",  4),
                enc_blk_nums   = cfg.get("enc_blk_nums",   [2, 2, 2, 4]),
                dec_blk_nums   = cfg.get("dec_blk_nums",   [2, 2, 2, 2]),
            )
            sd = ckpt.get("ema") or ckpt.get("model")
            model.load_state_dict(sd)
            model.eval().to(DEVICE)
            self._nafnet = model
            logger.info("NAFNet deblur loaded on %s", DEVICE)
        except Exception as exc:
            logger.warning("NAFNet load failed (%s) — deblur step skipped", exc)

    def _load_damage_detector(self) -> None:
        if not DAMAGE_DET_WEIGHTS.exists():
            logger.info("U-Net weights absent — heuristic scratch detection.")
            return
        try:
            m = DamageDetectorUNet(in_channels=3).to(DEVICE)
            m.load_state_dict(torch.load(DAMAGE_DET_WEIGHTS,
                                         map_location=DEVICE, weights_only=True))
            m.eval()
            self._damage_detector = m
            logger.info("U-Net damage detector loaded")
        except Exception as exc:
            logger.warning("Damage detector load failed (%s) — heuristic fallback", exc)

    def detect_damage(self, img: Image.Image) -> np.ndarray:
        if self._damage_detector is not None:
            from torchvision import transforms
            tf = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
            t = tf(img.convert("RGB")).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                prob = self._damage_detector(t)[0, 0].cpu().numpy()
            return cv2.resize(prob, (img.width, img.height),
                              interpolation=cv2.INTER_LINEAR)
        return _heuristic_scratch_mask(img)

    def _inpaint(self, img: Image.Image, prob: np.ndarray) -> Image.Image:
        if self._lama is None:
            return img
        mask = _mask_to_lama(prob)
        if mask.size != img.size:
            mask = mask.resize(img.size, Image.NEAREST)
        return self._lama(img.convert("RGB"), mask)

    def _deblur(self, img: Image.Image) -> Image.Image:
        if self._nafnet is None:
            return img
        try:
            arr    = np.array(img.convert("RGB")).astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
            H, W   = tensor.shape[2], tensor.shape[3]
            TILE, OVR = 512, 32

            if H <= TILE and W <= TILE:
                with torch.no_grad():
                    out = self._nafnet(tensor.to(DEVICE)).clamp(0, 1).cpu()
            else:
                out = self._deblur_tiled(tensor, TILE, OVR)

            out_np = (out[0].numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)

            drift = float(np.abs(out_np.astype(np.float32).mean(axis=(0, 1))
                                 - arr.mean(axis=(0, 1)) * 255).max())
            if drift > 50:
                logger.warning("NAFNet quality gate triggered (drift=%.1f) — deblur skipped", drift)
                return img

            logger.info("NAFNet deblur applied (max channel drift %.1f)", drift)
            return Image.fromarray(out_np)
        except Exception as exc:
            logger.warning("NAFNet deblur error: %s — skipping", exc)
            return img

    def _deblur_tiled(self, t: torch.Tensor, tile: int, ovr: int) -> torch.Tensor:
        _, _, H, W = t.shape
        stride = tile - ovr
        out    = torch.zeros_like(t)
        wt     = torch.zeros(1, 1, H, W)
        ramp   = torch.linspace(0, 1, ovr)
        ones   = torch.ones(tile - 2 * ovr)
        row    = torch.cat([ramp, ones, ramp.flip(0)])[:tile]
        kern   = row.unsqueeze(0) * row.unsqueeze(1)
        ys = sorted(set(range(0, H - tile + 1, stride)) | {max(H - tile, 0)})
        xs = sorted(set(range(0, W - tile + 1, stride)) | {max(W - tile, 0)})
        td = t.to(DEVICE)
        with torch.no_grad():
            for y in ys:
                for x in xs:
                    y2, x2 = min(y + tile, H), min(x + tile, W)
                    r = self._nafnet(td[:, :, y:y2, x:x2]).clamp(0, 1).cpu()
                    k = kern[:y2 - y, :x2 - x]
                    out[:, :, y:y2, x:x2] += r * k
                    wt [:, :, y:y2, x:x2] += k
        return (out / wt.clamp(min=1e-8)).clamp(0, 1)

    def _run_esrgan(self, img: Image.Image, outscale: float = 4.0) -> Image.Image:
        bgr = _pil_to_bgr(img)
        try:
            enhanced, _ = self._esrgan.enhance(bgr, outscale=outscale)
            return _bgr_to_pil(enhanced)
        except Exception as exc:
            logger.warning("ESRGAN failed: %s — returning input", exc)
            return _bgr_to_pil(bgr)

    @staticmethod
    def _add_film_grain(bgr: np.ndarray, intensity: float = 0.035) -> np.ndarray:
        """Injects microscopic photographic film grain to eliminate the 'plastic' AI look.
        
        This tricks the human eye into perceiving the image as a real, organic photograph
        rather than an over-smoothed digital painting.
        """
        h, w, c = bgr.shape
        # Generate Gaussian noise
        noise = np.random.normal(0, 1, (h, w, c)).astype(np.float32)
        # Slightly blur the noise to simulate analog film crystals instead of sharp digital noise
        noise = cv2.GaussianBlur(noise, (0, 0), 0.8)
        
        # Scale noise and add it to the image
        noisy_bgr = bgr.astype(np.float32) + (noise * 255 * intensity)
        return np.clip(noisy_bgr, 0, 255).astype(np.uint8)

    @staticmethod
    def _post_process(img: Image.Image, style: str = "default") -> Image.Image:
        """Context-Aware post-processing for photorealistic background sharpness.
        """
        original_bgr = _pil_to_bgr(img)
        bgr = original_bgr.copy()

        # 1. Bilateral — smooth noise while keeping edges razor-sharp
        bgr = cv2.bilateralFilter(bgr, d=7, sigmaColor=40, sigmaSpace=40)

        # Fix 2: Adaptive sharpening based on image blurriness
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        if lap_var < 80:
            sharpness = 1.4 if style == "blur" else 1.15
        elif lap_var < 150:
            sharpness = 1.25 if style == "blur" else 1.10
        else:
            sharpness = 1.05

        b1  = cv2.GaussianBlur(bgr, (0, 0), 1.0)
        bgr = cv2.addWeighted(bgr, sharpness, b1, -(sharpness - 1.0), 0)
        bgr = np.clip(bgr, 0, 255).astype(np.uint8)

        # Fix 3: Lower CLAHE
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe_clip = 1.5 if style == "low_light" else 1.2
        l   = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8)).apply(l)
        bgr = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        # Fix 4: Reduce vibrance
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.02, 0, 255)
        bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Fix 6: Blending (Mix original and processed to keep natural look)
        bgr = cv2.addWeighted(original_bgr, 0.4, bgr, 0.6, 0)

        # Fix 5: Make film grain conditional
        grain = 0.0 if style == "face" else 0.02
        bgr = RestorePipeline._add_film_grain(bgr, intensity=grain)

        return _bgr_to_pil(bgr)

    def restore(
        self, img: Image.Image
    ) -> Tuple[Image.Image, Image.Image, Image.Image, Optional[np.ndarray]]:
        img = _fit(img)

        # Stage 1 — scratch / damage detection
        damage_map = self.detect_damage(img)

        # Stage 2 — LaMa inpainting
        inpainted = self._inpaint(img, damage_map)

        # Stage 3 — NAFNet blind deblur
        deblurred = self._deblur(inpainted)
        trained_output = deblurred

        # Stage 4 — RealESRGAN-x4plus (4× SR)
        sharp = self._run_esrgan(deblurred, outscale=4.0)
        pretrained_output = sharp

        damage_map_upscaled = cv2.resize(
            damage_map, (sharp.width, sharp.height),
            interpolation=cv2.INTER_LINEAR
        )

        # RAG Context retrieval
        logger.info("Querying RAG for image context...")
        style = get_image_context(img)
        logger.info(f"RAG identified image style as: {style}")

        # Stage 5 — context-aware post-processing
        final = self._post_process(sharp, style=style)

        trained_output    = _resize_to_match(trained_output,    final)
        pretrained_output = _resize_to_match(pretrained_output, final)

        return trained_output, pretrained_output, final, damage_map_upscaled


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run the NeuroVision restoration pipeline (Standalone file)."
    )
    parser.add_argument("input_image", help="Path to the input image file.")
    parser.add_argument("--output_dir", default="outputs", help="Directory to save the outputs.")

    args = parser.parse_args()
    input_path = Path(args.input_image)

    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found.")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading models... (This may take a moment, especially on first run)")
    try:
        pipeline = RestorePipeline()
    except Exception as e:
        print(f"Failed to load models: {e}")
        sys.exit(1)

    print(f"\nReading image: {input_path}")
    try:
        img = Image.open(input_path).convert("RGB")
    except Exception as e:
        print(f"Failed to read image: {e}")
        sys.exit(1)

    print("Running restoration pipeline... Please wait.")
    try:
        trained_out, pretrained_out, final_out, damage_map = pipeline.restore(img)
    except Exception as e:
        print(f"Restoration failed: {e}")
        sys.exit(1)

    # Save outputs
    base_name       = input_path.stem
    trained_path    = out_dir / f"{base_name}_1_trained.png"
    pretrained_path = out_dir / f"{base_name}_2_pretrained.png"
    final_path      = out_dir / f"{base_name}_3_final.png"
    mask_path       = out_dir / f"{base_name}_mask.png"

    trained_out.save(trained_path)
    pretrained_out.save(pretrained_path)
    final_out.save(final_path)

    print(f"\n✅ Outputs successfully saved to '{out_dir}':")
    print(f"  - Trained Models Output:    {trained_path}")
    print(f"  - Pretrained Models Output: {pretrained_path}")
    print(f"  - Final Processed Output:   {final_path}")

    if damage_map is not None:
        heat = (damage_map * 255).astype(np.uint8)
        colored_bgr = cv2.applyColorMap(heat, cv2.COLORMAP_HOT)
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
        Image.fromarray(colored_rgb).save(mask_path)
        print(f"  - Damage Map:               {mask_path}")

    print("\n" + "=" * 60)
    print("                  MODELS USED IN PIPELINE")
    print("=" * 60)
    print("TRAINED MODELS:")
    print("  1. U-Net  (Damage/Scratch Detection)")
    print("  2. NAFNet (Blind Image Deblurring)")
    print()
    print("PRETRAINED MODELS:")
    print("  1. LaMa (SimpleLama)   - Fast image inpainting for filling")
    print("                           scratches detected by the mask.")
    print("  2. BSRGAN              - Uses RRDB-23 blocks for 4x Super-")
    print("                           Resolution. Preserves organic textures")
    print("                           better than standard RealESRGAN.")
    print("=" * 60)


if __name__ == "__main__":
    main()
