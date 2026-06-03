"""
Training Script — NAFNet-48 on GoPro (Kaggle T4)
=================================================
Target: ~33-35 dB PSNR after 100 epochs with sharp perceptual quality.

Model   : width=48, middle=6, enc=[2,2,4,8]  →  ~27 M params
Loss    : Charbonnier + SSIM + Perceptual + Gradient (Sobel) + Frequency (FFT)
Batch   : 8 @ patch 256×256
Steps/e : ~400 (GoPro 3214 pairs / batch 8)
Speed   : ~10-15 min/epoch on T4 (perceptual + gradient losses add overhead)

NaN fixes applied
-----------------
1. FrequencyLoss in float32  (cast before rfft2 — avoids float16 overflow)
2. VGG in float32            (deeper VGG activations overflow float16)
3. SSIM in float32           (idem)
4. NaN guard before step     (skips any batch where loss/grad is non-finite)
5. Gradient clip = 1.0       (prevents spike-driven NaN early in training)
6. AMP scaler per optimizer  (single correct scaler for no-GAN setup)

Checkpoint guarantees
---------------------
- latest.pth  : written atomically after EVERY epoch
- best.pth    : written whenever EMA PSNR improves
- epoch_NNNN  : safety copy every save_every epochs
- resume=auto : restores from latest.pth automatically on restart
"""

from __future__ import annotations

import gc
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# AMP — new API (PyTorch ≥ 2.0) with fallback to legacy
try:
    from torch.amp import GradScaler, autocast as _ac
    def autocast():
        return _ac("cuda")
    def _scaler(enabled: bool) -> GradScaler:
        return GradScaler("cuda", enabled=enabled)
except ImportError:
    from torch.cuda.amp import GradScaler, autocast  # type: ignore[assignment]
    def _scaler(enabled: bool) -> GradScaler:         # type: ignore[misc]
        return GradScaler(enabled=enabled)

from dataset import GoproDataset
from losses  import CombinedLoss
from model   import NAFNetDeblur
from utils   import (AverageMeter, EMA, TrainingLog,
                     calculate_psnr, calculate_ssim_batch,
                     save_checkpoint, save_comparison)


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

CFG = {
    # ── Paths ────────────────────────────────────────────────────────────────
    "data_root": "/kaggle/input/gopro-image-deblurring-dataset",
    "out_dir":   "/kaggle/working/deblur_out",
    "resume":    "auto",   # "auto" picks latest.pth if present, else fresh start

    # ── Pretrained init (optional) ────────────────────────────────────────────
    # To use official NAFNet-GoPro weights as a warm start:
    #   1. Add Kaggle dataset containing NAFNet-GoPro-width32.pth
    #   2. Set this path, e.g.:
    #      "pretrained": "/kaggle/input/nafnet-gopro-pretrained/NAFNet-GoPro-width32.pth"
    # Mismatched keys are silently skipped (partial load), so width mismatch is fine.
    "pretrained": None,

    # ── Model (~27 M params) ─────────────────────────────────────────────────
    "width":          48,
    "middle_blk_num": 6,
    "enc_blk_nums":   [2, 2, 4, 8],
    "dec_blk_nums":   [2, 2, 4, 4],

    # ── Training ─────────────────────────────────────────────────────────────
    "patch_size":   256,   # larger patches → better spatial context
    "batch_size":   8,     # reduced to fit wider model + 256px patches on T4
    "epochs":       100,
    "loss_type":    "charbonnier",
    "use_ssim":     True,  # structural sharpness
    "use_perc":     True,  # VGG perceptual — biggest driver of visual quality
    "use_grad":     True,  # Sobel edge loss — sharpens fine details
    "use_freq":     True,  # FFT magnitude loss — recovers high-freq texture
    "lr":           2e-4,
    "weight_decay": 1e-4,
    "grad_clip":    1.0,
    "ema_decay":    0.999,
    "T_0":          100,   # single cosine decay over full training run

    # ── Early stopping ────────────────────────────────────────────────────────
    "patience": 50,        # give the model room to breathe at 100 epochs

    # ── Dataloader ────────────────────────────────────────────────────────────
    "num_workers":    0,
    "prefetch_factor": None,

    # ── Validation ───────────────────────────────────────────────────────────
    "val_every":    2,
    "val_cap":      50,
    "val_batch":    4,

    # ── Checkpointing ────────────────────────────────────────────────────────
    "save_every":   5,
}

# ══════════════════════════════════════════════════════════════════════════════


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _auto_resume(cfg: dict) -> None:
    if cfg.get("resume") != "auto":
        return
    ckpt_dir = Path(cfg["out_dir"]) / "checkpoints"
    for name in ("latest.pth", "best.pth"):
        p = ckpt_dir / name
        if p.exists():
            cfg["resume"] = str(p)
            print(f"[auto-resume] {p}")
            return
    cfg["resume"] = None
    print("[auto-resume] No checkpoint found — starting fresh")


def _unwrap(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, nn.DataParallel) else m


# ── Dataloaders ───────────────────────────────────────────────────────────────

def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    train_ds = GoproDataset(cfg["data_root"], split="train",
                            patch_size=cfg["patch_size"])
    val_ds   = GoproDataset(cfg["data_root"], split="test",
                            patch_size=cfg["patch_size"],
                            val_patch_size=512)

    cap = cfg.get("val_cap")
    if cap and cap < len(val_ds):
        val_ds = Subset(val_ds, list(range(cap)))

    nw = cfg["num_workers"]
    pf = cfg.get("prefetch_factor", 2) if nw > 0 else None

    kw = dict(num_workers=nw, pin_memory=True,
               persistent_workers=False, prefetch_factor=pf)

    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"],
                          shuffle=True, drop_last=True, **kw)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.get("val_batch", 4),
                          shuffle=False, **kw)
    return train_dl, val_dl


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _state(epoch, model, ema, opt, scaler, best_psnr, cfg):
    return {"epoch": epoch, "model": _unwrap(model).state_dict(),
            "ema": ema.state_dict(), "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(), "best_psnr": best_psnr, "cfg": cfg}


def _save(state: dict, path: Path, tag: str) -> None:
    save_checkpoint(state, path)
    print(f"  [Checkpoint] {tag:12s} → {path.name}  (epoch {state['epoch']})")


# ── Training epoch ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, opt, scaler, device, cfg) -> dict:
    model.train()
    loss_m = AverageMeter()
    psnr_m = AverageMeter()
    nan_skipped = 0
    t0 = time.time()

    for step, (blur, sharp) in enumerate(loader):
        blur  = blur.to(device, non_blocking=True)
        sharp = sharp.to(device, non_blocking=True)

        # ── Forward ──────────────────────────────────────────────────────────
        with autocast():
            restored        = model(blur)
            loss, breakdown = criterion(restored, sharp)

        # ── NaN guard: skip batch if loss is non-finite ───────────────────────
        if not torch.isfinite(loss):
            nan_skipped += 1
            opt.zero_grad(set_to_none=True)
            if nan_skipped % 10 == 1:
                print(f"  [Step {step}] Non-finite loss — skipped "
                      f"({nan_skipped} total skips this epoch)")
            continue

        # ── Backward ─────────────────────────────────────────────────────────
        scaler.scale(loss).backward()
        scaler.unscale_(opt)

        # Skip optimizer step if gradients are non-finite (keeps scaler healthy)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        if torch.isfinite(grad_norm):
            scaler.step(opt)
        else:
            nan_skipped += 1

        scaler.update()
        opt.zero_grad(set_to_none=True)

        with torch.no_grad():
            psnr_val = calculate_psnr(restored, sharp)
        loss_m.update(loss.item(), blur.size(0))
        psnr_m.update(psnr_val, blur.size(0))

        if step % 50 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / (step + 1) * (len(loader) - step - 1)
            main_val = breakdown.get("l1", breakdown.get("charbonnier", 0.0))
            perc_val = breakdown.get("perc", 0.0)
            grad_val = breakdown.get("grad", 0.0)
            freq_val = breakdown.get("freq", 0.0)
            print(f"  [{step:4d}/{len(loader)}]  "
                  f"loss={loss_m.avg:.4f}  psnr={psnr_m.avg:.2f} dB  "
                  f"main={main_val:.4f}  perc={perc_val:.4f}  "
                  f"grad={grad_val:.4f}  freq={freq_val:.4f}  "
                  f"({elapsed:.0f}s / ETA {eta:.0f}s)")

    if nan_skipped:
        print(f"  [Epoch summary] Skipped {nan_skipped} non-finite batches")

    return {"train_loss": loss_m.avg, "train_psnr": psnr_m.avg}


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device, vis_dir=None, epoch=0) -> dict:
    model.eval()
    loss_m = AverageMeter()
    psnr_m = AverageMeter()
    ssim_m = AverageMeter()

    for i, (blur, sharp) in enumerate(loader):
        blur  = blur.to(device, non_blocking=True)
        sharp = sharp.to(device, non_blocking=True)
        out      = model(blur)
        loss, _  = criterion(out, sharp)
        psnr     = calculate_psnr(out, sharp)
        ssim     = calculate_ssim_batch(out, sharp)
        loss_m.update(loss.item())
        psnr_m.update(psnr)
        ssim_m.update(ssim)

        if vis_dir and i < 4:
            save_comparison(blur[0], out[0], sharp[0],
                            vis_dir / f"epoch{epoch:04d}_sample{i}.png",
                            label=f"PSNR={psnr:.2f}dB")

    return {"val_loss": loss_m.avg, "val_psnr": psnr_m.avg, "val_ssim": ssim_m.avg}


# ── Main ──────────────────────────────────────────────────────────────────────

def train(cfg: dict = CFG) -> None:
    # --- HARD OVERRIDES FOR KAGGLE STABILITY ---
    cfg["epochs"] = 100  # Enforce 100 epochs as requested
    
    # If using 1 GPU with a larger model, batch_size=16 will OOM. Cap it at 4.
    if cfg.get("batch_size", 16) > 4:
        cfg["batch_size"] = 4
        print("\n[SAFE MODE] Forced batch_size down to 4 to prevent GPU Out-Of-Memory!\n")

    set_seed(42)
    torch.backends.cudnn.benchmark = True

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus  = torch.cuda.device_count() if device.type == "cuda" else 0
    use_amp = device.type == "cuda"
    print(f"Device: {device}  |  GPUs: {n_gpus}  |  AMP: {use_amp}")

    out_dir  = Path(cfg["out_dir"])
    ckpt_dir = out_dir / "checkpoints"
    vis_dir  = out_dir / "visualisations"
    for d in [ckpt_dir, vis_dir]:
        d.mkdir(parents=True, exist_ok=True)

    _auto_resume(cfg)

    # ── Model ────────────────────────────────────────────────────────────────
    model_kwargs = dict(width=cfg["width"],
                        middle_blk_num=cfg["middle_blk_num"],
                        enc_blk_nums=cfg["enc_blk_nums"],
                        dec_blk_nums=cfg["dec_blk_nums"])
    if cfg.get("pretrained"):
        raw = NAFNetDeblur.from_pretrained(cfg["pretrained"], **model_kwargs)
    else:
        raw = NAFNetDeblur(**model_kwargs)
    raw = raw.to(device)
    ema   = EMA(raw, decay=cfg["ema_decay"])
    model = raw

    # torch.compile — skip if using DataParallel (causes TLS AssertionError with CUDA graphs)
    if n_gpus <= 1:
        try:
            model = torch.compile(raw, mode="reduce-overhead")
            print("torch.compile() enabled")
        except Exception:
            print("torch.compile() skipped")
    else:
        print("torch.compile() disabled (incompatible with DataParallel)")

    # DataParallel is explicitly disabled for safe mode (prevents gather-phase memory spikes)
    n_gpus = 1 # Force single GPU
    print("Safe Mode: DataParallel disabled (forced 1 GPU).")

    n_params = sum(p.numel() for p in raw.parameters()) / 1e6
    print(f"Generator: {n_params:.1f} M parameters")

    if n_params > 60:
        print("WARNING: model is larger than expected (>60 M). "
              "Check width/enc_blk_nums in CFG.")

    # ── Loss / optimiser / scheduler / scaler ─────────────────────────────────
    criterion = CombinedLoss(
        use_ssim=cfg.get("use_ssim",  True),
        use_perc=cfg.get("use_perc",  True),
        use_grad=cfg.get("use_grad",  True),
        use_freq=cfg.get("use_freq",  True),
        loss_type=cfg.get("loss_type", "charbonnier"),
    ).to(device)
    opt       = optim.AdamW(raw.parameters(),
                             lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=cfg["T_0"], T_mult=1, eta_min=1e-6)
    scaler    = _scaler(use_amp)

    # ── Dataloaders ───────────────────────────────────────────────────────────
    train_dl, val_dl = build_dataloaders(cfg)
    print(f"Train batches: {len(train_dl)}  |  Val batches: {len(val_dl)}")

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch, best_psnr = 0, 0.0
    if cfg.get("resume"):
        try:
            ck = torch.load(cfg["resume"], map_location="cpu", weights_only=False)
            raw.load_state_dict(ck["model"])
            if "optimizer" in ck: opt.load_state_dict(ck["optimizer"])
            if "ema"       in ck: ema.load_state_dict(ck["ema"])
            if "scaler"    in ck: scaler.load_state_dict(ck["scaler"])
            start_epoch = ck.get("epoch", 0)
            best_psnr   = ck.get("best_psnr", 0.0)
            print(f"Resumed epoch {start_epoch}  |  best PSNR {best_psnr:.2f} dB")
        except Exception as e:
            print(f"[resume] Failed to load checkpoint: {e}  — starting fresh")

    log          = TrainingLog(out_dir / "history.json")
    patience_ctr = 0

    # ══════════════════════════════════════════════════════════════════════════
    for epoch in range(start_epoch, cfg["epochs"]):
        lr_now = opt.param_groups[0]["lr"]
        print(f"\n{'='*68}")
        print(f"Epoch [{epoch+1}/{cfg['epochs']}]  "
              f"lr={lr_now:.2e}  GPUs={n_gpus}  batch={cfg['batch_size']}")
        print(f"{'='*68}")

        train_metrics = train_one_epoch(model, train_dl, criterion,
                                         opt, scaler, device, cfg)
        ema.update(raw)
        scheduler.step()

        # ── Always save latest.pth (atomic) ───────────────────────────────────
        st = _state(epoch + 1, model, ema, opt, scaler, best_psnr, cfg)
        _save(st, ckpt_dir / "latest.pth", "latest")

        # ── Safety copy every save_every epochs ───────────────────────────────
        if (epoch + 1) % cfg["save_every"] == 0:
            _save(st, ckpt_dir / f"epoch_{epoch+1:04d}.pth", f"epoch-{epoch+1}")

        # ── Validation ────────────────────────────────────────────────────────
        do_val = ((epoch + 1) % cfg["val_every"] == 0 or
                  (epoch + 1) == cfg["epochs"])
        if do_val:
            t_v = time.time()
            vm  = validate(ema.shadow, val_dl, criterion, device,
                           vis_dir=vis_dir, epoch=epoch + 1)
            ep  = vm["val_psnr"]

            log.append({"epoch": epoch + 1, "lr": lr_now,
                        **train_metrics, "ema_psnr": ep,
                        "ema_ssim": vm["val_ssim"]})

            print(f"\n  Train   loss={train_metrics['train_loss']:.4f}  "
                  f"psnr={train_metrics['train_psnr']:.2f} dB")
            print(f"  EMA val psnr={ep:.2f} dB  ssim={vm['val_ssim']:.4f}  "
                  f"({time.time()-t_v:.0f}s)")

            if ep > best_psnr:
                best_psnr    = ep
                patience_ctr = 0
                best_st = _state(epoch+1, model, ema, opt, scaler, best_psnr, cfg)
                _save(best_st, ckpt_dir / "best.pth", "best")
                print(f"  *** New best EMA PSNR: {best_psnr:.2f} dB ***")
            else:
                patience_ctr += cfg["val_every"]
                print(f"  No improvement ({patience_ctr}/{cfg['patience']})")

            if patience_ctr >= cfg["patience"]:
                print(f"\nEarly stopping (patience={cfg['patience']})")
                break
        else:
            print(f"  Train   loss={train_metrics['train_loss']:.4f}  "
                  f"psnr={train_metrics['train_psnr']:.2f} dB  [val skipped]")

        # Force aggressive garbage collection after every epoch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nDone.  Best EMA PSNR: {best_psnr:.2f} dB")
    print(f"Checkpoints in: {ckpt_dir}")


if __name__ == "__main__":
    train(CFG)
