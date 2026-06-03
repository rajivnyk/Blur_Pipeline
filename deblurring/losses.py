"""
Combined Loss — Charbonnier/L1 + SSIM + Perceptual (VGG19) + Gradient + Frequency
===================================================================================
All auxiliary losses cast inputs to float32 internally so they are safe inside
torch.amp.autocast('cuda').

FrequencyLoss: L1 on FFT magnitude — safe because we cast to float32 BEFORE
rfft2, avoiding the float16 overflow that caused NaN in the old FFT loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class L1Loss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (smooth L1 variant) used by the official NAFNet."""
    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))


class SSIMLoss(nn.Module):
    """1 − SSIM, computed in float32 for numerical stability under AMP."""

    def __init__(self, window_size: int = 11, sigma: float = 1.5) -> None:
        super().__init__()
        k = self._gauss(window_size, sigma)
        self.register_buffer("kern", k)
        self.ws = window_size

    @staticmethod
    def _gauss(size: int, sigma: float) -> torch.Tensor:
        c = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)

    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        C   = x.shape[1]
        k   = self.kern.expand(C, 1, -1, -1)
        pad = self.ws // 2
        mx  = F.conv2d(x, k, padding=pad, groups=C)
        my  = F.conv2d(y, k, padding=pad, groups=C)
        vx  = F.conv2d(x * x, k, padding=pad, groups=C) - mx * mx
        vy  = F.conv2d(y * y, k, padding=pad, groups=C) - my * my
        vxy = F.conv2d(x * y, k, padding=pad, groups=C) - mx * my
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        num = (2 * mx * my + C1) * (2 * vxy + C2)
        den = (mx * mx + my * my + C1) * (vx + vy + C2)
        return (num / den).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - self._ssim(pred.float().clamp(0, 1),
                                  target.float().clamp(0, 1))


class PerceptualLoss(nn.Module):
    """
    VGG19 feature matching at relu1_2 (idx 3) and relu2_2 (idx 8).
    VGG forward is always in float32 — large activations at deeper layers
    overflow float16 and produce NaN under AMP.
    """

    _LAYERS = [3, 8]   # relu1_2, relu2_2

    def __init__(self) -> None:
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        slices, prev = [], 0
        for end in self._LAYERS:
            slices.append(nn.Sequential(*list(vgg.children())[prev: end + 1]))
            prev = end + 1
        self.slices = nn.ModuleList(slices)
        for p in self.parameters():
            p.requires_grad_(False)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self._norm(pred.float().clamp(0, 1))
        t = self._norm(target.float().clamp(0, 1))
        loss = pred.new_zeros(1, dtype=torch.float32).squeeze()
        for s in self.slices:
            p = s(p)
            t = s(t)
            loss = loss + F.l1_loss(p, t.detach())
        return loss


class GradientLoss(nn.Module):
    """
    Sobel edge-gradient loss — strongly encourages sharp edges.
    Computed in float32; safe under AMP.
    """

    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _grad_mag(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.view(B * C, 1, H, W)
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = pred.float().clamp(0, 1)
        t = target.float().clamp(0, 1)
        return F.l1_loss(self._grad_mag(p), self._grad_mag(t))


class FrequencyLoss(nn.Module):
    """
    L1 loss on FFT magnitude spectrum.
    Promotes recovery of high-frequency detail (texture, fine edges).

    Safe under AMP: inputs are cast to float32 BEFORE rfft2, so there is no
    float16 overflow. (The old FFT loss failed because rfft2 ran inside
    autocast float16 and overflowed.)
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = pred.float().clamp(0, 1)
        t = target.float().clamp(0, 1)
        fp = torch.fft.rfft2(p, norm="ortho")
        ft = torch.fft.rfft2(t, norm="ortho")
        return F.l1_loss(torch.abs(fp), torch.abs(ft))


class CombinedLoss(nn.Module):
    """
    Dynamically constructs the loss from enabled components.

    Weights (tuned for perceptual sharpness):
        main (charb/l1)  : 1.00
        SSIM             : 0.15
        Perceptual (VGG) : 0.04
        Gradient (Sobel) : 0.10
        Frequency (FFT)  : 0.05
    """

    def __init__(
        self,
        use_ssim:  bool = True,
        use_perc:  bool = True,
        use_grad:  bool = True,
        use_freq:  bool = True,
        loss_type: str  = "charbonnier",
    ) -> None:
        super().__init__()
        self.use_ssim  = use_ssim
        self.use_perc  = use_perc
        self.use_grad  = use_grad
        self.use_freq  = use_freq
        self.loss_type = loss_type

        self.main_loss = CharbonnierLoss() if loss_type == "charbonnier" else L1Loss()
        if use_ssim: self.ssim = SSIMLoss()
        if use_perc: self.perc = PerceptualLoss()
        if use_grad: self.grad = GradientLoss()
        if use_freq: self.freq = FrequencyLoss()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        main_val = self.main_loss(pred, target)
        total    = main_val
        breakdown = {self.loss_type: main_val.item()}

        if self.use_ssim:
            v = self.ssim(pred, target)
            total = total + 0.15 * v
            breakdown["ssim"] = v.item()

        if self.use_perc:
            v = self.perc(pred, target)
            total = total + 0.04 * v
            breakdown["perc"] = v.item()

        if self.use_grad:
            v = self.grad(pred, target)
            total = total + 0.10 * v
            breakdown["grad"] = v.item()

        if self.use_freq:
            v = self.freq(pred, target)
            total = total + 0.05 * v
            breakdown["freq"] = v.item()

        return total, breakdown
