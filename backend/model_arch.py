import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """Lightweight U-Net for binary damage segmentation.

    Output is a [0, 1] sigmoid map where 1 = damaged pixel.
    Designed to be trained on clean images + synthetic scratch/noise augmentation.
    """

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
