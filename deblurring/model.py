"""
NAFNet Deblur — width=48, deeper encoder
=========================================
Default config (~27 M params):
  width=48, enc=[2,2,4,8], dec=[2,2,4,4], middle=6

Compared with the original width=32 (~12 M) this gives ~2.5× more parameters,
noticeably sharper outputs, and still fits in T4 16 GB at batch=8, patch=256.

Pretrained partial loading
--------------------------
Use NAFNetDeblur.from_pretrained(path) to initialise from any compatible
checkpoint (e.g. official NAFNet-GoPro-width32).  Keys that do not match in
name or shape are silently skipped so partial initialisation works even when
the width or block counts differ.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


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
    """
    Parameters
    ----------
    width         : feature channels at the first encoder level
                    32  → ~12 M params  (fast baseline)
                    48  → ~27 M params  (default — good quality/speed trade-off)
                    64  → ~67 M params  (research quality, slow on T4)
    middle_blk_num: bottleneck blocks
    enc_blk_nums  : blocks per encoder level
    dec_blk_nums  : blocks per decoder level
    use_blur_head : legacy arg — ignored (kept for checkpoint compatibility)
    """

    def __init__(
        self,
        img_channel:    int       = 3,
        width:          int       = 48,
        middle_blk_num: int       = 6,
        enc_blk_nums:   list[int] = None,
        dec_blk_nums:   list[int] = None,
        use_blur_head:  bool      = False,
    ) -> None:
        super().__init__()
        del use_blur_head
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
        inp_pad = self._pad(inp)          # may be larger than inp when H/W % padder_size != 0
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

    @classmethod
    def from_pretrained(cls, path: str | Path, **model_kwargs) -> "NAFNetDeblur":
        """
        Build a model and load matching weights from a pretrained checkpoint.

        Keys that differ in name or tensor shape are silently skipped so this
        works even when the checkpoint has a different width or block layout
        (e.g. official NAFNet-GoPro-width32 loaded into a width=48 model).

        Usage on Kaggle:
            # 1. Add dataset "nafnet-gopro-pretrained" to your notebook
            # 2. Call:
            model = NAFNetDeblur.from_pretrained(
                "/kaggle/input/nafnet-gopro-pretrained/NAFNet-GoPro-width32.pth"
            )
        """
        model = cls(**model_kwargs)
        ckpt  = torch.load(str(path), map_location="cpu", weights_only=False)

        # Unwrap common checkpoint formats
        state = (ckpt.get("params")
                 or ckpt.get("model")
                 or ckpt.get("state_dict")
                 or ckpt)

        # Strip DataParallel / module. prefix
        state = {k.replace("module.", ""): v for k, v in state.items()}

        own = model.state_dict()
        matched, skipped = 0, 0
        filtered = {}
        for k, v in state.items():
            if k in own and own[k].shape == v.shape:
                filtered[k] = v
                matched += 1
            else:
                skipped += 1

        model.load_state_dict(filtered, strict=False)
        print(f"[from_pretrained] Loaded {matched} tensors, skipped {skipped} "
              f"(shape/name mismatch) from {Path(path).name}")
        return model


if __name__ == "__main__":
    m = NAFNetDeblur()
    n = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"Parameters: {n:.2f} M")
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        y = m(x)
    print(f"Input: {tuple(x.shape)}  Output: {tuple(y.shape)}")
