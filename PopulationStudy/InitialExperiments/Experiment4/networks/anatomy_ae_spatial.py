"""Spatial-bottleneck anatomy AE + random cube masking.

Differs from:
  - UNet AE: no skip connections (blocks identity copy)
  - vector bottleneck: keeps a 4³×C map instead of GAP→32 (more capacity)

Train-time: zero out random cubes in the CT so the encoder must use context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.anatomy_ae import ResidualBlock, Down


class SpatialBottleneckEncoder(nn.Module):
    def __init__(self, bottleneck_ch=16):
        super().__init__()
        self.enc1 = ResidualBlock(1, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 32)
        self.down4 = Down(32, 64)
        self.down5 = Down(64, 64)
        self.squeeze = nn.Conv3d(64, bottleneck_ch, kernel_size=1)
        self.bottleneck_ch = bottleneck_ch

    def forward(self, x):
        h = self.enc1(x)
        h = self.down2(h)
        h = self.down3(h)
        h = self.down4(h)
        h = self.down5(h)
        return self.squeeze(h)


class SpatialBottleneckDecoder(nn.Module):
    def __init__(self, bottleneck_ch=16):
        super().__init__()
        self.expand = ResidualBlock(bottleneck_ch, 64)
        self.up4 = ResidualBlock(64, 64)
        self.up3 = ResidualBlock(64, 32)
        self.up2 = ResidualBlock(32, 32)
        self.up1 = ResidualBlock(32, 16)
        self.out = nn.Conv3d(16, 1, kernel_size=3, padding=1)

    def forward(self, z, size):
        x = self.expand(z)
        for block in (self.up4, self.up3, self.up2, self.up1):
            x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=True)
            x = block(x)
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=True)
        return self.out(x)


class SpatialBottleneckAE(nn.Module):
    def __init__(self, bottleneck_ch=16):
        super().__init__()
        self.encoder = SpatialBottleneckEncoder(bottleneck_ch=bottleneck_ch)
        self.decoder = SpatialBottleneckDecoder(bottleneck_ch=bottleneck_ch)

    def forward(self, ct):
        z = self.encoder(ct)
        return self.decoder(z, size=ct.shape[2:])


def random_cube_mask(ct, n_cubes=4, cube_frac=0.25):
    """Zero random cubes in-place on a clone. ct: (B,1,D,H,W)."""
    out = ct.clone()
    b, _, d, h, w = ct.shape
    side = max(4, int(round(min(d, h, w) * cube_frac)))
    for i in range(b):
        for _ in range(n_cubes):
            z0 = int(torch.randint(0, max(1, d - side + 1), (1,)).item())
            y0 = int(torch.randint(0, max(1, h - side + 1), (1,)).item())
            x0 = int(torch.randint(0, max(1, w - side + 1), (1,)).item())
            out[i, :, z0 : z0 + side, y0 : y0 + side, x0 : x0 + side] = 0.0
    return out
