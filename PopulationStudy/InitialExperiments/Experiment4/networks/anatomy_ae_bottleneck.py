"""Compressed anatomy AE: no skip connections, GAP → 32-D latent.

Unlike the original UNet-style AE (skips can pass pixels → near-identity),
this forces all anatomy through a tiny code before decoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.anatomy_ae import ResidualBlock, Down


class BottleneckAnatomyEncoder(nn.Module):
    """Same channel path as AnatomyEncoder, then GAP → latent."""

    def __init__(self, latent_dim=32):
        super().__init__()
        self.enc1 = ResidualBlock(1, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 32)
        self.down4 = Down(32, 64)
        self.down5 = Down(64, 64)
        self.proj = nn.Linear(64, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x):
        h = self.enc1(x)
        h = self.down2(h)
        h = self.down3(h)
        h = self.down4(h)
        h = self.down5(h)
        z = self.proj(h.mean(dim=(2, 3, 4)))
        return z


class BottleneckAnatomyDecoder(nn.Module):
    """Decode from latent only (no skips). Upsamples to target size."""

    def __init__(self, latent_dim=32, base=4):
        super().__init__()
        self.base = base
        self.fc = nn.Linear(latent_dim, 64 * base * base * base)
        self.up4 = ResidualBlock(64, 64)
        self.up3 = ResidualBlock(64, 32)
        self.up2 = ResidualBlock(32, 32)
        self.up1 = ResidualBlock(32, 16)
        self.out = nn.Conv3d(16, 1, kernel_size=3, padding=1)

    def forward(self, z, size):
        b = z.shape[0]
        x = self.fc(z).view(b, 64, self.base, self.base, self.base)
        for block in (self.up4, self.up3, self.up2, self.up1):
            x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=True)
            x = block(x)
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=True)
        return self.out(x)


class BottleneckAnatomyAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = BottleneckAnatomyEncoder(latent_dim=latent_dim)
        self.decoder = BottleneckAnatomyDecoder(latent_dim=latent_dim)

    def forward(self, ct):
        z = self.encoder(ct)
        return self.decoder(z, size=ct.shape[2:])
