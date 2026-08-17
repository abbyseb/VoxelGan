"""3D U-Net generator with FiLM phase conditioning → diffeomorphic DVF.

Anatomy path (encoder) is unconditioned. FiLM is applied at the bottleneck
and every decoder scale *after* skip fusion. Output is a stationary velocity
field integrated by scaling-and-squaring (see svf.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities.film import FiLMConditioner, apply_film
from utilities.svf import scaling_and_squaring


class ConvBlock(nn.Module):
    """[Conv3³ → IN → LeakyReLU(0.2)] × 2, residual when in_ch == out_ch."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.residual = (in_ch == out_ch)
        self.proj = None if self.residual else nn.Conv3d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        y = self.block(x)
        if self.residual:
            return y + x
        return y + self.proj(x)


class Down(nn.Module):
    """Strided Conv3³ (s=2) → IN → LeakyReLU, then conv block."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv = ConvBlock(out_ch, out_ch)

    def forward(self, x):
        return self.conv(self.down(x))


class Up(nn.Module):
    """Trilinear upsample → Conv3³ to out_ch → concat skip → conv → FiLM → conv."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # after concat: out_ch + skip_ch → out_ch
        self.conv1 = ConvBlock(out_ch + skip_ch, out_ch)
        self.conv2 = ConvBlock(out_ch, out_ch)

    def forward(self, x, skip, gamma, beta):
        x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=True)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = apply_film(x, gamma, beta)
        x = self.conv2(x)
        return x


class UNetFiLM(nn.Module):
    def __init__(self, im_size=128, n_phases=10, int_steps=6):
        super().__init__()
        self.im_size = im_size
        self.int_steps = int_steps

        # Stem: 1 → 16
        self.stem = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.enc1 = Down(16, 32)    # → 64³
        self.enc2 = Down(32, 64)    # → 32³
        self.enc3 = Down(64, 128)   # → 16³
        self.enc4 = Down(128, 256)  # → 8³

        self.bot1 = ConvBlock(256, 256)
        self.bot2 = ConvBlock(256, 256)

        # Dec4..Dec1: in_ch from below, skip_ch, out_ch
        self.dec4 = Up(256, 128, 128)
        self.dec3 = Up(128, 64, 64)
        self.dec2 = Up(64, 32, 32)
        self.dec1 = Up(32, 16, 16)

        self.film = FiLMConditioner(
            n_phases=n_phases,
            embed_dim=8,
            channel_list=[256, 128, 64, 32, 16],
        )

        self.out_conv = nn.Conv3d(16, 3, kernel_size=1)

    def forward(self, reference_ct, ref_phase, target_phase):
        # --- encode (unconditioned) ---
        s0 = self.stem(reference_ct)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        x = self.enc4(s3)

        # --- bottleneck + FiLM ---
        film_params = self.film(ref_phase, target_phase)
        # film_params: [bottleneck, Dec4, Dec3, Dec2, Dec1]
        x = self.bot1(x)
        x = self.bot2(x)
        g, b = film_params[0]
        x = apply_film(x, g, b)

        # --- decode (FiLM after each skip fusion) ---
        x = self.dec4(x, s3, *film_params[1])
        x = self.dec3(x, s2, *film_params[2])
        x = self.dec2(x, s1, *film_params[3])
        x = self.dec1(x, s0, *film_params[4])

        v = self.out_conv(x)
        dvf = scaling_and_squaring(v, int_steps=self.int_steps)
        return dvf
