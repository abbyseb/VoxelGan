"""3D PatchGAN discriminator for phase-conditioned DVF synthesis.

Two input modes (same backbone; set in_channels to match):
  - dvf  (baseline): concat[ref CT (1), DVF (3)] → 4 channels
  - warp (scenario): concat[warped CT (1), target CT (1)] → 2 channels

Output is an 8³ grid of real/fake logits. Spectral norm on every conv.
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=4, base_channels=32):
        super().__init__()
        # base_channels=32 is the baseline width (→ 2×, 4×, 8×).
        # Scenario "weaker D" uses base_channels=16.
        c = base_channels
        self.net = nn.Sequential(
            # D1: 64³ → 32³, no norm
            spectral_norm(nn.Conv3d(in_channels, c, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            # D2: 32³ → 16³
            spectral_norm(nn.Conv3d(c, c * 2, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm3d(c * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # D3: 16³ → 8³
            spectral_norm(nn.Conv3d(c * 2, c * 4, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm3d(c * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # D4: 8³ → 8³
            spectral_norm(nn.Conv3d(c * 4, c * 8, kernel_size=4, stride=1, padding='same')),
            nn.InstanceNorm3d(c * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Out: 8³ → 8³ logits
            spectral_norm(nn.Conv3d(c * 8, 1, kernel_size=4, stride=1, padding='same')),
        )

    def forward(self, a, b):
        # a, b are volumes; channels must sum to in_channels
        x = torch.cat([a, b], dim=1)
        return self.net(x)
