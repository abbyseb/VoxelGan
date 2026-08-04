"""3D PatchGAN discriminator for phase-conditioned DVF synthesis.

Input is concat[reference CT (1), DVF (3)] = 4 channels. Output is an 8³
grid of real/fake logits (not a scalar). Spectral norm on every conv.
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        # in_channels=4 = ref_CT(1) + DVF(3)
        self.net = nn.Sequential(
            # D1: 64³ → 32³, no norm
            spectral_norm(nn.Conv3d(in_channels, 32, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            # D2: 32³ → 16³
            spectral_norm(nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # D3: 16³ → 8³
            spectral_norm(nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # D4: 8³ → 8³ (padding='same' keeps spatial size with k=4, s=1)
            spectral_norm(nn.Conv3d(128, 256, kernel_size=4, stride=1, padding='same')),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Out: 8³ → 8³ logits
            spectral_norm(nn.Conv3d(256, 1, kernel_size=4, stride=1, padding='same')),
        )

    def forward(self, reference_ct, dvf):
        x = torch.cat([reference_ct, dvf], dim=1)
        return self.net(x)
