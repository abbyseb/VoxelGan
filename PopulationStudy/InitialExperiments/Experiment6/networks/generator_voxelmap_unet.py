"""3D U-Net with VoxelMap's blocks, synthesizer inputs.

VoxelMap's concatenated net encodes two 2D X-rays and grows a volume from one
code. It has no skip links. This keeps that net's residual stride-2 blocks,
batch-norm, and doubling channels, and changes the front:

  - one 3D CT instead of two X-rays
  - phase pair (t_ref, t_tgt), same coding as UNetCRBDecoder
  - skip links, so it is a U-Net: the decoder sees the encoder map at each size
  - encoder is anatomy only; FiLM from the phase pair is on the decoder

Depth follows the volume. Each down halves the grid, and stops when the next
grid would be odd or smaller than 2³. 160³ → 5 downs, channels 4-8-16-32-64, bottleneck 5³.
128³ → 6 downs, channels 4-8-16-32-64-128, bottleneck 2³.
The last step to 1³ is left out: batch-norm on a single voxel breaks.
Output is a direct 3-channel DVF, no scaling-and-squaring.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseFiLM(nn.Module):
    """Scale and shift a 3D feature map from the two phase numbers."""

    def __init__(self, num_features: int, cond_dim: int = 2):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, num_features)
        self.beta = nn.Linear(cond_dim, num_features)
        nn.init.zeros_(self.gamma.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        g = self.gamma(cond).view(-1, x.shape[1], 1, 1, 1)
        b = self.beta(cond).view(-1, x.shape[1], 1, 1, 1)
        return g * x + b


class DownBlock3D(nn.Module):
    """VoxelMap down block: stride-2 conv, second conv, batch-norm, residual."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = F.relu(self.conv1(x), inplace=True)
        conv2 = self.bn(self.conv2(conv1))
        return F.relu(conv1 + conv2, inplace=True)


class UpBlock3D(nn.Module):
    """VoxelMap up block, plus the encoder skip and phase FiLM."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, cond_dim: int = 2):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv3d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(out_ch)
        self.film = PhaseFiLM(out_ch, cond_dim)
        self.proj = nn.Conv3d(out_ch + skip_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        conv1 = F.relu(self.up(x), inplace=True)
        if conv1.shape[2:] != skip.shape[2:]:
            conv1 = F.interpolate(conv1, size=skip.shape[2:], mode="trilinear", align_corners=True)
        cat = torch.cat([conv1, skip], dim=1)
        conv2 = self.film(self.bn(self.conv2(cat)), cond)
        return F.relu(self.proj(cat) + conv2, inplace=True)


def _n_downs(im_size: int) -> int:
    """How many stride-2 steps stay on an even grid of at least 2³."""
    n = 0
    size = int(im_size)
    while size % 2 == 0 and size // 2 >= 2:
        size //= 2
        n += 1
    return n


class UNetVoxelMapCT(nn.Module):
    """CT + two phases → 3-channel DVF. Same call as UNetCRBDecoder."""

    def __init__(self, im_size: int = 160, n_phases: int = 10):
        super().__init__()
        self.im_size = int(im_size)
        self.n_phases = int(n_phases)
        n_down = _n_downs(self.im_size)
        if n_down < 1:
            raise ValueError(f"im_size {im_size} cannot be halved")

        # VoxelMap's list starts at 4 and doubles. 160³ stops at 64 (5³). 128³ stops at 128 (2³).
        channels = [2 ** (i + 2) for i in range(n_down)]
        self.channels = channels
        stem_ch = channels[0]

        self.stem = nn.Sequential(
            nn.Conv3d(1, stem_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(stem_ch),
            nn.ReLU(inplace=True),
        )
        downs = []
        prev = stem_ch
        for ch in channels:
            downs.append(DownBlock3D(prev, ch))
            prev = ch
        self.downs = nn.ModuleList(downs)

        ups = []
        skip_channels = [stem_ch] + channels[:-1]
        for ch, skip_ch in zip(reversed(channels), reversed(skip_channels)):
            ups.append(UpBlock3D(prev, skip_ch, ch))
            prev = ch
        self.ups = nn.ModuleList(ups)
        self.out_conv = nn.Conv3d(prev, 3, kernel_size=3, padding=1)

    def _phase_vec(self, ref_phase: torch.Tensor, target_phase: torch.Tensor) -> torch.Tensor:
        denom = float(max(self.n_phases - 1, 1))
        t_ref = ref_phase.float().unsqueeze(-1) / denom
        t_tgt = target_phase.float().unsqueeze(-1) / denom
        return torch.cat([t_ref, t_tgt], dim=1)

    def forward(self, reference_ct: torch.Tensor, ref_phase: torch.Tensor, target_phase: torch.Tensor) -> torch.Tensor:
        cond = self._phase_vec(ref_phase, target_phase)
        skips = [self.stem(reference_ct)]
        x = skips[0]
        for down in self.downs:
            x = down(x)
            skips.append(x)
        # bottleneck is the last encoder map; decoder skips are the ones above it
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip, cond)
        return self.out_conv(x)
