"""3D U-Net with VoxelMap's stride-2 blocks and the CRB phase block.

One 3D CT, phase pair (t_ref, t_tgt) coded like UNetCRBDecoder, skip links.
The encoder is anatomy only. Each decoder block turns the two phase numbers
into a scale and a shift with the CRB MLP (2 → 32 → 16 → 2C), applies that
before the ReLU, and adds a skip that is not normalized.

No batch-norm. Channels start at 16 and double up to 64, so 160³ is
16-32-64-64-64 with a 5³ bottleneck. The 4-channel batch-norm FiLM version
predicted a zero field on SPARE.

Depth follows the volume. Each down halves the grid, and stops when the next
grid would be odd or smaller than 2³. Output is a direct 3-channel DVF.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseScaleShift(nn.Module):
    """CRB phase MLP. y * a + b, with (a, b) from the two phase numbers."""

    def __init__(self, num_features: int, cond_dim: int = 2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(cond_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2 * num_features),
        )

    def forward(self, y: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        a, b = self.fc(cond).chunk(2, dim=1)
        a = a.view(-1, y.shape[1], 1, 1, 1)
        b = b.view(-1, y.shape[1], 1, 1, 1)
        return y * a + b


class DownBlock3D(nn.Module):
    """Stride-2 conv, second conv, residual. No batch-norm."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = F.relu(self.conv1(x))
        conv2 = self.conv2(conv1)
        return F.relu(conv1 + conv2)


class UpBlock3D(nn.Module):
    """Learned upsample, encoder skip, then the CRB phase block."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, cond_dim: int = 2):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv3d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1)
        self.phase = PhaseScaleShift(out_ch, cond_dim)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.proj = nn.Conv3d(out_ch + skip_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        up = F.relu(self.up(x))
        if up.shape[2:] != skip.shape[2:]:
            up = F.interpolate(up, size=skip.shape[2:], mode="trilinear", align_corners=True)
        cat = torch.cat([up, skip], dim=1)
        y = self.phase(self.conv1(cat), cond)
        y = F.relu(y)
        y = self.conv2(y)
        return F.relu(y + self.proj(cat))


def _n_downs(im_size: int) -> int:
    """How many stride-2 steps stay on an even grid of at least 2³."""
    n = 0
    size = int(im_size)
    while size % 2 == 0 and size // 2 >= 2:
        size //= 2
        n += 1
    return n


def _channels(n_down: int) -> list[int]:
    """Start at 16 and double, capped at 64. 160³ → 16-32-64-64-64."""
    channels = []
    width = 16
    for _ in range(n_down):
        channels.append(width)
        width = min(width * 2, 64)
    return channels


class UNetVoxelMapCT(nn.Module):
    """CT + two phases → 3-channel DVF. Same call as UNetCRBDecoder."""

    def __init__(self, im_size: int = 160, n_phases: int = 10):
        super().__init__()
        self.im_size = int(im_size)
        self.n_phases = int(n_phases)
        n_down = _n_downs(self.im_size)
        if n_down < 1:
            raise ValueError(f"im_size {im_size} cannot be halved")

        channels = _channels(n_down)
        self.channels = channels
        stem_ch = channels[0]

        self.stem = nn.Sequential(
            nn.Conv3d(1, stem_ch, kernel_size=3, padding=1),
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
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip, cond)
        return self.out_conv(x)
