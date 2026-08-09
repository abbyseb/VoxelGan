"""UNetCRB — from-scratch replica of Sang & Ruan (Med Phys 2023) Figures 2–3.

Adapted for single reference CT + two phase codes [t_ref, t_tgt] instead of
the paper's two-image + scalar-t conditioning. Encoder uses Conditional
Residual Blocks (CRB); decoder is unconditioned plain convolutions.

Known deviations from the paper / from parent Voxel_GAN UNetFiLM:
  - Conditioning vector is 2-D (ref + target phase), not scalar t.
  - CRB (a, b) are per-channel vectors (strengthening over ambiguous scalar Fig. 3).
  - Output is a direct 3-channel DVF with linear activation — NO scaling-and-
    squaring (svf.py is not used). Fold prevention relies only on L_smooth.
  - 1×1 projection on the residual path when in_ch ≠ out_ch.
  - No BatchNorm / InstanceNorm anywhere (paper design).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalResidualBlock(nn.Module):
    """CRB (Fig. 3): private FC → (a, b); conv → FiLM-like → conv + residual.

    Residual wraps both convolutions so an unconditioned identity path exists
    through the block. Each instance owns its own FC weights (not shared).
    """

    def __init__(self, in_ch, out_ch, cond_dim=2):
        super().__init__()
        self.out_ch = out_ch
        # cond (2) → 32 → 16 → 2 * out_ch  (per-channel a, b)
        self.fc = nn.Sequential(
            nn.Linear(cond_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2 * out_ch),
        )
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        # engineering necessity when channels change (not shown in paper figure)
        self.proj = None if in_ch == out_ch else nn.Conv3d(in_ch, out_ch, kernel_size=1)

    def forward(self, x, cond):
        # cond: (B, 2)
        ab = self.fc(cond)
        a, b = ab.chunk(2, dim=1)
        a = a.view(-1, self.out_ch, 1, 1, 1)
        b = b.view(-1, self.out_ch, 1, 1, 1)

        y = self.conv1(x)
        y = y * a + b
        y = F.relu(y, inplace=True)
        y = self.conv2(y)
        res = x if self.proj is None else self.proj(x)
        return F.relu(y + res, inplace=True)


class Down(nn.Module):
    """Average-pool ×2 then CRB (encoder stage)."""

    def __init__(self, in_ch, out_ch, cond_dim=2):
        super().__init__()
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)
        self.crb = ConditionalResidualBlock(in_ch, out_ch, cond_dim=cond_dim)

    def forward(self, x, cond):
        return self.crb(self.pool(x), cond)


class Up(nn.Module):
    """Fixed upsample ×2, skip concat, plain convs (no CRB / no FiLM)."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        # after concat: in_ch + skip_ch → out_ch
        self.conv1 = nn.Conv3d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        return x


class UNetCRB(nn.Module):
    """5-level conditional U-Net (Fig. 2) → direct 3-ch DVF."""

    def __init__(self, im_size=128, n_phases=10):
        super().__init__()
        self.im_size = im_size
        self.n_phases = n_phases
        cond_dim = 2

        # Encoder: CRB at every level including bottleneck (5 CRBs, independent FCs)
        # Channel progression: 16, 16→32, 32→32, 32→64, 64→64
        self.enc1 = ConditionalResidualBlock(1, 16, cond_dim=cond_dim)
        self.down2 = Down(16, 32, cond_dim=cond_dim)
        self.down3 = Down(32, 32, cond_dim=cond_dim)
        self.down4 = Down(32, 64, cond_dim=cond_dim)
        self.down5 = Down(64, 64, cond_dim=cond_dim)  # bottleneck

        # Decoder: plain convs only
        self.up4 = Up(64, 64, 64)
        self.up3 = Up(64, 32, 32)
        self.up2 = Up(32, 32, 32)
        self.up1 = Up(32, 16, 16)

        # Final linear 3-ch DVF — NO scaling-and-squaring / diffeomorphism guarantee
        self.out_conv = nn.Conv3d(16, 3, kernel_size=3, padding=1)

    def _phase_vec(self, ref_phase, target_phase):
        # normalize to [0, 1] via /(n_phases - 1); concat → (B, 2)
        denom = float(max(self.n_phases - 1, 1))
        t_ref = ref_phase.float().unsqueeze(-1) / denom
        t_tgt = target_phase.float().unsqueeze(-1) / denom
        return torch.cat([t_ref, t_tgt], dim=1)

    def forward(self, reference_ct, ref_phase, target_phase):
        cond = self._phase_vec(ref_phase, target_phase)

        s1 = self.enc1(reference_ct, cond)
        s2 = self.down2(s1, cond)
        s3 = self.down3(s2, cond)
        s4 = self.down4(s3, cond)
        x = self.down5(s4, cond)

        x = self.up4(x, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)

        # direct DVF (linear); folds only discouraged by smoothness loss at train time
        return self.out_conv(x)
