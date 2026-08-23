"""UNetCRB — Experiment 4 copy with frozen anatomy side-branch.

Phase-CRB encoder is unchanged. Frozen AnatomyEncoder → GAP → concat onto
[t_ref, t_tgt]. Code is ready; E4 does not train this arch by default.
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
    """5-level conditional U-Net + frozen anatomy vector (E4 side-branch)."""

    def __init__(self, im_size=128, n_phases=10, anatomy_dim=32):
        super().__init__()
        self.im_size = im_size
        self.n_phases = n_phases
        self.anatomy_dim = int(anatomy_dim)
        cond_dim = 2 + self.anatomy_dim
        from networks.anatomy_ae import FrozenAnatomyVec

        self.anatomy = FrozenAnatomyVec(out_dim=self.anatomy_dim)

        self.enc1 = ConditionalResidualBlock(1, 16, cond_dim=cond_dim)
        self.down2 = Down(16, 32, cond_dim=cond_dim)
        self.down3 = Down(32, 32, cond_dim=cond_dim)
        self.down4 = Down(32, 64, cond_dim=cond_dim)
        self.down5 = Down(64, 64, cond_dim=cond_dim)

        self.up4 = Up(64, 64, 64)
        self.up3 = Up(64, 32, 32)
        self.up2 = Up(32, 32, 32)
        self.up1 = Up(32, 16, 16)
        self.out_conv = nn.Conv3d(16, 3, kernel_size=3, padding=1)

    def load_anatomy_encoder(self, path):
        self.anatomy.load_encoder(path)

    def _phase_vec(self, ref_phase, target_phase):
        denom = float(max(self.n_phases - 1, 1))
        t_ref = ref_phase.float().unsqueeze(-1) / denom
        t_tgt = target_phase.float().unsqueeze(-1) / denom
        return torch.cat([t_ref, t_tgt], dim=1)

    def forward(self, reference_ct, ref_phase, target_phase):
        cond = torch.cat(
            [self._phase_vec(ref_phase, target_phase), self.anatomy(reference_ct)],
            dim=1,
        )
        s1 = self.enc1(reference_ct, cond)
        s2 = self.down2(s1, cond)
        s3 = self.down3(s2, cond)
        s4 = self.down4(s3, cond)
        x = self.down5(s4, cond)
        x = self.up4(x, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        return self.out_conv(x)

