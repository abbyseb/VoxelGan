"""UNetCRB-Both — ClinicalExperiments Experiment 2 (cyclic phase encoding).

Same both-CRB architecture as InitialExperiments E1, but phase codes are
cyclic: [cos 2πθ_ref, sin 2πθ_ref, cos 2πθ_tgt, sin 2πθ_tgt] (cond_dim=4).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalResidualBlock(nn.Module):
    """CRB: FC → (a, b); conv → scale/shift → conv + residual."""

    def __init__(self, in_ch, out_ch, cond_dim=4):
        super().__init__()
        self.out_ch = out_ch
        self.fc = nn.Sequential(
            nn.Linear(cond_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2 * out_ch),
        )
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.proj = None if in_ch == out_ch else nn.Conv3d(in_ch, out_ch, kernel_size=1)

    def forward(self, x, cond):
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


class DownCRB(nn.Module):
    """AvgPool ×2 then CRB."""

    def __init__(self, in_ch, out_ch, cond_dim=4):
        super().__init__()
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)
        self.crb = ConditionalResidualBlock(in_ch, out_ch, cond_dim=cond_dim)

    def forward(self, x, cond):
        return self.crb(self.pool(x), cond)


class UpCRB(nn.Module):
    """Upsample ×2, skip concat, then CRB."""

    def __init__(self, in_ch, skip_ch, out_ch, cond_dim=4):
        super().__init__()
        self.crb = ConditionalResidualBlock(in_ch + skip_ch, out_ch, cond_dim=cond_dim)

    def forward(self, x, skip, cond):
        x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.crb(x, cond)


class UNetCRBBoth(nn.Module):
    """5-level U-Net: CRB encoder + CRB decoder → direct 3-ch DVF."""

    def __init__(self, im_size=128, n_phases=10):
        super().__init__()
        self.im_size = im_size
        self.n_phases = n_phases
        self.cond_dim = 4
        cond_dim = self.cond_dim

        self.enc1 = ConditionalResidualBlock(1, 16, cond_dim=cond_dim)
        self.down2 = DownCRB(16, 32, cond_dim=cond_dim)
        self.down3 = DownCRB(32, 32, cond_dim=cond_dim)
        self.down4 = DownCRB(32, 64, cond_dim=cond_dim)
        self.down5 = DownCRB(64, 64, cond_dim=cond_dim)

        self.up4 = UpCRB(64, 64, 64, cond_dim=cond_dim)
        self.up3 = UpCRB(64, 32, 32, cond_dim=cond_dim)
        self.up2 = UpCRB(32, 32, 32, cond_dim=cond_dim)
        self.up1 = UpCRB(32, 16, 16, cond_dim=cond_dim)

        self.out_conv = nn.Conv3d(16, 3, kernel_size=3, padding=1)

    def _phase_vec(self, ref_phase, target_phase):
        two_pi_n = 2.0 * math.pi / float(self.n_phases)
        th_r = ref_phase.float() * two_pi_n
        th_t = target_phase.float() * two_pi_n
        return torch.cat(
            [
                torch.cos(th_r).unsqueeze(-1),
                torch.sin(th_r).unsqueeze(-1),
                torch.cos(th_t).unsqueeze(-1),
                torch.sin(th_t).unsqueeze(-1),
            ],
            dim=1,
        )

    def forward(self, reference_ct, ref_phase, target_phase):
        cond = self._phase_vec(ref_phase, target_phase)
        s1 = self.enc1(reference_ct, cond)
        s2 = self.down2(s1, cond)
        s3 = self.down3(s2, cond)
        s4 = self.down4(s3, cond)
        x = self.down5(s4, cond)
        x = self.up4(x, s4, cond)
        x = self.up3(x, s3, cond)
        x = self.up2(x, s2, cond)
        x = self.up1(x, s1, cond)
        return self.out_conv(x)
