"""UNetCRB-Decoder with cyclic phases + oracle log(r_p).

cond_dim=5: [cos 2πθ_ref, sin 2πθ_ref, cos 2πθ_tgt, sin 2πθ_tgt, log(r_p)]
θ = phase_index / n_phases (0-indexed).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.proj = None if in_ch == out_ch else nn.Conv3d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        y = F.relu(self.conv1(x), inplace=True)
        y = self.conv2(y)
        res = x if self.proj is None else self.proj(x)
        return F.relu(y + res, inplace=True)


class ConditionalResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim=5):
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


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)
        self.block = ResidualBlock(in_ch, out_ch)

    def forward(self, x):
        return self.block(self.pool(x))


class UpCRB(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, cond_dim=5):
        super().__init__()
        self.crb = ConditionalResidualBlock(in_ch + skip_ch, out_ch, cond_dim=cond_dim)

    def forward(self, x, skip, cond):
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.crb(x, cond)


class UNetCRBDecoderAmp(nn.Module):
    """Same name as E1 for drop-in train script; cyclic phases + oracle amp."""

    def __init__(self, im_size=64, n_phases=10, cond_dim=5):
        super().__init__()
        self.im_size = im_size
        self.n_phases = n_phases
        self.cond_dim = cond_dim

        self.enc1 = ResidualBlock(1, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 32)
        self.down4 = Down(32, 64)
        self.down5 = Down(64, 64)

        self.up4 = UpCRB(64, 64, 64, cond_dim=cond_dim)
        self.up3 = UpCRB(64, 32, 32, cond_dim=cond_dim)
        self.up2 = UpCRB(32, 32, 32, cond_dim=cond_dim)
        self.up1 = UpCRB(32, 16, 16, cond_dim=cond_dim)
        self.out_conv = nn.Conv3d(16, 3, kernel_size=3, padding=1)

    def _cond_vec(self, ref_phase, target_phase, log_r):
        two_pi_n = 2.0 * math.pi / float(self.n_phases)
        th_r = ref_phase.float() * two_pi_n
        th_t = target_phase.float() * two_pi_n
        if log_r.dim() == 1:
            log_r = log_r.unsqueeze(-1)
        return torch.cat(
            [
                torch.cos(th_r).unsqueeze(-1),
                torch.sin(th_r).unsqueeze(-1),
                torch.cos(th_t).unsqueeze(-1),
                torch.sin(th_t).unsqueeze(-1),
                log_r.float(),
            ],
            dim=1,
        )

    def forward(self, reference_ct, ref_phase, target_phase, log_r):
        cond = self._cond_vec(ref_phase, target_phase, log_r)
        s1 = self.enc1(reference_ct)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        s4 = self.down4(s3)
        x = self.down5(s4)
        x = self.up4(x, s4, cond)
        x = self.up3(x, s3, cond)
        x = self.up2(x, s2, cond)
        x = self.up1(x, s1, cond)
        return self.out_conv(x)
