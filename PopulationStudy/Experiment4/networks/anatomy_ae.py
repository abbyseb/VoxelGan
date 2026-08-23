"""Masked CT autoencoder whose encoder matches Decoder-CRB ResidualBlocks.

Channel progression 1→16→32→32→64→64 so `AnatomyEncoder` weights load into
`UNetCRBDecoder.enc1 / down2…down5`. Decoder here is pretrain-only (not spliced).

Encoder/Both CRB keep their phase-CRB encoder; they consume a frozen
`AnatomyEncoder` as a side-branch (see Experiment4/README.md).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Same unconditioned block as UNetCRBDecoder."""

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


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)
        self.block = ResidualBlock(in_ch, out_ch)

    def forward(self, x):
        return self.block(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.block = ResidualBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=True)
        return self.block(torch.cat([x, skip], dim=1))


class AnatomyEncoder(nn.Module):
    """Returns bottleneck + skips (s1..s4). Input (B,1,128,128,128)."""

    def __init__(self):
        super().__init__()
        self.enc1 = ResidualBlock(1, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 32)
        self.down4 = Down(32, 64)
        self.down5 = Down(64, 64)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        s4 = self.down4(s3)
        bottleneck = self.down5(s4)
        return bottleneck, (s1, s2, s3, s4)


class AnatomyDecoder(nn.Module):
    """Pretrain recon head only — not copied into CRB nets."""

    def __init__(self):
        super().__init__()
        self.up4 = Up(64, 64, 64)
        self.up3 = Up(64, 32, 32)
        self.up2 = Up(32, 32, 32)
        self.up1 = Up(32, 16, 16)
        self.out_conv = nn.Conv3d(16, 1, kernel_size=3, padding=1)

    def forward(self, bottleneck, skips):
        s1, s2, s3, s4 = skips
        x = self.up4(bottleneck, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        return self.out_conv(x)


class AnatomyAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AnatomyEncoder()
        self.decoder = AnatomyDecoder()

    def forward(self, ct):
        z, skips = self.encoder(ct)
        return self.decoder(z, skips)

    def encoder_state_dict(self):
        return self.encoder.state_dict()


def load_encoder_state_dict(path):
    """Accept full AE checkpoint or encoder-only state_dict."""
    blob = torch.load(path, map_location="cpu")
    if isinstance(blob, dict) and "encoder" in blob:
        return blob["encoder"]
    if isinstance(blob, dict) and any(k.startswith("encoder.") for k in blob):
        return {k[len("encoder.") :]: v for k, v in blob.items() if k.startswith("encoder.")}
    return blob


class FrozenAnatomyVec(nn.Module):
    """Side-branch for Encoder/Both CRB: GAP bottleneck → D-vector.

    AnatomyEncoder is frozen after load; `proj` is trained with the DVF head.
    """

    def __init__(self, out_dim=32):
        super().__init__()
        self.encoder = AnatomyEncoder()
        self.proj = nn.Linear(64, out_dim)
        self._frozen = False

    def load_encoder(self, path):
        self.encoder.load_state_dict(load_encoder_state_dict(path), strict=True)
        self.freeze_encoder()

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        self._frozen = True

    def train(self, mode=True):
        super().train(mode)
        if self._frozen:
            self.encoder.eval()
        return self

    def forward(self, ct):
        if self._frozen:
            with torch.no_grad():
                z, _ = self.encoder(ct)
        else:
            z, _ = self.encoder(ct)
        return self.proj(z.mean(dim=(2, 3, 4)))
