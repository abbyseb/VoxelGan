"""Warp helper for image-similarity loss and warp-space discriminator.

Copied from parent svf.warp — no scaling-and-squaring (UNetCRB predicts DVF directly).
"""

import torch
import torch.nn.functional as F


def _meshgrid(shape, device, dtype):
    d, h, w = shape
    zz, yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, d, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
        indexing='ij',
    )
    grid = torch.stack([xx, yy, zz], dim=0).unsqueeze(0)
    return grid


def _flow_to_grid(flow):
    _, _, d, h, w = flow.shape
    grid = _meshgrid((d, h, w), flow.device, flow.dtype)
    scale = torch.tensor(
        [2.0 / max(w - 1, 1), 2.0 / max(h - 1, 1), 2.0 / max(d - 1, 1)],
        device=flow.device,
        dtype=flow.dtype,
    ).view(1, 3, 1, 1, 1)
    disp = flow * scale
    sample_grid = (grid + disp).permute(0, 2, 3, 4, 1)
    return sample_grid.expand(flow.size(0), -1, -1, -1, -1)


def warp(vol, flow):
    """Warp a (B, C, D, H, W) volume by voxel-space flow (B, 3, D, H, W)."""
    grid = _flow_to_grid(flow)
    return F.grid_sample(
        vol, grid, mode='bilinear', padding_mode='border', align_corners=True
    )
