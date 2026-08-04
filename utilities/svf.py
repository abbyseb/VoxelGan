"""Scaling-and-squaring integration: stationary velocity field → diffeomorphic DVF.

Fixed, non-trainable post-processing. v is integrated by recursive composition
for `int_steps` squarings (default 6).
"""

import torch
import torch.nn.functional as F


def _meshgrid(shape, device, dtype):
    """Identity sampling grid in align_corners=True normalized coords, (1, 3, D, H, W)."""
    d, h, w = shape
    zz, yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, d, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
        indexing='ij',
    )
    # grid_sample expects (x, y, z) = (W, H, D) order in last dim
    grid = torch.stack([xx, yy, zz], dim=0).unsqueeze(0)
    return grid


def _flow_to_grid(flow):
    """Convert voxel-space displacement (B, 3, D, H, W) to sampling grid."""
    _, _, d, h, w = flow.shape
    grid = _meshgrid((d, h, w), flow.device, flow.dtype)
    # normalize voxel displacements to [-1, 1] grid units
    scale = torch.tensor([2.0 / max(w - 1, 1),
                          2.0 / max(h - 1, 1),
                          2.0 / max(d - 1, 1)],
                         device=flow.device, dtype=flow.dtype).view(1, 3, 1, 1, 1)
    # flow channels are (dx, dy, dz) = (x, y, z) matching grid order
    disp = flow * scale
    sample_grid = (grid + disp).permute(0, 2, 3, 4, 1)  # (B, D, H, W, 3)
    return sample_grid.expand(flow.size(0), -1, -1, -1, -1)


def warp(vol, flow):
    """Warp a (B, C, D, H, W) volume by voxel-space flow (B, 3, D, H, W)."""
    grid = _flow_to_grid(flow)
    return F.grid_sample(vol, grid, mode='bilinear', padding_mode='border',
                         align_corners=True)


def scaling_and_squaring(v, int_steps=6):
    """Integrate stationary velocity field v → displacement field (DVF)."""
    flow = v / (2 ** int_steps)
    for _ in range(int_steps):
        flow = flow + warp(flow, flow)
    return flow
