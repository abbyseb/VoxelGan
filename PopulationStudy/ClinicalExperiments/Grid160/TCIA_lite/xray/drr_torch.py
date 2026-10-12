"""Differentiable cone-beam DRR (forward projection) and ray back-projection in torch, RTK geometry convention.

Volume: tensor (B,1,D,H,W) on an isotropic grid of `spacing` mm, centred on the isocentre, axes = ITK (z,y,x).
For our R3 160³ volumes that is (AP, SI, LR), so the gantry rotates about H (SI), as in RTK.
World point of voxel (k,j,i) = ((i,j,k) - (N-1)/2) * spacing  (identity direction, no flips).

Projection matrix P (3x4) maps world (x,y,z,1) to detector (u,v) mm; built exactly like RTK's
ThreeDCircularProjectionGeometry (no tilts), checked against RTK in check_drr_torch.py:
    P = [[-SDD c - ox s, 0, SDD s - ox c, ox SID],
         [-oy s,        -SDD, -oy c,       oy SID],
         [ s,            0,    c,          -SID  ]]          c, s = cos, sin(gantry angle)
Detector: n_det x n_det pixels of det_spacing mm, centred (pixel (r,c) -> u = (c - (n-1)/2) * ds, v likewise).

drr():          line integral (sum of value * step mm) along each source->pixel ray (trilinear, zero outside).
backproject():  each voxel gets the detector value of the ray through its centre ("smear along the real ray").
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

SID, SDD = 1000.0, 1500.0
N_DET, DET_SPACING = 128, 4.0          # 512 mm detector -> ~341 mm at isocentre, covers the 320 mm cube


def rtk_matrix(angle_deg: float, sid: float = SID, sdd: float = SDD, ox: float = 0.0, oy: float = 0.0) -> np.ndarray:
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    return np.array([[-sdd * c - ox * s, 0.0, sdd * s - ox * c, ox * sid],
                     [-oy * s, -sdd, -oy * c, oy * sid],
                     [s, 0.0, c, -sid]], dtype=np.float64)


def _world_to_grid(p: torch.Tensor, shape, spacing: float) -> torch.Tensor:
    """world mm (...,3 as x,y,z) -> grid_sample coords in [-1,1] (align_corners=True), order (x->W, y->H, z->D)."""
    d, h, w = shape
    n = torch.tensor([w, h, d], device=p.device, dtype=p.dtype)
    return p / spacing / ((n - 1) / 2)


def _rays(P: np.ndarray, n_det: int, ds: float, device):
    """Source position C and unit directions (n_det, n_det, 3) for every detector pixel (row = v, col = u)."""
    M, p4 = P[:, :3], P[:, 3]
    C = -np.linalg.solve(M, p4)
    idx = (np.arange(n_det) - (n_det - 1) / 2) * ds
    uu, vv = np.meshgrid(idx, idx, indexing="xy")                       # (row, col)
    hom = np.stack([uu, vv, np.ones_like(uu)], -1)
    d = hom @ np.linalg.inv(M).T
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    d *= np.sign(d @ -C)[..., None]                                      # point from the source towards the isocentre
    to = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    return to(C), to(d)


def drr(vol: torch.Tensor, angles_deg, spacing: float = 2.0, n_det: int = N_DET, ds: float = DET_SPACING,
        step: float = 1.0, sid: float = SID, sdd: float = SDD, ox: float = 0.0, oy: float = 0.0) -> torch.Tensor:
    """vol (B,1,D,H,W) -> projections (B, n_angles, n_det, n_det); units value*mm. Differentiable w.r.t. vol."""
    b, _, D, H, W = vol.shape
    half = 0.5 * spacing * math.sqrt((D - 1) ** 2 + (H - 1) ** 2 + (W - 1) ** 2) + spacing
    t = torch.arange(sid - half, sid + half + step, step, device=vol.device)
    out = []
    for a in angles_deg:
        C, d = _rays(rtk_matrix(a, sid, sdd, ox, oy), n_det, ds, vol.device)
        pts = C + d[:, :, None, :] * t[None, None, :, None]               # (r, c, T, 3)
        g = _world_to_grid(pts, (D, H, W), spacing)
        s = F.grid_sample(vol, g[None].expand(b, -1, -1, -1, -1), mode="bilinear", padding_mode="zeros",
                          align_corners=True)                              # (B,1,r,c,T)
        out.append(s.sum(-1)[:, 0] * step)
    return torch.stack(out, 1)


def project_points(xyz_mm: np.ndarray, angle_deg: float, n_det: int = N_DET, ds: float = DET_SPACING,
                   **kw) -> np.ndarray:
    """world mm (N,3) -> detector (col,row) pixel coords."""
    P = rtk_matrix(angle_deg, **kw)
    h = np.c_[xyz_mm, np.ones(len(xyz_mm))] @ P.T
    uv = h[:, :2] / h[:, 2:3]
    return uv / ds + (n_det - 1) / 2


def backproject(proj: torch.Tensor, angles_deg, shape, spacing: float = 2.0, ds: float = DET_SPACING,
                sid: float = SID, sdd: float = SDD, ox: float = 0.0, oy: float = 0.0) -> torch.Tensor:
    """proj (B, n_angles, n, n) -> (B, n_angles, D, H, W): each voxel = detector value at its projected pixel."""
    b, na, n, _ = proj.shape
    D, H, W = shape
    dev = proj.device
    k, j, i = torch.meshgrid(*(torch.arange(m, device=dev, dtype=torch.float32) for m in (D, H, W)), indexing="ij")
    X = torch.stack([(i - (W - 1) / 2), (j - (H - 1) / 2), (k - (D - 1) / 2), torch.full_like(i, 1 / spacing)], -1) * spacing
    out = []
    for a_i, a in enumerate(angles_deg):
        P = torch.as_tensor(rtk_matrix(a, sid, sdd, ox, oy), dtype=torch.float32, device=dev)
        h = X @ P.T
        uv = h[..., :2] / h[..., 2:3]
        g = uv / ds / ((n - 1) / 2)                                        # -> [-1,1], (u -> col = x, v -> row = y)
        s = F.grid_sample(proj[:, a_i:a_i + 1], g.reshape(1, D * H, W, 2).expand(b, -1, -1, -1),
                          mode="bilinear", padding_mode="zeros", align_corners=True)
        out.append(s.reshape(b, D, H, W))
    return torch.stack(out, 1)
