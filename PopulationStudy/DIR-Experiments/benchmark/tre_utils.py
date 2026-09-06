"""Shared TRE / landmark utilities for DIR-Lab benchmark."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PHASE_T00, PHASE_T50 = 1, 6


def load_landmarks_xyz(path: Path) -> np.ndarray:
    pts = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        if len(parts) >= 3:
            pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.asarray(pts, dtype=np.float64)


def tre_mm(pred_xyz: np.ndarray, gt_xyz: np.ndarray, spacing_xyz_mm) -> dict:
    sx, sy, sz = spacing_xyz_mm
    d = (pred_xyz - gt_xyz) * np.array([sx, sy, sz], dtype=np.float64)
    err = np.linalg.norm(d, axis=1)
    return {
        "n": int(len(err)),
        "mean": float(np.mean(err)),
        "std": float(np.std(err)),
        "median": float(np.median(err)),
        "p95": float(np.percentile(err, 95)),
        "max": float(np.max(err)),
    }


def find_300_landmarks(packed_dir: Path, pid: str, meta: dict):
    lm_dir = packed_dir / pid / "landmarks"
    ref_key = tgt_key = None
    for name in meta["landmarks"]:
        if "300" in name and "T00" in name:
            ref_key = name
        if "300" in name and "T50" in name:
            tgt_key = name
    if ref_key is None or tgt_key is None:
        raise FileNotFoundError(f"{pid}: missing 300 T00/T50 landmarks")
    stem_ref = Path(ref_key).stem
    stem_tgt = Path(tgt_key).stem
    ref_native = np.load(lm_dir / f"{stem_ref}_native_xyz.npy")
    tgt_native = np.load(lm_dir / f"{stem_tgt}_native_xyz.npy")
    ref_packed = np.load(lm_dir / f"{stem_ref}_packed_xyz.npy")
    return ref_native, tgt_native, ref_packed


def packed_voxel_to_native_xyz(xyz_packed: np.ndarray, meta: dict) -> np.ndarray:
    """Packed (x,y,z) in pack layout → native DIR ITK (x,y,z)."""
    z0, z1, y0, y1, x0, x1 = meta["bbox_zyx"]
    out_size = meta["out_size"]
    sx = out_size / max(x1 - x0, 1)
    sy = out_size / max(y1 - y0, 1)
    sz = out_size / max(z1 - z0, 1)
    spare = np.empty_like(xyz_packed)
    spare[:, 0] = xyz_packed[:, 0] / sx + x0
    spare[:, 1] = xyz_packed[:, 1] / sy + y0
    spare[:, 2] = xyz_packed[:, 2] / sz + z0
    layout = meta.get("axis_remap", "")
    if "SPARE-like" in layout or meta.get("layout") == "spare_axes":
        # spare = (x, z_si, y) → native DIR (x, y, z_si)
        native = np.empty_like(spare)
        native[:, 0] = spare[:, 0]
        native[:, 1] = spare[:, 2]
        native[:, 2] = spare[:, 1]
        return native
    # axial layout: packed axes match native DIR (x,y,z)
    return spare


def native_to_packed_xyz(xyz_native: np.ndarray, meta: dict) -> np.ndarray:
    z0, z1, y0, y1, x0, x1 = meta["bbox_zyx"]
    out_size = meta["out_size"]
    sx = out_size / max(x1 - x0, 1)
    sy = out_size / max(y1 - y0, 1)
    sz = out_size / max(z1 - z0, 1)
    layout = meta.get("axis_remap", "")
    if "SPARE-like" in layout or meta.get("layout") == "spare_axes":
        spare = np.empty_like(xyz_native)
        spare[:, 0] = xyz_native[:, 0]
        spare[:, 1] = xyz_native[:, 2]
        spare[:, 2] = xyz_native[:, 1]
        xyz = spare
    else:
        xyz = xyz_native
    out = np.empty_like(xyz)
    out[:, 0] = (xyz[:, 0] - x0) * sx
    out[:, 1] = (xyz[:, 1] - y0) * sy
    out[:, 2] = (xyz[:, 2] - z0) * sz
    return out


def packed_disp_to_native_mm(disp_packed: np.ndarray, meta: dict, spacing_xyz_mm) -> np.ndarray:
    """Displacement in packed voxels → mm in native DIR axes (per landmark)."""
    z0, z1, y0, y1, x0, x1 = meta["bbox_zyx"]
    out_size = meta["out_size"]
    # mm per packed voxel along each packed axis
    scale_x = (x1 - x0) / out_size * spacing_xyz_mm[0]
    scale_y = (y1 - y0) / out_size * spacing_xyz_mm[1]
    scale_z = (z1 - z0) / out_size * spacing_xyz_mm[2]
    layout = meta.get("axis_remap", "")
    if "SPARE-like" in layout or meta.get("layout") == "spare_axes":
        # disp in spare-packed (dx, dy_si, dz_ap) → native (dx, dy_ap, dz_si)
        out = np.empty_like(disp_packed)
        out[:, 0] = disp_packed[:, 0] * scale_x
        out[:, 1] = disp_packed[:, 2] * scale_z
        out[:, 2] = disp_packed[:, 1] * scale_y
        return out
    out = disp_packed.copy()
    out[:, 0] *= scale_x
    out[:, 1] *= scale_y
    out[:, 2] *= scale_z
    return out


def sample_dvf_at_xyz(dvf_cdhw: np.ndarray, xyz_packed: np.ndarray) -> np.ndarray:
    c, d, h, w = dvf_cdhw.shape
    t = torch.from_numpy(dvf_cdhw)[None].float()
    x, y, z = xyz_packed[:, 0], xyz_packed[:, 1], xyz_packed[:, 2]
    xn = 2.0 * x / max(w - 1, 1) - 1.0
    yn = 2.0 * y / max(h - 1, 1) - 1.0
    zn = 2.0 * z / max(d - 1, 1) - 1.0
    grid = torch.stack(
        [
            torch.from_numpy(xn).float(),
            torch.from_numpy(yn).float(),
            torch.from_numpy(zn).float(),
        ],
        dim=-1,
    ).view(1, -1, 1, 1, 3)
    samp = F.grid_sample(t, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return samp[0, :, :, 0, 0].permute(1, 0).numpy()


def eval_tre_from_disp(
    ref_native: np.ndarray,
    tgt_native: np.ndarray,
    ref_packed: np.ndarray,
    disp_packed: np.ndarray,
    meta: dict,
    spacing_xyz_mm,
    *,
    grid: str,
) -> dict:
    """Apply DVF to T00 landmarks and score vs expert T50.

    ITK Transformix deformation fields follow the image-warp convention used in
    ``prepare_clinical_dvf_library.register_pair`` / ``utilities.warp``: a point
    at ``p`` in the moving (reference) image is *sampled from* ``p + u(p)`` when
    building the warped volume. The corresponding landmark in the fixed (target)
    frame is therefore ``p - u(p)``, not ``p + u(p)``.
    """
    spacing = list(spacing_xyz_mm)
    pred_packed = ref_packed - disp_packed
    pred = packed_voxel_to_native_xyz(pred_packed, meta)
    tre = tre_mm(pred, tgt_native, spacing)
    tre_id = tre_mm(ref_native, tgt_native, spacing)
    return {"tre_mm": tre, "tre_identity_mm": tre_id, "grid": grid}


def load_pack_meta(packed_root: Path, pid: str) -> dict:
    return json.loads((packed_root / pid / "pack_meta.json").read_text())
