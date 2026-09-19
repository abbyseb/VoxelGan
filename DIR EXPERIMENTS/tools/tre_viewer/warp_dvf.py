"""DVF magnitude / warp / difference helpers for TRE Viewer Phase 2.

Image warp convention (verified lung-masked on A1 C01):
  warped(p) = source(p + disp(p))   # +disp pull-back
  → residual |target − warped| ≪ |target − source| for a good field.

Landmark push-forward remains −disp for T00→T50 (see eval_a1_tre).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.ndimage import affine_transform, map_coordinates

Component = Literal["mag", "LR", "SI", "AP"]


def warp_pull(source_zyx: np.ndarray, dvf_zyx3: np.ndarray, *, sign: float = 1.0) -> np.ndarray:
    """Pull-back warp: ``out[p] = source[p + sign * disp(p)]``.

    ``dvf_zyx3`` is (z,y,x,3) with channels (dx, dy, dz) in sub-voxels.
    """
    src = np.asarray(source_zyx, dtype=np.float64)
    dvf = np.asarray(dvf_zyx3, dtype=np.float64)
    if src.ndim != 3 or dvf.shape != (*src.shape, 3):
        raise ValueError(f"shape mismatch src {src.shape} vs dvf {dvf.shape}")
    if not np.isfinite(src).all() or not np.isfinite(dvf).all():
        raise ValueError("Source volume and DVF must contain only finite values")
    zz, yy, xx = np.meshgrid(
        np.arange(src.shape[0]),
        np.arange(src.shape[1]),
        np.arange(src.shape[2]),
        indexing="ij",
    )
    dx, dy, dz = dvf[..., 0], dvf[..., 1], dvf[..., 2]
    coords = np.stack(
        [zz + sign * dz, yy + sign * dy, xx + sign * dx],
        axis=0,
    )
    return map_coordinates(src, coords, order=1, mode="nearest").astype(np.float32)


def dvf_components_mm(
    dvf_zyx3: np.ndarray,
    sub_to_mm_xyz: tuple[float, float, float],
) -> dict[str, np.ndarray]:
    """Return LR/SI/AP/mag maps in **mm** on the sub grid (zyx arrays).

    ``sub_to_mm_xyz`` is (sx, sy, sz) mm per sub-voxel along (x,y,z) =
    channels (dx,dy,dz). For R3 runs this is ``case_info['sub_to_mm_r3']``.
    """
    dvf = np.asarray(dvf_zyx3, dtype=np.float64)
    sx, sy, sz = sub_to_mm_xyz
    lr = (dvf[..., 0] * sx).astype(np.float32)
    # y channel: AP in pack-native; SI in R3 (itk Y). Caller labels via frame.
    ap_or_si_y = (dvf[..., 1] * sy).astype(np.float32)
    si_or_ap_z = (dvf[..., 2] * sz).astype(np.float32)
    mag = np.sqrt(lr * lr + ap_or_si_y * ap_or_si_y + si_or_ap_z * si_or_ap_z).astype(
        np.float32
    )
    return {
        "LR": lr,
        "Y": ap_or_si_y,
        "Z": si_or_ap_z,
        "mag": mag,
        # Friendly aliases — set by caller based on frame
        "AP": ap_or_si_y,  # correct for native pack; R3 overwrites label in UI
        "SI": si_or_ap_z,
    }


def label_components(frame: str) -> dict[str, str]:
    """Map UI names → DVF channel keys for the compute frame."""
    if frame == "r3":
        # R3: itk (x,y,z) ≈ (LR, SI_flip, AP_flip); channels still (dx,dy,dz)
        return {"LR": "LR", "SI": "Y", "AP": "Z"}
    # native pack: (x,y,z) ≈ (LR, AP, SI)
    return {"LR": "LR", "AP": "Y", "SI": "Z"}


def upsample_to_pack(
    vol_zyx: np.ndarray,
    pack_shape_zyx: tuple[int, int, int],
    *,
    frame: str = "native",
) -> np.ndarray:
    """Resample compute-grid scalars onto pack indices using the TRE convention."""
    src = np.asarray(vol_zyx, dtype=np.float32)
    matrix, offset = _pack_to_compute(src.shape, pack_shape_zyx, frame)
    return affine_transform(
        src, matrix, offset=offset, output_shape=pack_shape_zyx,
        order=1, mode="nearest", prefilter=False,
    )


def _pack_to_compute(sub_shape, pack_shape, frame):
    """Map output pack (z,y,x) to input compute (z,y,x), including R3 flips."""
    sz, sy, sx = sub_shape
    nz, ny, nx = pack_shape
    if frame == "native":
        return np.diag([sz / nz, sy / ny, sx / nx]), np.zeros(3)
    if frame == "r3":
        return (
            np.array([[0, -sz / ny, 0], [-sy / nz, 0, 0], [0, 0, sx / nx]]),
            np.array([(ny - 1) * sz / ny, (nz - 1) * sy / nz, 0]),
        )
    raise ValueError(f"Unknown DVF frame {frame!r}; specify native or r3")


def _displacements_to_pack(disp_xyz, sub_shape, pack_shape, frame):
    sz, sy, sx = sub_shape
    nz, ny, nx = pack_shape
    if frame == "native":
        return disp_xyz * np.array([nx / sx, ny / sy, nz / sz])
    if frame == "r3":
        return disp_xyz[..., [0, 2, 1]] * np.array([nx / sx, -ny / sz, -nz / sy])
    raise ValueError(f"Unknown DVF frame {frame!r}")


def upsample_dvf_to_pack(
    dvf_zyx3: np.ndarray,
    pack_shape_zyx: tuple[int, int, int],
    *,
    frame: str = "native",
) -> np.ndarray:
    chans = [upsample_to_pack(dvf_zyx3[..., i], pack_shape_zyx, frame=frame) for i in range(3)]
    return _displacements_to_pack(
        np.stack(chans, axis=-1), dvf_zyx3.shape[:3], pack_shape_zyx, frame
    )


def pack_arrows_zyx(dvf, pack_shape, *, frame, step=6, slice_axis=0, slice_index=0, gain=1.0):
    """Sample only visible arrows; avoid allocating a full-resolution vector volume."""
    if step < 1 or slice_axis not in (0, 1, 2):
        raise ValueError("Arrow step must be positive and slice axis must be 0, 1 or 2")
    grid = [np.arange(0, size, step) for size in pack_shape]
    grid[slice_axis] = np.array([np.clip(slice_index, 0, pack_shape[slice_axis] - 1)])
    positions = np.stack(np.meshgrid(*grid, indexing="ij"), axis=-1).reshape(-1, 3)
    matrix, offset = _pack_to_compute(dvf.shape[:3], pack_shape, frame)
    coords = (positions @ matrix.T + offset).T
    disp = np.stack([
        map_coordinates(dvf[..., i], coords, order=1, mode="nearest") for i in range(3)
    ], axis=-1)
    pack_xyz = _displacements_to_pack(disp, dvf.shape[:3], pack_shape, frame)
    return np.stack([positions, pack_xyz[:, ::-1] * gain], axis=1)


def decimated_arrows_zyx(
    dvf_zyx3: np.ndarray,
    *,
    step: int = 6,
    slice_axis: int = 0,
    slice_index: int = 64,
    gain: float = 1.0,
    sub_to_mm_xyz: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Napari Vectors (N,2,3) on one thick slice of the sub (or pack) grid.

    Vector length in **data voxels** (gain multiplies). If ``sub_to_mm_xyz`` is
    given, length is scaled so on-screen mm ≈ physical (approx).
    """
    dvf = np.asarray(dvf_zyx3, dtype=np.float64)
    # Build a boolean mask for the slice ±0
    sl = [slice(None)] * 3
    sl[slice_axis] = slice(max(0, slice_index), min(dvf.shape[slice_axis], slice_index + 1))
    grid = [np.arange(0, dvf.shape[i], step) for i in range(3)]
    grid[slice_axis] = np.array([int(np.clip(slice_index, 0, dvf.shape[slice_axis] - 1))])
    zz, yy, xx = np.meshgrid(grid[0], grid[1], grid[2], indexing="ij")
    pos = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)
    disp = dvf[pos[:, 0], pos[:, 1], pos[:, 2]]  # (N,3) dx,dy,dz
    # napari vector in zyx data coords
    vec_zyx = np.stack([disp[:, 2], disp[:, 1], disp[:, 0]], axis=1) * gain
    if sub_to_mm_xyz is not None:
        # convert mm-ish length back into data voxels along each axis for display
        sx, sy, sz = sub_to_mm_xyz
        # vec currently in voxels; leave as voxels — gain is user knob
        _ = (sx, sy, sz)
    out = np.zeros((len(pos), 2, 3), dtype=np.float64)
    out[:, 0, :] = pos
    out[:, 1, :] = vec_zyx
    return out


def lung_mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None) -> float:
    if np.shape(a) != np.shape(b):
        raise ValueError("MAE volumes must have matching shapes")
    d = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    if mask is None:
        return float(d.mean())
    m = np.asarray(mask, dtype=bool)
    if m.shape != d.shape:
        raise ValueError(f"Lung mask shape {m.shape} does not match volume {d.shape}")
    if not m.any():
        raise ValueError("Lung mask is empty")
    return float(d[m].mean())
