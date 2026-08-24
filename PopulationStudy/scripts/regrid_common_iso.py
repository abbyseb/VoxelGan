#!/usr/bin/env python3
"""P0-A: regrid SPARE Elastix library to a common isotropic mm grid.

Output: PopulationStudy/data_iso/P*/all/
  CT_01..10.npy, Mask_Lung.npy, Mask_Lung_unpadded.npy
  {rr}_to_{tt}_pair.npy  — DVF in **iso-grid voxels** (1 vx = spacing_mm)
  meta.json

Default grid (ChangesNeeded / AmplitudeConditioning):
  spacing = 2.0 mm isotropic, size = 160³ (FOV 320 mm), centred on unpadded lung centroid.

DVF path: recover original 128³ anisotropic crop → convert vectors to mm →
resample onto common grid → store as iso-voxels (mm / spacing).

  cd PopulationStudy
  /path/to/venv/bin/python scripts/regrid_common_iso.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

POP = Path(__file__).resolve().parents[1]
RAW = POP / "raw"
PADDED = POP / "PaddedLungMasks"
DATA128 = POP / "data"
OUT = POP / "data_iso"


def find_padded_mask(patient: str) -> Path:
    pats = list(PADDED.glob(f"Mask_Lung_pad*_P{patient[1:]}.mha"))
    if not pats:
        raise FileNotFoundError(patient)
    return sorted(pats, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def bbox_from_mask(mask: np.ndarray, pad: int = 8):
    coords = np.argwhere(mask > 0)
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1
    z0 = max(0, int(z0) - pad)
    y0 = max(0, int(y0) - pad)
    x0 = max(0, int(x0) - pad)
    z1 = min(mask.shape[0], int(z1) + pad)
    y1 = min(mask.shape[1], int(y1) + pad)
    x1 = min(mask.shape[2], int(x1) + pad)
    return z0, z1, y0, y1, x0, x1


def load_mha(path: Path) -> np.ndarray:
    import itk

    return itk.array_from_image(itk.imread(str(path), itk.F)).astype(np.float32)


def load_mha_mask(path: Path) -> np.ndarray:
    import itk

    return (itk.array_from_image(itk.imread(str(path))) > 0).astype(np.uint8)


def sample_scalar(vol: np.ndarray, zyx: np.ndarray, order: int = 1, cval: float = 0.0) -> np.ndarray:
    """zyx: (3, N) continuous indices into vol."""
    return map_coordinates(vol, zyx, order=order, mode="constant", cval=cval, prefilter=True)


def sample_dvf_zyx3(dvf: np.ndarray, zyx: np.ndarray) -> np.ndarray:
    """dvf (Z,Y,X,3) channels dx,dy,dz → (3, N)."""
    out = np.zeros((3, zyx.shape[1]), dtype=np.float64)
    for c in range(3):
        out[c] = sample_scalar(dvf[..., c], zyx, order=1, cval=0.0)
    return out


def process_patient(
    pid: str,
    spacing_mm: float,
    size: int,
    bbox_pad: int,
    out_size_128: int = 128,
):
    import itk

    half = 0.5 * size * spacing_mm  # 160 mm for defaults
    raw_dir = RAW / pid
    lung_raw = load_mha_mask(raw_dir / "Mask_Lung.mha")

    coords = np.argwhere(lung_raw > 0)
    centroid = coords.mean(axis=0)  # z,y,x in raw 1mm indices
    # common grid origin (corner of voxel 0) in raw mm (== index since 1mm)
    origin = centroid - half

    pad_mask = load_mha_mask(find_padded_mask(pid))
    z0, z1, y0, y1, x0, x1 = bbox_from_mask(pad_mask, pad=bbox_pad)
    extent = np.array([z1 - z0, y1 - y0, x1 - x0], dtype=np.float64)
    sp128 = extent / float(out_size_128)  # mm per 128-voxel along z,y,x

    # mesh of common-grid centers in world/raw mm
    # use voxel-corner indexing consistent with map_coordinates
    ii = np.arange(size, dtype=np.float64)
    zz, yy, xx = np.meshgrid(
        origin[0] + ii * spacing_mm,
        origin[1] + ii * spacing_mm,
        origin[2] + ii * spacing_mm,
        indexing="ij",
    )
    world = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=0)  # (3, N)

    # indices into 128 crop
    idx128 = np.stack(
        [
            (world[0] - z0) / sp128[0],
            (world[1] - y0) / sp128[1],
            (world[2] - x0) / sp128[2],
        ],
        axis=0,
    )
    # indices into raw 1mm
    idx_raw = world.copy()  # 1mm spacing, origin 0

    out_dir = OUT / pid / "all"
    out_dir.mkdir(parents=True, exist_ok=True)

    # unpadded + padded lung on common grid (nearest)
    mask_u = sample_scalar(lung_raw.astype(np.float32), idx_raw, order=0, cval=0.0)
    mask_u = (mask_u.reshape(size, size, size) > 0.5).astype(np.uint8)
    mask_p = sample_scalar(pad_mask.astype(np.float32), idx_raw, order=0, cval=0.0)
    mask_p = (mask_p.reshape(size, size, size) > 0.5).astype(np.uint8)
    np.save(out_dir / "Mask_Lung_unpadded.npy", mask_u)
    np.save(out_dir / "Mask_Lung.npy", mask_p)  # training mask (padded, matches Elastix FOV)

    air = float(load_mha(raw_dir / "GTVol_01.mha").min())
    for ph in range(1, 11):
        ct = load_mha(raw_dir / f"GTVol_{ph:02d}.mha")
        samp = sample_scalar(ct, idx_raw, order=1, cval=air).reshape(size, size, size)
        np.save(out_dir / f"CT_{ph:02d}.npy", samp.astype(np.float32))
        print(f"  [{pid}] CT_{ph:02d}", flush=True)

    src128 = DATA128 / pid / "all"
    for ref in range(1, 11):
        for tgt in range(1, 11):
            name = f"{ref:02d}_to_{tgt:02d}_pair.npy"
            if ref == tgt:
                dvf_iso = np.zeros((size, size, size, 3), dtype=np.float32)
            else:
                u = np.load(src128 / name)
                if u.ndim == 4 and u.shape[0] == 3:
                    u = np.moveaxis(u, 0, -1)
                # sample in 128 voxel units
                vox = sample_dvf_zyx3(u, idx128)  # (3,N) dx,dy,dz in 128-voxels
                # → mm (LR, AP, SI) = (x,y,z) channels
                mm = np.zeros_like(vox)
                mm[0] = vox[0] * sp128[2]  # dx * sp_x
                mm[1] = vox[1] * sp128[1]  # dy * sp_y
                mm[2] = vox[2] * sp128[0]  # dz * sp_z
                # → iso-grid voxels
                iso = (mm / spacing_mm).astype(np.float32)
                dvf_iso = np.moveaxis(iso.reshape(3, size, size, size), 0, -1)
            np.save(out_dir / name, dvf_iso.astype(np.float32))
        print(f"  [{pid}] DVF ref={ref:02d} done", flush=True)

    meta = {
        "patient": pid,
        "spacing_mm": spacing_mm,
        "size": size,
        "fov_mm": size * spacing_mm,
        "origin_zyx_mm": origin.tolist(),
        "lung_centroid_zyx_mm": centroid.tolist(),
        "bbox128_zyx": [int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)],
        "spacing128_zyx_mm": sp128.tolist(),
        "dvf_units": "iso_grid_voxels",
        "dvf_mm_per_voxel": spacing_mm,
        "note": "DVF channels (dx,dy,dz)=(LR,AP,SI). Multiply by spacing_mm for mm.",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (OUT / pid / "meta.json").write_text(json.dumps(meta, indent=2))
    print(
        f"[{pid}] wrote {out_dir} | lung_u={int(mask_u.sum())} lung_p={int(mask_p.sum())} "
        f"centroid={centroid.round(1).tolist()} sp128={sp128.round(3).tolist()}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spacing_mm", type=float, default=2.0)
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--bbox_pad", type=int, default=8)
    ap.add_argument("--patients", default="P1,P2,P3,P4,P5,P6,P7,P8,P9")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    patients = [p.strip() for p in args.patients.split(",") if p.strip()]
    for pid in patients:
        print(f"=== {pid} ===", flush=True)
        process_patient(pid, args.spacing_mm, args.size, args.bbox_pad)
    (OUT / "grid.json").write_text(
        json.dumps(
            {
                "spacing_mm": args.spacing_mm,
                "size": args.size,
                "fov_mm": args.size * args.spacing_mm,
                "patients": patients,
            },
            indent=2,
        )
    )
    print(f"done → {OUT}", flush=True)


if __name__ == "__main__":
    main()
