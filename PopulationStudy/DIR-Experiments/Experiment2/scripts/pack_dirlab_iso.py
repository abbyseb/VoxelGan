#!/usr/bin/env python3
"""Pack DIR-Lab to P0-A common iso grid (2 mm, 160³) — same frame as data_iso.

Per patient:
  1. lungmask on GTVol_01 (if missing)
  2. EDT pad lung mask
  3. Lung-centroid–centred 160³ @ 2 mm (shared FOV 320 mm)
  4. Resample CT (HU) + mask onto iso grid
  5. Map landmarks → iso-grid voxel indices (continuous)
  6. Write packed_iso/P*_DIR/all/ + pack_meta.json

DVFs: run prepare_dir_dvf_library_iso.py after pack (Elastix on iso grid).

  cd PopulationStudy/DIR-Experiments/Experiment2
  /path/to/LEARN-GUI/.venv/bin/python scripts/pack_dirlab_iso.py --gpu 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates

E2 = Path(__file__).resolve().parents[1]
DIR = E2.parent
DATA_NATIVE = DIR / "data"
PACKED_ISO = E2 / "packed_iso"
SPACING_MM = 2.0
SIZE = 160


def morph_pad(mask: np.ndarray, pad_vox: int) -> np.ndarray:
    m = mask.astype(bool)
    if pad_vox == 0:
        return m
    r = abs(int(pad_vox))
    coords = np.argwhere(m)
    if coords.size == 0:
        return m
    margin = r + 2
    z0, y0, x0 = np.maximum(coords.min(axis=0) - margin, 0)
    z1, y1, x1 = np.minimum(coords.max(axis=0) + margin + 1, np.array(m.shape))
    sub = m[z0:z1, y0:y1, x0:x1]
    out = np.zeros_like(m)
    if pad_vox > 0:
        dist = ndi.distance_transform_edt(~sub)
        out[z0:z1, y0:y1, x0:x1] = dist <= r
    else:
        dist = ndi.distance_transform_edt(sub)
        out[z0:z1, y0:y1, x0:x1] = dist > r
    return out


def sample_scalar(vol: np.ndarray, zyx: np.ndarray, order: int = 1, cval: float = 0.0) -> np.ndarray:
    return map_coordinates(vol, zyx, order=order, mode="constant", cval=cval, prefilter=True)


def ensure_lung_mask(patient_dir: Path, force_cpu: bool, batch_size: int) -> Path:
    import SimpleITK as sitk
    from lungmask import LMInferer

    mask_path = patient_dir / "Mask_Lung.mha"
    if mask_path.is_file():
        return mask_path
    ct_path = patient_dir / "GTVol_01.mha"
    print(f"  lungmask R231 on {ct_path.name} …", flush=True)
    ct = sitk.ReadImage(str(ct_path))
    inferer = LMInferer(modelname="R231", force_cpu=force_cpu, batch_size=batch_size, tqdm_disable=True)
    seg = inferer.apply(ct)
    mask_np = (np.asarray(seg) > 0).astype(np.uint8)
    mask = sitk.GetImageFromArray(mask_np)
    mask.CopyInformation(ct)
    sitk.WriteImage(mask, str(mask_path), useCompression=True)
    return mask_path


def load_landmarks_xyz(path: Path) -> np.ndarray:
    pts = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 3:
            continue
        pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.asarray(pts, dtype=np.float64)


def index_to_physical_zyx(idx_zyx: np.ndarray, origin_zyx, spacing_zyx) -> np.ndarray:
    """Continuous index (z,y,x) → physical mm (z,y,x)."""
    o = np.asarray(origin_zyx, dtype=np.float64)
    s = np.asarray(spacing_zyx, dtype=np.float64)
    return o + idx_zyx * s


def physical_to_iso_index(phys_zyx: np.ndarray, origin_iso_zyx: np.ndarray) -> np.ndarray:
    """phys_zyx (N,3) z,y,x mm → iso grid indices."""
    o = np.asarray(origin_iso_zyx, dtype=np.float64)
    return (phys_zyx - o) / SPACING_MM


def pack_patient(
    pid: str,
    *,
    mask_pad: int,
    force_cpu: bool,
    batch_size: int,
    skip_existing: bool,
):
    import SimpleITK as sitk

    src = DATA_NATIVE / pid
    out_all = PACKED_ISO / pid / "all"
    meta_path = PACKED_ISO / pid / "pack_meta.json"
    if skip_existing and (out_all / "CT_01.npy").is_file() and meta_path.is_file():
        print(f"[{pid}] skip (packed_iso exists)", flush=True)
        return

    meta_native = json.loads((src / "metadata.json").read_text())
    spacing_xyz = np.array(meta_native["spacing_xyz_mm"], dtype=np.float64)
    spacing_zyx = spacing_xyz[[2, 1, 0]]

    ensure_lung_mask(src, force_cpu=force_cpu, batch_size=batch_size)
    ct_img = sitk.ReadImage(str(src / "GTVol_01.mha"))
    origin_xyz = np.array(ct_img.GetOrigin(), dtype=np.float64)
    origin_zyx = origin_xyz[[2, 1, 0]]

    mask_img = sitk.ReadImage(str(src / "Mask_Lung.mha"))
    mask_raw = sitk.GetArrayFromImage(mask_img) > 0
    mask_pad_arr = morph_pad(mask_raw, mask_pad)

    coords = np.argwhere(mask_raw)
    centroid_idx = coords.mean(axis=0)
    centroid_phys = index_to_physical_zyx(centroid_idx, origin_zyx, spacing_zyx)

    half = 0.5 * SIZE * SPACING_MM
    origin_iso = centroid_phys - half

    ii = np.arange(SIZE, dtype=np.float64)
    zz, yy, xx = np.meshgrid(
        origin_iso[0] + ii * SPACING_MM,
        origin_iso[1] + ii * SPACING_MM,
        origin_iso[2] + ii * SPACING_MM,
        indexing="ij",
    )
    world_zyx = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=0)

    idx_native = np.stack(
        [
            (world_zyx[0] - origin_zyx[0]) / spacing_zyx[0],
            (world_zyx[1] - origin_zyx[1]) / spacing_zyx[1],
            (world_zyx[2] - origin_zyx[2]) / spacing_zyx[2],
        ],
        axis=0,
    )

    out_all.mkdir(parents=True, exist_ok=True)
    mask_u = sample_scalar(mask_raw.astype(np.float32), idx_native, order=0, cval=0.0)
    mask_u = (mask_u.reshape(SIZE, SIZE, SIZE) > 0.5).astype(np.uint8)
    mask_p = sample_scalar(mask_pad_arr.astype(np.float32), idx_native, order=0, cval=0.0)
    mask_p = (mask_p.reshape(SIZE, SIZE, SIZE) > 0.5).astype(np.uint8)
    np.save(out_all / "Mask_Lung_unpadded.npy", mask_u)
    np.save(out_all / "Mask_Lung.npy", mask_p)

    air = float(sitk.GetArrayFromImage(ct_img).min())
    for ph in range(1, 11):
        vol = sitk.GetArrayFromImage(sitk.ReadImage(str(src / f"GTVol_{ph:02d}.mha"))).astype(np.float32)
        samp = sample_scalar(vol, idx_native, order=1, cval=air).reshape(SIZE, SIZE, SIZE)
        np.save(out_all / f"CT_{ph:02d}.npy", samp.astype(np.float32))
        print(f"[{pid}] CT_{ph:02d} HU[{vol.min():.0f},{vol.max():.0f}]", flush=True)

    lm_out = PACKED_ISO / pid / "landmarks"
    lm_out.mkdir(parents=True, exist_ok=True)
    lm_index = {}
    for lm_path in sorted((src / "landmarks").glob("*.txt")):
        native_xyz = load_landmarks_xyz(lm_path)
        native_zyx = native_xyz[:, [2, 1, 0]]
        phys = index_to_physical_zyx(native_zyx, origin_zyx, spacing_zyx)
        iso_zyx = physical_to_iso_index(phys, origin_iso)
        iso_xyz = iso_zyx[:, [2, 1, 0]]
        np.save(lm_out / f"{lm_path.stem}_native_xyz.npy", native_xyz.astype(np.float64))
        np.save(lm_out / f"{lm_path.stem}_iso_xyz.npy", iso_xyz.astype(np.float64))
        lm_index[lm_path.name] = {
            "n": int(len(native_xyz)),
            "native_npy": f"landmarks/{lm_path.stem}_native_xyz.npy",
            "iso_npy": f"landmarks/{lm_path.stem}_iso_xyz.npy",
        }

    pack_meta = {
        "patient_id": pid,
        "grid": f"iso {SPACING_MM}mm {SIZE}³",
        "spacing_mm": SPACING_MM,
        "size": SIZE,
        "fov_mm": SIZE * SPACING_MM,
        "origin_iso_zyx_mm": origin_iso.tolist(),
        "lung_centroid_zyx_mm": centroid_phys.tolist(),
        "native_spacing_xyz_mm": spacing_xyz.tolist(),
        "native_origin_xyz_mm": origin_xyz.tolist(),
        "mask_pad_vox": mask_pad,
        "intensity": "HU (same as data_iso SPARE)",
        "dvf_units": "iso_grid_voxels",
        "dvf_mm_per_voxel": SPACING_MM,
        "landmarks": lm_index,
        "mask_voxels_packed": int(mask_p.sum()),
    }
    meta_path.write_text(json.dumps(pack_meta, indent=2) + "\n")
    print(f"[{pid}] wrote {out_all} mask_vox={int(mask_p.sum())}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patients", default=",".join(f"P{i}_DIR" for i in range(1, 11)))
    ap.add_argument("--mask_pad", type=int, default=8)
    ap.add_argument("--force-cpu", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()

    try:
        import SimpleITK  # noqa: F401
        from lungmask import LMInferer  # noqa: F401
    except ImportError as e:
        print(f"Missing dependency: {e}\nUse LEARN-GUI venv python.", file=sys.stderr)
        sys.exit(1)

    patients = [p.strip() for p in args.patients.split(",") if p.strip()]
    print(f"Pack DIR-Lab → iso {SPACING_MM}mm {SIZE}³  n={len(patients)}", flush=True)
    for pid in patients:
        pack_patient(
            pid,
            mask_pad=args.mask_pad,
            force_cpu=args.force_cpu,
            batch_size=args.batch_size,
            skip_existing=not args.redo,
        )
    grid_meta = {
        "spacing_mm": SPACING_MM,
        "size": SIZE,
        "fov_mm": SIZE * SPACING_MM,
        "patients": patients,
        "matches": "PopulationStudy/data_iso grid.json (P0-A)",
    }
    (PACKED_ISO / "grid.json").write_text(json.dumps(grid_meta, indent=2) + "\n")
    print(f"DONE → {PACKED_ISO}", flush=True)


if __name__ == "__main__":
    main()
