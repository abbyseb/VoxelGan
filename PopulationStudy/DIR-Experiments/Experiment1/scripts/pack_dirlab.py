#!/usr/bin/env python3
"""Pack DIR-Lab patients to SPARE-like 128³ µ volumes for E6 QC.

Per patient:
  1. lungmask R231 on GTVol_01 → Mask_Lung.mha (native)
  2. EDT pad → lung bbox crop → resample CT+mask → 128³
  3. HU → µ  (µ = 0.02 * (1 + HU/1000))
  4. Remap landmarks through the same transform; save pack_meta.json

  cd PopulationStudy/DIR-Experiments/Experiment1
  /path/to/LEARN-GUI/.venv/bin/python scripts/pack_dirlab.py --gpu 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

E1 = Path(__file__).resolve().parents[1]
DIR = E1.parent
DATA_NATIVE = DIR / "data"
PACKED = E1 / "packed"
MU_WATER = 0.02


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


def bbox_from_mask(mask_zyx: np.ndarray, pad: int = 8):
    coords = np.argwhere(mask_zyx > 0)
    if coords.size == 0:
        raise RuntimeError("Lung mask is empty")
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1
    z0 = max(0, z0 - pad)
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    z1 = min(mask_zyx.shape[0], z1 + pad)
    y1 = min(mask_zyx.shape[1], y1 + pad)
    x1 = min(mask_zyx.shape[2], x1 + pad)
    return (int(z0), int(z1), int(y0), int(y1), int(x0), int(x1))


def resample_zyx(vol_zyx: np.ndarray, out_size: int, is_mask: bool = False) -> np.ndarray:
    import itk

    if is_mask:
        src = itk.GetImageFromArray(vol_zyx.astype(np.uint8))
    else:
        src = itk.GetImageFromArray(vol_zyx.astype(np.float32))
    src.SetSpacing((1.0, 1.0, 1.0))
    src.SetOrigin((0.0, 0.0, 0.0))

    in_size_xyz = np.array(
        [vol_zyx.shape[2], vol_zyx.shape[1], vol_zyx.shape[0]], dtype=np.float64
    )
    out_size_xyz = np.array([out_size, out_size, out_size], dtype=np.float64)
    out_spacing = (in_size_xyz / out_size_xyz).tolist()

    interpolator = (
        itk.NearestNeighborInterpolateImageFunction.New(src)
        if is_mask
        else itk.LinearInterpolateImageFunction.New(src)
    )
    resample = itk.ResampleImageFilter.New(src)
    resample.SetInterpolator(interpolator)
    resample.SetSize([int(out_size)] * 3)
    resample.SetOutputSpacing(out_spacing)
    resample.SetOutputOrigin(src.GetOrigin())
    resample.SetOutputDirection(src.GetDirection())
    resample.SetDefaultPixelValue(0 if is_mask else float(vol_zyx.min()))
    resample.Update()
    out = itk.array_from_image(resample.GetOutput())
    if is_mask:
        return (out > 0).astype(np.uint8)
    return out.astype(np.float32)


def hu_to_mu(hu: np.ndarray, mu_water: float = MU_WATER) -> np.ndarray:
    """µ = µ_water * (1 + HU/1000). Air HU≈-1000 → 0; water HU≈0 → µ_water."""
    return (mu_water * (1.0 + hu.astype(np.float32) / 1000.0)).astype(np.float32)


def load_landmarks_xyz(path: Path) -> np.ndarray:
    """Nx3 float array in (x, y, z) voxel indices."""
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


def native_xyz_to_packed(xyz: np.ndarray, bbox, out_size: int) -> np.ndarray:
    """Map native (x,y,z) voxel → packed (x,y,z) in [0, out_size)."""
    z0, z1, y0, y1, x0, x1 = bbox
    sx = out_size / max(x1 - x0, 1)
    sy = out_size / max(y1 - y0, 1)
    sz = out_size / max(z1 - z0, 1)
    out = np.empty_like(xyz)
    out[:, 0] = (xyz[:, 0] - x0) * sx
    out[:, 1] = (xyz[:, 1] - y0) * sy
    out[:, 2] = (xyz[:, 2] - z0) * sz
    return out


def packed_xyz_to_native(xyz: np.ndarray, bbox, out_size: int) -> np.ndarray:
    z0, z1, y0, y1, x0, x1 = bbox
    sx = out_size / max(x1 - x0, 1)
    sy = out_size / max(y1 - y0, 1)
    sz = out_size / max(z1 - z0, 1)
    out = np.empty_like(xyz)
    out[:, 0] = xyz[:, 0] / sx + x0
    out[:, 1] = xyz[:, 1] / sy + y0
    out[:, 2] = xyz[:, 2] / sz + z0
    return out


def ensure_lung_mask(patient_dir: Path, force_cpu: bool, batch_size: int) -> Path:
    import SimpleITK as sitk
    from lungmask import LMInferer

    mask_path = patient_dir / "Mask_Lung.mha"
    if mask_path.is_file():
        print(f"  mask exists: {mask_path.name}", flush=True)
        return mask_path

    ct_path = patient_dir / "GTVol_01.mha"
    print(f"  lungmask R231 on {ct_path.name} …", flush=True)
    ct = sitk.ReadImage(str(ct_path))
    inferer = LMInferer(
        modelname="R231",
        force_cpu=force_cpu,
        batch_size=batch_size,
        tqdm_disable=True,
    )
    seg = inferer.apply(ct)
    mask_np = (np.asarray(seg) > 0).astype(np.uint8)
    mask = sitk.GetImageFromArray(mask_np)
    mask.CopyInformation(ct)
    sitk.WriteImage(mask, str(mask_path), useCompression=True)
    print(f"  mask voxels={int(mask_np.sum())} frac={mask_np.mean():.3f}", flush=True)
    return mask_path


def to_spare_axes(vol_zyx: np.ndarray) -> np.ndarray:
    """DIR axial (Z_si,Y,X) → SPARE-like (Y, Z_si, X) so SI is middle axis.

    SPARE GTVol array shape is (Z, Y_si, X); DIR is (Z_si, Y, X). E6 learned
    breathing along ITK Y / array axis 1 — match that before crop/resample.
    """
    return np.transpose(vol_zyx, (1, 0, 2))


def landmarks_dir_to_spare_xyz(xyz: np.ndarray) -> np.ndarray:
    """DIR ITK (x,y,z_si) → SPARE-like ITK (x, z_si, y) after to_spare_axes."""
    out = np.empty_like(xyz)
    out[:, 0] = xyz[:, 0]  # x
    out[:, 1] = xyz[:, 2]  # y' = z_si
    out[:, 2] = xyz[:, 1]  # z' = y
    return out


def spacing_dir_to_spare(spacing_xyz):
    sx, sy, sz = spacing_xyz
    return [float(sx), float(sz), float(sy)]


def pack_patient(
    pid: str,
    *,
    packed_root: Path,
    layout: str,
    out_size: int,
    bbox_pad: int,
    mask_pad: int,
    force_cpu: bool,
    batch_size: int,
    skip_existing: bool,
):
    import SimpleITK as sitk

    src = DATA_NATIVE / pid
    out_all = packed_root / pid / "all"
    meta_path = packed_root / pid / "pack_meta.json"
    if skip_existing and (out_all / "CT_01.npy").is_file() and meta_path.is_file():
        print(f"[{pid}] skip (packed exists)", flush=True)
        return

    if not src.is_dir():
        raise FileNotFoundError(src)

    meta_native = json.loads((src / "metadata.json").read_text())
    spacing_dir = list(meta_native["spacing_xyz_mm"])
    use_spare = layout == "spare_axes"
    spacing = spacing_dir_to_spare(spacing_dir) if use_spare else spacing_dir

    ensure_lung_mask(src, force_cpu=force_cpu, batch_size=batch_size)
    mask_img = sitk.ReadImage(str(src / "Mask_Lung.mha"))
    mask_raw = sitk.GetArrayFromImage(mask_img) > 0
    if use_spare:
        mask_raw = to_spare_axes(mask_raw)
    mask = morph_pad(mask_raw, mask_pad)
    bbox = bbox_from_mask(mask, pad=bbox_pad)
    z0, z1, y0, y1, x0, x1 = bbox
    print(
        f"[{pid}] layout={layout} bbox zyx=[{z0}:{z1},{y0}:{y1},{x0}:{x1}] "
        f"spacing={spacing_dir}",
        flush=True,
    )

    cts = []
    for ph in range(1, 11):
        vol = sitk.GetArrayFromImage(sitk.ReadImage(str(src / f"GTVol_{ph:02d}.mha"))).astype(
            np.float32
        )
        if use_spare:
            vol = to_spare_axes(vol)
        cropped = vol[z0:z1, y0:y1, x0:x1]
        packed = resample_zyx(cropped, out_size, is_mask=False)
        cts.append(hu_to_mu(packed))
        print(
            f"[{pid}]   CT_{ph:02d} HU[{vol.min():.0f},{vol.max():.0f}] → "
            f"µ[{cts[-1].min():.4f},{cts[-1].max():.4f}] shape={cts[-1].shape}",
            flush=True,
        )

    mask_r = resample_zyx(mask[z0:z1, y0:y1, x0:x1].astype(np.uint8), out_size, is_mask=True)
    out_all.mkdir(parents=True, exist_ok=True)
    for i, ct in enumerate(cts, 1):
        np.save(out_all / f"CT_{i:02d}.npy", ct.astype(np.float32))
    np.save(out_all / "Mask_Lung.npy", mask_r.astype(np.uint8))

    lm_out = packed_root / pid / "landmarks"
    lm_out.mkdir(parents=True, exist_ok=True)
    lm_index = {}
    for lm_path in sorted((src / "landmarks").glob("*.txt")):
        native_dir = load_landmarks_xyz(lm_path)
        pack_xyz = landmarks_dir_to_spare_xyz(native_dir) if use_spare else native_dir
        packed_lm = native_xyz_to_packed(pack_xyz, bbox, out_size)
        np.save(lm_out / f"{lm_path.stem}_native_xyz.npy", native_dir.astype(np.float64))
        np.save(lm_out / f"{lm_path.stem}_packed_xyz.npy", packed_lm.astype(np.float64))
        if use_spare:
            np.save(lm_out / f"{lm_path.stem}_spare_xyz.npy", pack_xyz.astype(np.float64))
        lm_index[lm_path.name] = {
            "n": int(len(native_dir)),
            "native_npy": f"landmarks/{lm_path.stem}_native_xyz.npy",
            "packed_npy": f"landmarks/{lm_path.stem}_packed_xyz.npy",
        }

    if use_spare:
        axis_note = "DIR (Z_si,Y,X) → SPARE-like (Y,Z_si,X); landmarks (x,y,z)→(x,z,y)"
        layout_key = "spare_axes"
    else:
        axis_note = "native axial DIR (no axis permute)"
        layout_key = "axial"

    pack_meta = {
        "patient_id": pid,
        "source": str(src),
        "layout": layout_key,
        "out_size": out_size,
        "axis_remap": axis_note,
        "bbox_zyx": [z0, z1, y0, y1, x0, x1],
        "mask_pad_vox": mask_pad,
        "bbox_pad": bbox_pad,
        "native_spacing_xyz_mm": spacing_dir,
        "hu_to_mu": f"mu = {MU_WATER} * (1 + HU/1000)",
        "mu_water": MU_WATER,
        "mask_voxels_packed": int(mask_r.sum()),
        "landmarks": lm_index,
        "crop_size_zyx": [z1 - z0, y1 - y0, x1 - x0],
    }
    if use_spare:
        pack_meta["spare_spacing_xyz_mm"] = spacing
    meta_path.write_text(json.dumps(pack_meta, indent=2) + "\n")
    print(f"[{pid}] wrote {out_all}  mask_vox={int(mask_r.sum())}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patients", default=",".join(f"P{i}_DIR" for i in range(1, 11)))
    ap.add_argument("--out_size", type=int, default=128)
    ap.add_argument("--bbox_pad", type=int, default=8)
    ap.add_argument("--mask_pad", type=int, default=8)
    ap.add_argument("--force-cpu", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument(
        "--layout",
        choices=("spare_axes", "axial"),
        default="spare_axes",
        help="spare_axes=match E6 coronal SI-on-Y; axial=native DIR layout (Phase 1)",
    )
    ap.add_argument(
        "--packed-dir",
        type=Path,
        default=None,
        help="default: Experiment1/packed or packed_axial if --layout axial",
    )
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()

    packed_root = args.packed_dir or (
        E1 / ("packed_axial" if args.layout == "axial" else "packed")
    )
    global PACKED
    PACKED = packed_root

    try:
        import itk  # noqa: F401
        import SimpleITK  # noqa: F401
        from lungmask import LMInferer  # noqa: F401
    except ImportError as e:
        print(
            f"Missing dependency: {e}\n"
            "Use LEARN-GUI venv python (has lungmask + itk).",
            file=sys.stderr,
        )
        sys.exit(1)

    patients = [p.strip() for p in args.patients.split(",") if p.strip()]
    print(f"Pack DIR-Lab → {packed_root}  layout={args.layout}  n={len(patients)}", flush=True)
    for pid in patients:
        pack_patient(
            pid,
            packed_root=packed_root,
            layout=args.layout,
            out_size=args.out_size,
            bbox_pad=args.bbox_pad,
            mask_pad=args.mask_pad,
            force_cpu=args.force_cpu,
            batch_size=args.batch_size,
            skip_existing=not args.redo,
        )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
