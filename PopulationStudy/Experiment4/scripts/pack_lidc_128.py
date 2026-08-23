#!/usr/bin/env python3
"""Crop LIDC R231 volumes to lung bbox and resample to 128³ (SPARE recipe).

Same as PopulationStudy/scripts/prepare_dvf_library.py preprocess:
  lung-mask bbox + bbox_pad → resample cube 128, 1 mm-equivalent after crop.

Writes Experiment4/data/lidc128/<PatientID>/{CT,Mask_Lung}.npy + manifest.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

E4 = Path(__file__).resolve().parents[1]
PS = E4.parent
VOLS = PS / "AnatomyPretrain" / "data" / "lidc-idri" / "volumes"
OUT = E4 / "data" / "lidc128"
SEED_PATH = E4 / "seed.json"


def bbox_from_mask(mask_zyx: np.ndarray, pad: int):
    coords = np.argwhere(mask_zyx > 0)
    if coords.size == 0:
        raise RuntimeError("Lung mask is empty")
    z0, y0, x0 = np.maximum(coords.min(axis=0) - pad, 0)
    z1, y1, x1 = np.minimum(coords.max(axis=0) + pad + 1, np.array(mask_zyx.shape))
    return int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)


def resample_zyx(vol_zyx: np.ndarray, out_size: int, is_mask: bool) -> np.ndarray:
    """Match Elastix prep: treat crop voxels as 1 mm, resample to out_size³."""
    if is_mask:
        img = sitk.GetImageFromArray(vol_zyx.astype(np.uint8))
    else:
        img = sitk.GetImageFromArray(vol_zyx.astype(np.float32))
    img.SetSpacing((1.0, 1.0, 1.0))
    old = np.array(img.GetSize(), dtype=np.float64)
    new_spacing = (old / float(out_size)).tolist()
    resample = sitk.ResampleImageFilter()
    resample.SetSize([int(out_size)] * 3)
    resample.SetOutputSpacing(new_spacing)
    resample.SetOutputOrigin(img.GetOrigin())
    resample.SetOutputDirection(img.GetDirection())
    resample.SetInterpolator(
        sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
    )
    resample.SetDefaultPixelValue(0 if is_mask else float(np.min(vol_zyx)))
    out = sitk.GetArrayFromImage(resample.Execute(img))
    if is_mask:
        return (out > 0).astype(np.uint8)
    return out.astype(np.float32)


def pack_one(pid: str, out_size: int, bbox_pad: int) -> dict:
    src = VOLS / pid
    ct_img = sitk.ReadImage(str(src / "CT.mha"))
    mask_img = sitk.ReadImage(str(src / "Mask_Lung.mha"))
    ct = sitk.GetArrayFromImage(ct_img).astype(np.float32)
    mask = sitk.GetArrayFromImage(mask_img) > 0
    z0, z1, y0, y1, x0, x1 = bbox_from_mask(mask, bbox_pad)
    ct_c = ct[z0:z1, y0:y1, x0:x1]
    m_c = mask[z0:z1, y0:y1, x0:x1]
    ct_r = resample_zyx(ct_c, out_size, is_mask=False)
    ct_r = np.clip(ct_r, -1024.0, 3071.0)
    m_r = resample_zyx(m_c.astype(np.uint8), out_size, is_mask=True)
    dest = OUT / pid
    dest.mkdir(parents=True, exist_ok=True)
    np.save(dest / "CT.npy", ct_r)
    np.save(dest / "Mask_Lung.npy", m_r)
    return {
        "patient": pid,
        "native_zyx": list(ct.shape),
        "bbox_zyx": [z0, z1, y0, y1, x0, x1],
        "crop_zyx": list(ct_c.shape),
        "out_zyx": list(ct_r.shape),
        "spacing_xyz": [float(s) for s in ct_img.GetSpacing()],
        "hu_min": float(ct_r.min()),
        "hu_max": float(ct_r.max()),
        "lung_frac": float(m_r.mean()),
        "lung_voxels": int(m_r.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_size", type=int, default=None)
    ap.add_argument("--bbox_pad", type=int, default=None)
    args = ap.parse_args()
    cfg = json.loads(SEED_PATH.read_text())
    out_size = args.out_size or int(cfg["pack"]["out_size"])
    bbox_pad = args.bbox_pad if args.bbox_pad is not None else int(cfg["pack"]["bbox_pad"])

    patients = sorted(p.name for p in VOLS.iterdir() if p.is_dir() and (p / "Mask_Lung.mha").exists())
    if not patients:
        raise SystemExit(f"no R231 volumes under {VOLS}")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"Packing {len(patients)} LIDC cases → {OUT}  {out_size}³  bbox_pad={bbox_pad}", flush=True)
    for i, pid in enumerate(patients, start=1):
        rec = pack_one(pid, out_size, bbox_pad)
        rows.append(rec)
        print(
            f"[{i}/{len(patients)}] {pid}  "
            f"{rec['native_zyx']} crop {rec['crop_zyx']} → {rec['out_zyx']}  "
            f"lung={rec['lung_frac']:.3f}  HU[{rec['hu_min']:.0f},{rec['hu_max']:.0f}]",
            flush=True,
        )

    rng = __import__("random").Random(int(cfg["lidc_val_split_seed"]))
    ids = [r["patient"] for r in rows]
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * float(cfg["lidc_val_fraction"]))))
    val_ids = sorted(ids[:n_val])
    train_ids = sorted(ids[n_val:])
    manifest = {
        "out_size": out_size,
        "bbox_pad": bbox_pad,
        "n": len(rows),
        "train_patients": train_ids,
        "val_patients": val_ids,
        "cases": rows,
    }
    man_path = OUT / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {man_path}  train={len(train_ids)} val={len(val_ids)}", flush=True)


if __name__ == "__main__":
    main()
