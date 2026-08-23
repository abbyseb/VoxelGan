#!/usr/bin/env python3
"""Stack LIDC-IDRI DICOM → HU MHA and run lungmask R231.

Writes per patient:
  data/lidc-idri/volumes/<PatientID>/CT.mha
  data/lidc-idri/volumes/<PatientID>/Mask_Lung.mha   # binary (R+L)

R231 labels 1=right, 2=left are collapsed. Cite Hofmanninger et al. 2020
https://doi.org/10.1186/s41747-020-00173-2
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parent
DICOM = ROOT / "data" / "lidc-idri" / "dicom"
VOLS = ROOT / "data" / "lidc-idri" / "volumes"
MANIFEST = ROOT / "lidc_subset_n40_seed20260819.csv"


def read_series(dicom_dir: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    uids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if not uids:
        raise RuntimeError(f"No DICOM series in {dicom_dir}")
    best, nbest = None, 0
    for uid in uids:
        names = reader.GetGDCMSeriesFileNames(str(dicom_dir), uid)
        if len(names) > nbest:
            best, nbest = names, len(names)
    reader.SetFileNames(best)
    return reader.Execute()


def binary_lung(seg: np.ndarray) -> np.ndarray:
    return (seg > 0).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--out", type=Path, default=VOLS)
    ap.add_argument("--force-cpu", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()
    if args.redo:
        args.skip_existing = False

    from lungmask import LMInferer

    rows = list(csv.DictReader(args.manifest.open()))
    inferer = LMInferer(
        modelname="R231",
        force_cpu=args.force_cpu,
        batch_size=args.batch_size,
        tqdm_disable=False,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Patients: {len(rows)}  out={args.out}  cpu={args.force_cpu}", flush=True)

    for i, row in enumerate(rows, start=1):
        pid = row["PatientID"]
        uid = row["SeriesInstanceUID"]
        out_dir = args.out / pid
        ct_path = out_dir / "CT.mha"
        mask_path = out_dir / "Mask_Lung.mha"
        if args.skip_existing and ct_path.exists() and mask_path.exists():
            print(f"[{i}/{len(rows)}] {pid} skip (exists)", flush=True)
            continue
        dicom_dir = DICOM / uid
        print(f"[{i}/{len(rows)}] {pid}  reading {dicom_dir.name}", flush=True)
        ct = read_series(dicom_dir)
        print(f"    CT {ct.GetSize()} spacing={tuple(round(s, 3) for s in ct.GetSpacing())}", flush=True)
        seg = inferer.apply(ct)
        mask_np = binary_lung(np.asarray(seg))
        mask = sitk.GetImageFromArray(mask_np)
        mask.CopyInformation(ct)
        out_dir.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(ct, str(ct_path), useCompression=True)
        sitk.WriteImage(mask, str(mask_path), useCompression=True)
        frac = float(mask_np.mean())
        print(
            f"    mask voxels={int(mask_np.sum())}  frac={frac:.3f}  -> {out_dir.name}",
            flush=True,
        )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
