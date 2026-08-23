#!/usr/bin/env python3
"""Build PopulationStudy DVF library via masked B-spline Elastix.

For each patient P1–P9:
  - GTVol from raw/P*/
  - Lung mask from PaddedLungMasks/Mask_Lung_pad*_P*.mha
  - Crop to padded-lung bbox → resample 128³
  - Register all 10×10 directed phase pairs (identity = zeros)
  - Write data/P{k}/all/  (CT_*.npy, Mask_Lung.npy, {rr}_to_{tt}_pair.npy)

Also mirrors train/val leave-phase-out 5&9 under data/P{k}/{train,val}/ for bookkeeping.

Requires LEARN-GUI venv (itk-elastix):
  /home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python \\
      scripts/prepare_dvf_library.py --patients P1,P2
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

POP = Path(__file__).resolve().parents[1]
REPO = POP.parent
RAW = POP / "raw"
PADDED = POP / "PaddedLungMasks"
DATA = POP / "data"
PARAM_DEFAULT = REPO / "My v1.0" / "configs" / "elastix_bspline_masked.txt"


def _bbox_from_mask(mask_zyx: np.ndarray, pad: int = 8):
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
    return (z0, z1, y0, y1, x0, x1)


def _resample_zyx(vol_zyx: np.ndarray, out_size: int, is_mask: bool = False) -> np.ndarray:
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


def _to_itk_float(vol_zyx: np.ndarray):
    import itk

    img = itk.GetImageFromArray(vol_zyx.astype(np.float32))
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection(itk.matrix_from_array(np.eye(3)))
    return img


def _to_itk_mask(mask_zyx: np.ndarray):
    import itk

    img = itk.GetImageFromArray(mask_zyx.astype(np.uint8))
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection(itk.matrix_from_array(np.eye(3)))
    return img


def register_pair(fixed_zyx, moving_zyx, mask_zyx, param_file: Path):
    """Elastix: fixed=target, moving=reference. DVF warps moving → fixed."""
    import itk

    fixed = _to_itk_float(fixed_zyx)
    moving = _to_itk_float(moving_zyx)
    mask = _to_itk_mask(mask_zyx)

    param_obj = itk.ParameterObject.New()
    param_obj.AddParameterFile(str(param_file))

    ImageType = type(fixed)
    elastix = itk.ElastixRegistrationMethod[ImageType, ImageType].New()
    elastix.SetFixedImage(fixed)
    elastix.SetMovingImage(moving)
    elastix.SetFixedMask(mask)
    elastix.SetMovingMask(mask)
    elastix.SetParameterObject(param_obj)
    elastix.LogToConsoleOff()
    elastix.Update()

    transform_params = elastix.GetTransformParameterObject()
    transformix = itk.TransformixFilter[ImageType].New()
    transformix.SetMovingImage(moving)
    transformix.SetTransformParameterObject(transform_params)
    transformix.SetComputeDeformationField(True)
    transformix.Update()

    dvf_img = transformix.GetOutputDeformationField()
    return itk.array_from_image(dvf_img).astype(np.float32)


def find_padded_mask(patient: str) -> Path:
    """Match Mask_Lung_padp08_P2.mha style."""
    pats = list(PADDED.glob(f"Mask_Lung_pad*_P{patient[1:]}.mha"))
    if not pats:
        # also try P01 style
        pats = list(PADDED.glob(f"Mask_Lung_pad*_{patient}.mha"))
    if not pats:
        raise FileNotFoundError(f"No padded mask for {patient} under {PADDED}")
    if len(pats) > 1:
        pats = sorted(pats, key=lambda p: p.stat().st_mtime, reverse=True)
        print(f"  warn: multiple padded masks for {patient}, using {pats[0].name}")
    return pats[0]


def parse_pad_from_name(path: Path) -> int | None:
    m = re.search(r"pad([pm])(\d+)", path.stem)
    if not m:
        return None
    return int(m.group(2)) if m.group(1) == "p" else -int(m.group(2))


def preprocess_patient(patient: str, out_size: int, bbox_pad: int):
    import itk

    case_dir = RAW / patient
    mask_path = find_padded_mask(patient)
    pad_vx = parse_pad_from_name(mask_path)
    print(f"[{patient}] padded mask: {mask_path.name} (pad={pad_vx})")

    mask = itk.array_from_image(itk.imread(str(mask_path)))
    bbox = _bbox_from_mask(mask, pad=bbox_pad)
    z0, z1, y0, y1, x0, x1 = bbox
    print(f"[{patient}] lung bbox (zyx): [{z0}:{z1}, {y0}:{y1}, {x0}:{x1}]")

    cts = []
    for p in range(1, 11):
        vol = itk.array_from_image(
            itk.imread(str(case_dir / f"GTVol_{p:02d}.mha"), itk.F)
        ).astype(np.float32)
        cropped = vol[z0:z1, y0:y1, x0:x1]
        cts.append(_resample_zyx(cropped, out_size, is_mask=False))
        print(f"[{patient}]   CT_{p:02d}: {vol.shape} → {cropped.shape} → {cts[-1].shape}")

    mask_r = _resample_zyx(mask[z0:z1, y0:y1, x0:x1], out_size, is_mask=True)
    print(f"[{patient}]   Mask_Lung → {mask_r.shape}, voxels={int(mask_r.sum())}")
    return cts, mask_r, mask_path


def write_volumes(out_dir: Path, cts, mask):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, ct in enumerate(cts, start=1):
        np.save(out_dir / f"CT_{i:02d}.npy", ct.astype(np.float32))
    np.save(out_dir / "Mask_Lung.npy", mask.astype(np.uint8))


def process_patient(
    patient: str,
    param_file: Path,
    out_size: int,
    bbox_pad: int,
    held_out: set[int],
    skip_existing: bool,
    max_pairs: int | None,
):
    t0 = time.time()
    cts, mask, mask_path = preprocess_patient(patient, out_size, bbox_pad)

    all_dir = DATA / patient / "all"
    train_dir = DATA / patient / "train"
    val_dir = DATA / patient / "val"
    for d in (all_dir, train_dir, val_dir):
        write_volumes(d, cts, mask)

    # record which padded mask was used
    meta = DATA / patient / "source_mask.txt"
    meta.write_text(f"{mask_path}\npad_from_name={parse_pad_from_name(mask_path)}\n")

    jobs = [(i, j) for i in range(1, 11) for j in range(1, 11)]
    if max_pairs is not None:
        jobs = jobs[:max_pairs]

    n_done = n_skip = n_id = 0
    for n, (ref, tgt) in enumerate(jobs, start=1):
        split = "val" if (ref in held_out or tgt in held_out) else "train"
        name = f"{ref:02d}_to_{tgt:02d}_pair.npy"
        all_path = all_dir / name
        split_path = (val_dir if split == "val" else train_dir) / name

        if skip_existing and all_path.exists():
            dvf = np.load(all_path)
            n_skip += 1
            print(f"[{patient}] [{n}/{len(jobs)}] skip {name}", flush=True)
        elif ref == tgt:
            dvf = np.zeros(cts[0].shape + (3,), dtype=np.float32)
            n_id += 1
            print(f"[{patient}] [{n}/{len(jobs)}] identity {name}", flush=True)
        else:
            print(f"[{patient}] [{n}/{len(jobs)}] Elastix {name} ({split}) …", flush=True)
            tic = time.time()
            dvf = register_pair(cts[tgt - 1], cts[ref - 1], mask, param_file)
            print(
                f"[{patient}]     {time.time()-tic:.1f}s  "
                f"DVF range "
                f"{dvf[...,0].min():.2f}:{dvf[...,0].max():.2f}, "
                f"{dvf[...,1].min():.2f}:{dvf[...,1].max():.2f}, "
                f"{dvf[...,2].min():.2f}:{dvf[...,2].max():.2f}",
                flush=True,
            )
            n_done += 1

        np.save(all_path, dvf.astype(np.float32))
        np.save(split_path, dvf.astype(np.float32))

    elapsed = time.time() - t0
    print(
        f"[{patient}] DONE in {elapsed/60:.1f} min  "
        f"registered={n_done} identity={n_id} skipped={n_skip}  → {all_dir}",
        flush=True,
    )
    return all_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patients", default="P1,P2,P3,P4,P5,P6,P7,P8,P9")
    ap.add_argument("--out_size", type=int, default=128)
    ap.add_argument("--bbox_pad", type=int, default=8, help="Extra voxels around padded-mask bbox before resample")
    ap.add_argument("--param_file", type=Path, default=PARAM_DEFAULT)
    ap.add_argument("--held_out", type=int, nargs="+", default=[5, 9])
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--max_pairs", type=int, default=None, help="Smoke-test cap")
    args = ap.parse_args()

    try:
        import itk  # noqa: F401
    except ImportError:
        print(
            "itk-elastix missing. Use:\n"
            "  /home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python "
            "scripts/prepare_dvf_library.py",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.param_file.exists():
        raise SystemExit(f"param file not found: {args.param_file}")

    patients = [p.strip() for p in args.patients.split(",") if p.strip()]
    held = set(args.held_out)
    print(f"DVF library patients={patients}  held_out={sorted(held)}  param={args.param_file}")
    print(f"Padded masks dir: {PADDED}")

    for patient in patients:
        if not (RAW / patient).is_dir():
            raise SystemExit(f"missing raw patient dir: {RAW / patient}")
        process_patient(
            patient,
            args.param_file,
            args.out_size,
            args.bbox_pad,
            held,
            args.skip_existing,
            args.max_pairs,
        )

    print("All requested patients finished.")


if __name__ == "__main__":
    main()
