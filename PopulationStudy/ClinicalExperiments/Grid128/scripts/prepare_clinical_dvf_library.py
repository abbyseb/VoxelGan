#!/usr/bin/env python3
"""Build Clinical DVF library via masked B-spline Elastix (same as SPARE).

Supports Clinical Varian (`CV_*`) and Clinical Elekta (`CE_*`) Evaluation GT.

Reads GTVol + Mask_Lung, pads lung mask (3D EDT), crops/resamples 128³,
registers all 10×10 directed pairs.

  cd PopulationStudy/ClinicalExperiments
  python scripts/prepare_clinical_dvf_library.py --scan CV_P1_V_01
  python scripts/prepare_clinical_dvf_library.py --scan CE_P1_V_01

Requires itk-elastix (LEARN-GUI venv).
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

CE = Path(__file__).resolve().parents[1]
POP = CE.parent
REPO = POP.parent
VARIAN_EVAL = POP / "varian" / "Evaluation" / "ClinicalVarianDatasets"
ELEKTA_EVAL = POP / "elekta" / "Evaluation" / "ClinicalElektaDatasets"
SPAREDVFS_VARIAN = Path(
    "/home/abhishek/SpareDVFs/SPARE_GroundTruth/ClinicalVarianDatasets"
)
SPAREDVFS_ELEKTA = Path(
    "/home/abhishek/SpareDVFs/SPARE_GroundTruth/ClinicalElektaDatasets"
)
DATA = CE / "Experiment1" / "data"
PARAM_DEFAULT = REPO / "My v1.0" / "configs" / "elastix_bspline_masked.txt"


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


def parse_scan_id(scan_id: str) -> tuple[str, str, str]:
    """Return (patient, scan_id, vendor) with vendor in {'varian','elekta'}."""
    m = re.match(r"(CV|CE)_P(\d+)_(V|T)_(\d+)", scan_id)
    if not m:
        raise ValueError(
            f"Expected CV_P<n>_V_<k> or CE_P<n>_V_<k> (or T), got {scan_id}"
        )
    vendor = "varian" if m.group(1) == "CV" else "elekta"
    return f"P{m.group(2)}", scan_id, vendor


def eval_dir(patient: str, scan_id: str, vendor: str) -> Path:
    if vendor == "varian":
        candidates = [
            VARIAN_EVAL / patient / scan_id,
            SPAREDVFS_VARIAN / patient / scan_id,
        ]
    else:
        candidates = [
            ELEKTA_EVAL / patient / scan_id,
            SPAREDVFS_ELEKTA / patient / scan_id,
        ]
    for p in candidates:
        if p.is_dir() and (p / "GTVol_01.mha").is_file():
            return p
    raise FileNotFoundError(
        f"No Evaluation dir for {scan_id}; tried: "
        + ", ".join(str(c) for c in candidates)
    )


def padded_mask_dir(vendor: str) -> Path:
    return POP / ("varian" if vendor == "varian" else "elekta") / "PaddedLungMasks"


def save_padded_mask_native(
    mask: np.ndarray, scan_id: str, pad_vox: int, ref_mha: Path, vendor: str
):
    import itk

    out_dir = padded_mask_dir(vendor)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"Mask_Lung_padp{pad_vox:02d}_{scan_id}.mha"
    arr = mask.astype(np.uint8)
    img = itk.GetImageFromArray(arr)
    if ref_mha.exists():
        ref = itk.imread(str(ref_mha), itk.UC)
        img.CopyInformation(ref)
    itk.imwrite(img, str(out))
    return out


def preprocess_scan(
    scan_id: str,
    out_size: int,
    bbox_pad: int,
    mask_pad_vox: int,
    save_native_pad: bool,
):
    import itk

    patient, _, vendor = parse_scan_id(scan_id)
    ev = eval_dir(patient, scan_id, vendor)
    mask_path = ev / "Mask_Lung.mha"
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)

    mask_raw = itk.array_from_image(itk.imread(str(mask_path), itk.UC)) > 0
    mask = morph_pad(mask_raw, mask_pad_vox)
    print(f"[{scan_id}] vendor={vendor} mask pad={mask_pad_vox:+d}  voxels={int(mask.sum())}")

    if save_native_pad:
        p = save_padded_mask_native(mask, scan_id, mask_pad_vox, mask_path, vendor)
        print(f"[{scan_id}] saved native padded mask: {p}")

    bbox = _bbox_from_mask(mask, pad=bbox_pad)
    z0, z1, y0, y1, x0, x1 = bbox
    print(f"[{scan_id}] lung bbox (zyx): [{z0}:{z1}, {y0}:{y1}, {x0}:{x1}]")

    cts = []
    for ph in range(1, 11):
        gt = ev / f"GTVol_{ph:02d}.mha"
        if not gt.is_file():
            raise FileNotFoundError(gt)
        vol = itk.array_from_image(itk.imread(str(gt), itk.F)).astype(np.float32)
        cropped = vol[z0:z1, y0:y1, x0:x1]
        cts.append(_resample_zyx(cropped, out_size, is_mask=False))
        print(f"[{scan_id}]   CT_{ph:02d}: {vol.shape} → {cropped.shape} → {cts[-1].shape}")

    mask_r = _resample_zyx(mask[z0:z1, y0:y1, x0:x1], out_size, is_mask=True)
    print(f"[{scan_id}]   Mask_Lung → {mask_r.shape}, voxels={int(mask_r.sum())}")
    return cts, mask_r, ev, vendor


def write_volumes(out_dir: Path, cts, mask):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, ct in enumerate(cts, start=1):
        np.save(out_dir / f"CT_{i:02d}.npy", ct.astype(np.float32))
    np.save(out_dir / "Mask_Lung.npy", mask.astype(np.uint8))


def process_scan(
    scan_id: str,
    param_file: Path,
    out_size: int,
    bbox_pad: int,
    mask_pad_vox: int,
    skip_existing: bool,
    max_pairs: int | None,
    save_native_pad: bool,
):
    t0 = time.time()
    cts, mask, ev, vendor = preprocess_scan(
        scan_id, out_size, bbox_pad, mask_pad_vox, save_native_pad
    )

    all_dir = DATA / scan_id / "all"
    write_volumes(all_dir, cts, mask)

    meta = DATA / scan_id / "source.txt"
    meta.write_text(
        f"vendor={vendor}\n"
        f"evaluation_dir={ev}\n"
        f"param_file={param_file}\n"
        f"mask_pad_vox={mask_pad_vox}\n"
        f"bbox_pad={bbox_pad}\n"
        f"out_size={out_size}\n"
    )

    jobs = [(i, j) for i in range(1, 11) for j in range(1, 11)]
    if max_pairs is not None:
        jobs = jobs[:max_pairs]

    n_done = n_skip = n_id = 0
    for n, (ref, tgt) in enumerate(jobs, start=1):
        name = f"{ref:02d}_to_{tgt:02d}_pair.npy"
        out_path = all_dir / name

        if skip_existing and out_path.exists():
            n_skip += 1
            print(f"[{scan_id}] [{n}/{len(jobs)}] skip {name}", flush=True)
            continue
        if ref == tgt:
            dvf = np.zeros(cts[0].shape + (3,), dtype=np.float32)
            n_id += 1
            print(f"[{scan_id}] [{n}/{len(jobs)}] identity {name}", flush=True)
        else:
            print(f"[{scan_id}] [{n}/{len(jobs)}] Elastix {name} …", flush=True)
            tic = time.time()
            dvf = register_pair(cts[tgt - 1], cts[ref - 1], mask, param_file)
            print(
                f"[{scan_id}]     {time.time()-tic:.1f}s  "
                f"DVF mm-ish range "
                f"x[{dvf[...,0].min():.2f},{dvf[...,0].max():.2f}] "
                f"y[{dvf[...,1].min():.2f},{dvf[...,1].max():.2f}] "
                f"z[{dvf[...,2].min():.2f},{dvf[...,2].max():.2f}]",
                flush=True,
            )
            n_done += 1

        np.save(out_path, dvf.astype(np.float32))

    elapsed = time.time() - t0
    print(
        f"[{scan_id}] DONE in {elapsed/60:.1f} min  "
        f"registered={n_done} identity={n_id} skipped={n_skip}  → {all_dir}",
        flush=True,
    )
    return all_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", default="CV_P1_V_01", help="e.g. CV_P1_V_01 or CE_P1_V_01")
    ap.add_argument("--scans", default="", help="Comma-separated list (overrides --scan)")
    ap.add_argument("--out_size", type=int, default=128)
    ap.add_argument("--bbox_pad", type=int, default=8)
    ap.add_argument("--mask_pad", type=int, default=8, help="3D EDT dilate on Mask_Lung before crop")
    ap.add_argument("--param_file", type=Path, default=PARAM_DEFAULT)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--max_pairs", type=int, default=None)
    ap.add_argument("--no_save_native_pad", action="store_true")
    args = ap.parse_args()

    try:
        import itk  # noqa: F401
    except ImportError:
        print(
            "itk-elastix missing. Use LEARN-GUI venv python.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.param_file.exists():
        raise SystemExit(f"param file not found: {args.param_file}")

    scans = [s.strip() for s in args.scans.split(",") if s.strip()] if args.scans else [args.scan]
    print(f"Clinical DVF scans={scans}  param={args.param_file}")
    print(f"Output: {DATA}")

    for scan_id in scans:
        process_scan(
            scan_id,
            args.param_file,
            args.out_size,
            args.bbox_pad,
            args.mask_pad,
            args.skip_existing,
            args.max_pairs,
            save_native_pad=not args.no_save_native_pad,
        )

    print("All requested scans finished.")


if __name__ == "__main__":
    main()
