#!/usr/bin/env python3
"""Elastix DVF library on DIR packed_iso volumes (2 mm, 160³).

DVFs stored as iso-grid voxels (1 vx = 2 mm), matching PopulationStudy/data_iso.

  cd PopulationStudy/DIR-Experiments/Experiment2
  /path/to/LEARN-GUI/.venv/bin/python scripts/prepare_dir_dvf_library_iso.py --skip_existing
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

E2 = Path(__file__).resolve().parents[1]
PACKED_ISO = E2 / "packed_iso"
SPACING_MM = 2.0
CE_SCRIPTS = E2.parent.parent / "ClinicalExperiments" / "scripts"
REPO = E2.parent.parent.parent
PARAM_DEFAULT = REPO / "My v1.0" / "configs" / "elastix_bspline_masked.txt"

sys.path.insert(0, str(CE_SCRIPTS))
from prepare_clinical_dvf_library import register_pair  # noqa: E402


def _to_itk_float_spaced(vol_zyx: np.ndarray, spacing_mm: float):
    import itk

    img = itk.GetImageFromArray(vol_zyx.astype(np.float32))
    img.SetSpacing((spacing_mm, spacing_mm, spacing_mm))
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection(itk.matrix_from_array(np.eye(3)))
    return img


def register_pair_iso(fixed_zyx, moving_zyx, mask_zyx, param_file: Path):
    """Elastix on iso grid; return DVF in iso-voxels (mm / spacing_mm)."""
    import itk

    fixed = _to_itk_float_spaced(fixed_zyx, SPACING_MM)
    moving = _to_itk_float_spaced(moving_zyx, SPACING_MM)
    mask = itk.GetImageFromArray(mask_zyx.astype(np.uint8))
    mask.SetSpacing((SPACING_MM, SPACING_MM, SPACING_MM))
    mask.SetOrigin((0.0, 0.0, 0.0))

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
    dvf = itk.array_from_image(dvf_img).astype(np.float32)
    # ITK deformation field is physical (mm); store iso-voxels
    return (dvf / SPACING_MM).astype(np.float32)


def process_patient(pid: str, packed_root: Path, param_file: Path, skip_existing: bool, max_pairs: int | None):
    all_dir = packed_root / pid / "all"
    if not all_dir.is_dir():
        raise FileNotFoundError(f"{all_dir} — run pack_dirlab_iso.py first")

    cts = [np.load(all_dir / f"CT_{i:02d}.npy").astype(np.float32) for i in range(1, 11)]
    mask = (np.load(all_dir / "Mask_Lung.npy") > 0).astype(np.uint8)
    print(f"[{pid}] iso CT {cts[0].shape} mask_vox={int(mask.sum())}", flush=True)

    jobs = [(i, j) for i in range(1, 11) for j in range(1, 11)]
    if max_pairs is not None:
        jobs = jobs[:max_pairs]

    n_done = n_skip = n_id = 0
    t0 = time.time()
    for n, (ref, tgt) in enumerate(jobs, start=1):
        name = f"{ref:02d}_to_{tgt:02d}_pair.npy"
        out_path = all_dir / name
        if skip_existing and out_path.is_file():
            n_skip += 1
            continue
        if ref == tgt:
            dvf = np.zeros(cts[0].shape + (3,), dtype=np.float32)
            n_id += 1
        else:
            print(f"[{pid}] [{n}/{len(jobs)}] Elastix iso {name} …", flush=True)
            tic = time.time()
            dvf = register_pair_iso(cts[tgt - 1], cts[ref - 1], mask, param_file)
            print(f"[{pid}]     {time.time()-tic:.0f}s", flush=True)
            n_done += 1
        np.save(out_path, dvf.astype(np.float32))

    print(f"[{pid}] DONE {time.time()-t0:.0f}s reg={n_done} id={n_id} skip={n_skip}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patient", default="P1_DIR")
    ap.add_argument("--patients", default="")
    ap.add_argument("--packed-root", type=Path, default=PACKED_ISO)
    ap.add_argument("--param-file", type=Path, default=PARAM_DEFAULT)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--max_pairs", type=int, default=None)
    args = ap.parse_args()

    try:
        import itk  # noqa: F401
    except ImportError:
        print("itk-elastix missing. Use LEARN-GUI venv python.", file=sys.stderr)
        sys.exit(1)

    patients = (
        [p.strip() for p in args.patients.split(",") if p.strip()]
        if args.patients
        else [args.patient]
    )
    for pid in patients:
        process_patient(pid, args.packed_root, args.param_file, args.skip_existing, args.max_pairs)
    print("All requested patients finished.", flush=True)


if __name__ == "__main__":
    main()
