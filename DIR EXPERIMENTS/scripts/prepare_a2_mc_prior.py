#!/usr/bin/env python3
"""A2: MC Val Prior P1–P9 → downsample → DRR → masked Elastix → prep_train.

Volumes: SpareDVFs/MonteCarloDatasets/Validation/P{N}/MC_V_P{N}_Prior/CT_*.mha
Geometry: that patient's NS_01 (or SC/LD fallback) Proj/Geometry.xml (OffsetY −2)
Lung mask: SPARE_GroundTruth Mask_Lung resampled onto Prior physical grid

  cd "DIR EXPERIMENTS"
  LEARN-GUI/.venv/bin/python scripts/prepare_a2_mc_prior.py --patient 1 --gpu 0
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DIR_EXP = Path(__file__).resolve().parents[1]
VMC = Path(os.environ.get("VOXELMAP_CLINICAL_ROOT", "/home/abhishek/Documents/VoxelMap_Clinical"))
LEARN = Path(os.environ.get("LEARN_GUI_ROOT", "/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python"))
SPARE = Path(os.environ.get("SPARE_DVFS_ROOT", "/home/abhishek/SpareDVFs"))
MC_VAL = SPARE / "MonteCarloDatasets" / "Validation"
MC_GT = SPARE / "SPARE_GroundTruth" / "MonteCarloDatasets" / "Validation"
A2 = DIR_EXP / "arms" / "A2_generic_spare"

sys.path.insert(0, str(LEARN))
sys.path.insert(0, str(VMC / "config"))

# Prefer NS, then SC/LD (P9 has SC only)
_SCAN_CANDIDATES = ("NS_01", "NS_02", "SC_01", "SC_02", "LD_01", "LD_02")


def scan_id_for_patient(patient: int) -> str:
    return f"MC_V_P{patient}_Prior"


def _protocol_dir(patient: int, kind: str) -> Path | None:
    """kind='raw' → MonteCarloDatasets; kind='gt' → SPARE_GroundTruth."""
    root = MC_VAL if kind == "raw" else MC_GT
    pdir = root / f"P{patient}"
    if not pdir.is_dir():
        return None
    for suf in _SCAN_CANDIDATES:
        d = pdir / f"MC_V_P{patient}_{suf}"
        if kind == "raw" and (d / "Proj" / "Geometry.xml").is_file():
            return d
        if kind == "gt" and (d / "Mask_Lung.mha").is_file():
            return d
    return None


def _load_spare_d2m():
    spec = importlib.util.spec_from_file_location(
        "spare_d2m", LEARN / "modules/dicom2mha/implementations/spare.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run


def _resample_lung_mask_to_prior(mask_src: Path, prior_ref: Path, out_path: Path) -> None:
    """Nearest-neighbor resample SPARE GT Mask_Lung onto Prior CT physical space."""
    import SimpleITK as sitk

    ct = sitk.ReadImage(str(prior_ref))
    mask = sitk.ReadImage(str(mask_src))
    rs = sitk.Resample(
        mask,
        ct,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    arr = sitk.GetArrayFromImage(rs)
    if int(arr.sum()) < 1000:
        raise RuntimeError(
            f"Resampled lung mask nearly empty ({arr.sum()} voxels): {mask_src} → {prior_ref}"
        )
    sitk.WriteImage(rs, str(out_path))
    logging.info(
        "Lung mask %s → Prior grid (frac=%.4f)",
        mask_src.parent.name,
        float(arr.mean()),
    )


def stage_patient(patient: int, staged: Path, geom_xml: Path, prior_dir: Path, mask_src: Path) -> None:
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)

    cts = sorted(prior_dir.glob("CT_*.mha"))
    if len(cts) != 10:
        raise SystemExit(f"Expected 10 Prior CTs in {prior_dir}, found {len(cts)}")

    # LEARN spare_d2m expects GTVol_*.mha
    for ct in cts:
        m = re.search(r"(\d+)", ct.stem)
        num = m.group(1).zfill(2) if m else ct.stem
        (staged / f"GTVol_{num}.mha").symlink_to(ct.resolve())

    ref = prior_dir / "CT_06.mha"
    _resample_lung_mask_to_prior(mask_src, ref, staged / "Mask_Lung.mha")

    import SimpleITK as sitk

    lung = sitk.ReadImage(str(staged / "Mask_Lung.mha"))
    body = sitk.Image(lung.GetSize(), sitk.sitkUInt8)
    body.CopyInformation(lung)
    body = sitk.Add(body, 1)
    sitk.WriteImage(body, str(staged / "Mask_Body.mha"))

    proj = staged / "Proj"
    proj.mkdir()
    shutil.copy2(geom_xml, proj / "Geometry.xml")
    n_proj = len(ET.parse(geom_xml).findall("Projection"))
    bins = [str((i % 10) + 1) for i in range(n_proj)]
    (proj / "RespBin.csv").write_text("\n".join(bins) + "\n")
    logging.info(
        "Staged Prior P%d (%d GTVol, %d projs, geom=%s) → %s",
        patient,
        len(cts),
        n_proj,
        geom_xml.parent.parent.name,
        staged,
    )


def ensure_layout(run_root: Path, staged: Path, scan_id: str, *, wipe_train: bool = True) -> Path:
    train = run_root / scan_id / "train"
    if wipe_train and train.exists():
        shutil.rmtree(train)
    train.mkdir(parents=True, exist_ok=True)
    for src in staged.glob("Mask_*.mha"):
        dst = train / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
    if not (train / "Proj" / "Geometry.xml").is_file():
        if (train / "Proj").exists():
            shutil.rmtree(train / "Proj")
        shutil.copytree(staged / "Proj", train / "Proj")
    for gt in staged.glob("GTVol_*.mha"):
        dst = train / gt.name
        if not dst.exists():
            dst.symlink_to(gt.resolve())
    return train


def downsample_volume_keep_hu(input_path: Path, output_path: Path, target_size=(128, 128, 128)) -> None:
    import itk
    import numpy as np

    # Cast to float — SPARE Normalize writes short; SetDefaultPixelValue(float) then fails.
    image = itk.imread(str(input_path), itk.F)
    in_size = itk.size(image)
    in_spacing = image.GetSpacing()
    out_spacing = [(in_size[i] * in_spacing[i]) / target_size[i] for i in range(3)]
    pad = float(np.min(itk.GetArrayFromImage(image)))

    interpolator = itk.LinearInterpolateImageFunction.New(image)
    resampler = itk.ResampleImageFilter.New(image)
    resampler.SetInterpolator(interpolator)
    resampler.SetSize(list(target_size))
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputSpacing(out_spacing)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetDefaultPixelValue(pad)
    resampler.Update()

    arr = itk.GetArrayFromImage(resampler.GetOutput()).astype(np.float32)
    final_image = itk.GetImageFromArray(arr)
    final_image.SetSpacing([1.0, 1.0, 1.0])
    final_image.SetOrigin([0.0, 0.0, 0.0])
    final_image.SetDirection(itk.matrix_from_array(np.eye(3)))
    itk.imwrite(final_image, str(output_path))
    logging.info(
        "Saved %s (HU kept, min=%.1f max=%.1f)",
        output_path.name,
        float(arr.min()),
        float(arr.max()),
    )


def run_downsample(train: Path) -> int:
    for f in sorted(train.glob("CT_*.mha")):
        out = train / f"sub_CT{f.name[-7:]}"
        downsample_volume_keep_hu(f, out)
    return len(list(train.glob("sub_CT_*.mha")))


def run_drr(train: Path, geom_xml: Path) -> int:
    from modules.drr_generation.run import run as run_drr_fn
    from elekta_drr import mc_varian_drr_opts_for_scan

    opts = mc_varian_drr_opts_for_scan(train)
    opts["geometry_path"] = str(geom_xml)
    mhas = sorted(train.glob("CT_*.mha"))
    for i, mha in enumerate(mhas, start=1):
        m = re.search(r"(\d+)", mha.stem)
        ct_num = int(m.group(1)) if m else i
        logging.info("DRR %s (%d/%d)", mha.name, i, len(mhas))
        ok, err = run_drr_fn(
            str(mha),
            str(train),
            geometry_file=str(geom_xml),
            ct_num=ct_num,
            dataset_type="spare",
            **opts,
        )
        if not ok:
            raise RuntimeError(f"DRR failed {mha.name}: {err}")
    return len(mhas)


def run_compress(train: Path) -> int:
    from modules.drr_compression.compress import process_directory

    process_directory(train)
    return len(list(train.glob("*_Proj_*.bin")))


def _downsample_mask_nn(mask_path: Path, target_size=(128, 128, 128)):
    import itk
    import numpy as np

    mask_native = itk.imread(str(mask_path))
    in_size = itk.size(mask_native)
    in_spacing = mask_native.GetSpacing()
    out_spacing = [(in_size[i] * in_spacing[i]) / target_size[i] for i in range(3)]
    nn = itk.NearestNeighborInterpolateImageFunction.New(mask_native)
    rs = itk.ResampleImageFilter.New(mask_native)
    rs.SetInterpolator(nn)
    rs.SetSize(list(target_size))
    rs.SetOutputOrigin(mask_native.GetOrigin())
    rs.SetOutputSpacing(out_spacing)
    rs.SetOutputDirection(mask_native.GetDirection())
    rs.SetDefaultPixelValue(0)
    rs.Update()
    arr = (itk.GetArrayFromImage(rs.GetOutput()) > 0).astype(np.uint8)
    img = itk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection(itk.matrix_from_array(np.eye(3)))
    return img


def _register_masked(fixed_img, moving_path: Path, mask_img, param_path: Path, out_path: Path) -> None:
    import itk
    import numpy as np

    moving = itk.imread(str(moving_path), itk.F)
    param_obj = itk.ParameterObject.New()
    param_obj.AddParameterFile(str(param_path))
    ImageType = type(fixed_img)
    elastix = itk.ElastixRegistrationMethod[ImageType, ImageType].New()
    elastix.SetFixedImage(fixed_img)
    elastix.SetMovingImage(moving)
    elastix.SetFixedMask(mask_img)
    elastix.SetMovingMask(mask_img)
    elastix.SetParameterObject(param_obj)
    elastix.LogToConsoleOff()
    elastix.Update()
    tp = elastix.GetTransformParameterObject()
    tx = itk.TransformixFilter[ImageType].New()
    tx.SetMovingImage(moving)
    tx.SetTransformParameterObject(tp)
    tx.SetComputeDeformationField(True)
    tx.Update()
    arr = itk.array_from_image(tx.GetOutputDeformationField()).astype(np.float32)
    out = itk.GetImageFromArray(arr, is_vector=True)
    out.SetSpacing((1.0, 1.0, 1.0))
    out.SetOrigin((0.0, 0.0, 0.0))
    out.SetDirection(itk.matrix_from_array(np.eye(3)))
    itk.imwrite(out, str(out_path))


def run_dvf(train: Path) -> int:
    import itk

    param = DIR_EXP / "arms" / "A1_oracle_dirlab" / "Elastix_BSpline_DIR_masked.txt"
    if not param.is_file():
        raise FileNotFoundError(param)
    fixed_path = train / "sub_CT_06.mha"
    if not fixed_path.is_file():
        raise FileNotFoundError(fixed_path)
    mask_path = train / "Mask_Lung.mha"
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)

    fixed = itk.imread(str(fixed_path), itk.F)
    mask = _downsample_mask_nn(mask_path)
    jobs = []
    for m in sorted(train.glob("sub_CT_*.mha")):
        num_m = re.search(r"(\d+)", m.stem)
        num = num_m.group(1) if num_m else ""
        if num == "06":
            continue
        jobs.append((m, train / f"DVF_sub_{num.zfill(2)}.mha"))

    os.environ.setdefault("ITK_NUMBER_OF_THREADS", "1")
    done = 0
    with ThreadPoolExecutor(max_workers=1) as pool:
        futs = {
            pool.submit(_register_masked, fixed, m, mask, param, o): m.name
            for m, o in jobs
        }
        for fut in as_completed(futs):
            fut.result()
            done += 1
            logging.info("DVF %d/%d %s (masked)", done, len(jobs), futs[fut])
    return done


def run_prep(run_root: Path, scan_id: str, geom_xml: Path, *, with_test: bool) -> None:
    from modules.prep_train.run import run_prep_train

    split = "Train+Test" if with_test else "train"
    run_prep_train(
        run_root,
        {scan_id: split},
        dataset_type="spare",
        on_log=logging.info,
        angles_xml_path=geom_xml,
        prefer_patient_xml=False,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patient", type=int, required=True, choices=range(1, 10))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--with-test", action="store_true", default=True)
    ap.add_argument("--skip-drr", action="store_true")
    ap.add_argument("--skip-dvf", action="store_true")
    ap.add_argument("--clean", action="store_true", default=True)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    patient = args.patient
    scan_id = scan_id_for_patient(patient)

    prior_dir = MC_VAL / f"P{patient}" / f"MC_V_P{patient}_Prior"
    if not prior_dir.is_dir():
        raise SystemExit(f"Missing Prior: {prior_dir}")

    raw_proto = _protocol_dir(patient, "raw")
    gt_proto = _protocol_dir(patient, "gt")
    if raw_proto is None:
        raise SystemExit(f"No Geometry.xml under MC Val P{patient}")
    if gt_proto is None:
        raise SystemExit(f"No Mask_Lung under SPARE_GroundTruth Val P{patient}")

    geom_xml = raw_proto / "Proj" / "Geometry.xml"
    mask_src = gt_proto / "Mask_Lung.mha"

    run_root = A2 / "runs" / scan_id
    staged = A2 / "data" / "staged" / scan_id
    logs = run_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logs / "prepare_a2.log", mode="w"),
        ],
        force=True,
    )

    if args.clean and run_root.exists():
        shutil.rmtree(run_root)
        logs.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    logging.info(
        "A2 prepare patient=%d scan_id=%s gpu=%s geom=%s mask=%s",
        patient,
        scan_id,
        args.gpu,
        raw_proto.name,
        gt_proto.name,
    )

    stage_patient(patient, staged, geom_xml, prior_dir, mask_src)
    train = ensure_layout(run_root, staged, scan_id, wipe_train=True)
    _load_spare_d2m()(staged, train)

    train_geom = train / "Proj" / "Geometry.xml"
    logging.info("=== DOWNSAMPLE (HU preserved) ===")
    logging.info("sub_CT count: %d", run_downsample(train))

    if not args.skip_drr:
        logging.info("=== DRR (MC/Varian half-fan, patient geom) ===")
        run_drr(train, train_geom)
        logging.info("=== COMPRESS ===")
        logging.info("bins: %d", run_compress(train))
    else:
        n_bin = len(list(train.glob("*_Proj_*.bin")))
        logging.info("=== SKIP DRR/COMPRESS (bins=%d) ===", n_bin)

    if not args.skip_dvf:
        logging.info("=== DVF (fixed=sub_CT_06, masked) ===")
        run_dvf(train)
    logging.info("=== PREP_TRAIN ===")
    run_prep(run_root, scan_id, train_geom, with_test=args.with_test)

    mt = run_root / "ModelTraining" / "train" / scan_id
    logging.info("Done in %.1f min → %s", (time.perf_counter() - t0) / 60, mt)
    return 0 if mt.is_dir() else 1


if __name__ == "__main__":
    raise SystemExit(main())
