#!/usr/bin/env python3
"""A1 prepare: stage DIR-Lab → R3 reorient → downsample → DRR → Elastix → prep_train.

R3 (locked): SI on itk-Y, AP+SI flip, volume centred on isocenter; DRR with
Geometry_SPARE.xml (OffsetY −2) + MC/Varian half-fan opts. Native DIR packs stay
untouched under PopulationStudy/; reorient is applied to train CT/masks only.

  cd "DIR EXPERIMENTS"
  LEARN-GUI/.venv/bin/python scripts/prepare_a1_case.py --case 1 --gpu 1
  # Elastix-only smoke (no DRR / prep_train):
  ... prepare_a1_case.py --case 1 --gpu 1 --skip-drr --skip-prep
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
VOXEL_GAN = DIR_EXP.parent
VMC = Path(os.environ.get("VOXELMAP_CLINICAL_ROOT", "/home/abhishek/Documents/VoxelMap_Clinical"))
LEARN = Path(os.environ.get("LEARN_GUI_ROOT", "/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python"))
DIR_NATIVE = VOXEL_GAN / "PopulationStudy" / "DIR-Experiments" / "data"

sys.path.insert(0, str(LEARN))
sys.path.insert(0, str(VMC / "config"))
sys.path.insert(0, str(DIR_EXP / "scripts"))


def scan_id_for_case(case: int) -> str:
    return f"DIR_C{case:02d}"


def _load_spare_d2m():
    spec = importlib.util.spec_from_file_location(
        "spare_d2m", LEARN / "modules/dicom2mha/implementations/spare.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run


def stage_case(case: int, staged: Path, geom_xml: Path) -> None:
    src = DIR_NATIVE / f"P{case}_DIR"
    if not src.is_dir():
        raise SystemExit(f"Missing native DIR pack: {src}")

    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)

    for gt in sorted(src.glob("GTVol_*.mha")):
        (staged / gt.name).symlink_to(gt.resolve())

    lung = src / "Mask_Lung.mha"
    if not lung.is_file():
        raise SystemExit(f"Missing {lung}")
    (staged / "Mask_Lung.mha").symlink_to(lung.resolve())

    # Minimal body mask for prep_train (full FOV)
    import SimpleITK as sitk

    lung_img = sitk.ReadImage(str(lung))
    body = sitk.Image(lung_img.GetSize(), sitk.sitkUInt8)
    body.CopyInformation(lung_img)
    body = sitk.Add(body, 1)  # all ones
    sitk.WriteImage(body, str(staged / "Mask_Body.mha"))

    proj = staged / "Proj"
    proj.mkdir()
    shutil.copy2(geom_xml, proj / "Geometry.xml")

    # One phase bin per projection, cycling 1..10 (matches GTVol_01..10)
    n_proj = len(ET.parse(geom_xml).findall("Projection"))
    bins = [str((i % 10) + 1) for i in range(n_proj)]
    (proj / "RespBin.csv").write_text("\n".join(bins) + "\n")
    logging.info("Staged %s (%d GTVol, %d projs) → %s", src.name, len(list(staged.glob("GTVol_*.mha"))), n_proj, staged)


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


def apply_r3_to_train(train: Path) -> None:
    """Rewrite CT_*.mha and Mask_*.mha in-place into the SPARE orbit frame (R3)."""
    import SimpleITK as sitk
    from reorient_dirlab import reorient_for_spare_orbit

    for ct in sorted(train.glob("CT_*.mha")):
        # d2m may have left a symlink to native GTVol — replace with real R3 file
        if ct.is_symlink() or ct.exists():
            native = sitk.ReadImage(str(ct.resolve() if ct.is_symlink() else ct), sitk.sitkFloat32)
            out = reorient_for_spare_orbit(native, clip_hu=True)
            if ct.is_symlink() or ct.exists():
                ct.unlink()
            sitk.WriteImage(out, str(ct), True)
            logging.info(
                "R3 %s size=%s origin=%s",
                ct.name,
                out.GetSize(),
                tuple(round(v, 2) for v in out.GetOrigin()),
            )

    for mask_path in sorted(train.glob("Mask_*.mha")):
        mask = sitk.ReadImage(str(mask_path))
        # binary / label: same axis recipe, no HU clip
        out = reorient_for_spare_orbit(sitk.Cast(mask, sitk.sitkFloat32), clip_hu=False)
        arr = (sitk.GetArrayFromImage(out) > 0.5).astype("uint8")
        img = sitk.GetImageFromArray(arr)
        img.CopyInformation(out)
        sitk.WriteImage(img, str(mask_path), True)
        logging.info("R3 %s size=%s", mask_path.name, img.GetSize())


def downsample_volume_keep_hu(input_path: Path, output_path: Path, target_size=(128, 128, 128)) -> None:
    """Like LEARN downsample, but do NOT clip HU < 0 → 0 (needed for DIR-Lab)."""
    import itk
    import numpy as np

    image = itk.imread(str(input_path))
    in_size = itk.size(image)
    in_spacing = image.GetSpacing()
    out_spacing = [(in_size[i] * in_spacing[i]) / target_size[i] for i in range(3)]

    interpolator = itk.LinearInterpolateImageFunction.New(image)
    resampler = itk.ResampleImageFilter.New(image)
    resampler.SetInterpolator(interpolator)
    resampler.SetSize(list(target_size))
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputSpacing(out_spacing)
    resampler.SetOutputDirection(image.GetDirection())
    # Preserve air HU (LEARN default pads/clips negatives away)
    try:
        resampler.SetDefaultPixelValue(float(np.min(itk.GetArrayFromImage(image))))
    except Exception:
        resampler.SetDefaultPixelValue(-1024.0)
    resampler.Update()

    arr = itk.GetArrayFromImage(resampler.GetOutput()).astype(np.float32)
    # NOTE: intentionally no arr[arr < 0] = 0
    final_image = itk.GetImageFromArray(arr)
    final_image.SetSpacing([1.0, 1.0, 1.0])
    final_image.SetOrigin([0.0, 0.0, 0.0])
    final_image.SetDirection(itk.matrix_from_array(np.eye(3)))
    itk.imwrite(final_image, str(output_path))
    logging.info("Saved %s (HU kept, min=%.1f max=%.1f)", output_path.name, float(arr.min()), float(arr.max()))


def run_downsample(train: Path) -> int:
    for f in sorted(train.glob("CT_*.mha")):
        out = train / f"sub_CT{f.name[-7:]}"  # sub_CT_XX.mha
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
    """Nearest-neighbor resample Mask_Lung → unit-spaced 128³ (matches sub_CT)."""
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
    """ITK-Elastix with lung masks (LEARN generate_dvf has no mask support)."""
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
    """DIR A1 default: masked B-spline (grid 16). LowRes unmasked underfits large-SI cases."""
    import itk

    param = DIR_EXP / "arms" / "A1_oracle_dirlab" / "Elastix_BSpline_DIR_masked.txt"
    if not param.is_file():
        raise FileNotFoundError(param)
    fixed_path = train / "sub_CT_06.mha"  # T50 in DIR pack convention
    if not fixed_path.is_file():
        raise FileNotFoundError(fixed_path)
    mask_path = train / "Mask_Lung.mha"
    if not mask_path.is_file():
        raise FileNotFoundError(f"Need lung mask for DIR Elastix: {mask_path}")

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
    workers = 1  # ITK elastix objects are not safe to share across threads
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(_register_masked, fixed, m, mask, param, o): m.name
            for m, o in jobs
        }
        for fut in as_completed(futs):
            fut.result()
            done += 1
            logging.info("DVF %d/%d %s (masked DIR param)", done, len(jobs), futs[fut])
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
    ap.add_argument("--case", type=int, default=1, choices=range(1, 11))
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--with-test", action="store_true", default=True)
    ap.add_argument("--skip-drr", action="store_true")
    ap.add_argument("--skip-dvf", action="store_true")
    ap.add_argument(
        "--skip-prep",
        action="store_true",
        help="Stop after Elastix (no prep_train). Use with --skip-drr for Elastix TRE smoke.",
    )
    ap.add_argument("--clean", action="store_true", default=True)
    ap.add_argument(
        "--redo-hu",
        action="store_true",
        help="Keep existing DRRs; re-downsample (HU kept), re-Elastix DVF, re-prep_train only",
    )
    ap.add_argument(
        "--no-r3",
        action="store_true",
        help="Skip R3 reorient (legacy native frame; not for corrected A1).",
    )
    args = ap.parse_args()

    # DIR EXPERIMENTS hard rule: physical GPU 1 only (never GPU 0).
    if int(args.gpu) != 1:
        raise SystemExit(f"Refusing --gpu {args.gpu}: use --gpu 1 only")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    scan_id = scan_id_for_case(args.case)
    geom_xml = DIR_EXP / "arms" / "A1_oracle_dirlab" / "Geometry.xml"
    if not geom_xml.is_file() or geom_xml.stat().st_size < 1000:
        raise SystemExit(f"Missing/empty Geometry.xml: {geom_xml}")

    run_root = DIR_EXP / "arms" / "A1_oracle_dirlab" / "runs" / scan_id
    staged = DIR_EXP / "arms" / "A1_oracle_dirlab" / "data" / "staged" / scan_id
    logs = run_root / "logs"

    logs.mkdir(parents=True, exist_ok=True)
    log_name = "prepare_a1_redo_hu.log" if args.redo_hu else "prepare_a1.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logs / log_name, mode="w"),
        ],
        force=True,
    )

    if args.redo_hu:
        train = run_root / scan_id / "train"
        if not train.is_dir() or not list(train.glob("CT_*.mha")):
            raise SystemExit(f"--redo-hu needs existing CT_*.mha under {train}")
        for name in ("checkpoints_nofilm", "plots_nofilm", "tre_75", "ModelTraining"):
            src = run_root / name
            if src.exists():
                dst = run_root / f"{name}_hu_clip_bak"
                if dst.exists():
                    shutil.rmtree(dst)
                src.rename(dst)
                logging.info("Archived %s → %s", src.name, dst.name)
        for p in list(train.glob("sub_CT_*.mha")) + list(train.glob("DVF_sub_*.mha")):
            p.unlink()
        args.clean = False
        args.skip_drr = True
    elif args.clean and run_root.exists():
        shutil.rmtree(run_root)
        logs.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    logging.info(
        "A1 prepare case=%d scan_id=%s gpu=%s redo_hu=%s r3=%s keep_hu_downsample=True",
        args.case,
        scan_id,
        args.gpu,
        args.redo_hu,
        not args.no_r3,
    )

    if not args.redo_hu:
        stage_case(args.case, staged, geom_xml)
        train = ensure_layout(run_root, staged, scan_id, wipe_train=True)
        _load_spare_d2m()(staged, train)
        if not args.no_r3:
            logging.info("=== R3 reorient (SI→Y, AP+SI flip, isocentre) ===")
            apply_r3_to_train(train)
    else:
        if not staged.is_dir():
            stage_case(args.case, staged, geom_xml)
        train = ensure_layout(run_root, staged, scan_id, wipe_train=False)
        if not list(train.glob("CT_*.mha")):
            _load_spare_d2m()(staged, train)
            if not args.no_r3:
                apply_r3_to_train(train)

    train_geom = train / "Proj" / "Geometry.xml"
    logging.info("=== DOWNSAMPLE (HU preserved) ===")
    logging.info("sub_CT count: %d", run_downsample(train))

    if not args.skip_drr:
        logging.info("=== DRR (MC/Varian half-fan, Geometry_SPARE) ===")
        run_drr(train, train_geom)
        logging.info("=== COMPRESS ===")
        logging.info("bins: %d", run_compress(train))
    else:
        n_bin = len(list(train.glob("*_Proj_*.bin")))
        logging.info("=== SKIP DRR/COMPRESS (existing bins=%d) ===", n_bin)
        if n_bin < 100 and not args.skip_prep:
            raise SystemExit("skip-drr but almost no projection bins present (use --skip-prep for Elastix-only)")

    if not args.skip_dvf:
        logging.info("=== DVF (fixed=sub_CT_06=T50, HU-preserving sub_CT) ===")
        run_dvf(train)

    if args.skip_prep:
        logging.info(
            "=== SKIP PREP_TRAIN === Done Elastix-only in %.1f min → %s",
            (time.perf_counter() - t0) / 60,
            train,
        )
        return 0 if (train / "DVF_sub_01.mha").is_file() or args.skip_dvf else 1

    logging.info("=== PREP_TRAIN ===")
    run_prep(run_root, scan_id, train_geom, with_test=args.with_test)

    mt = run_root / "ModelTraining" / "train" / scan_id
    logging.info("Done in %.1f min → %s", (time.perf_counter() - t0) / 60, mt)
    return 0 if mt.is_dir() else 1


if __name__ == "__main__":
    raise SystemExit(main())
