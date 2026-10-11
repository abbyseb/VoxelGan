#!/usr/bin/env python3
"""A2-TCIA: one TCIA 4D-Lung scan -> R3 CT mha -> downsample -> DRR -> masked Elastix -> prep_train.

Same steps and settings as prepare_a1_case.py (the DIR-Lab test data), so training DRRs are made exactly like the
DIR-Lab test DRRs; only the input differs:
  CTs   : packed_r3_hu/S<n>/all/CT_XX.npy (HU, 2 mm 160^3, already in the R3 SPARE-orbit frame, centred on the lung)
          -> written as mha with spacing 2 mm, identity direction, grid centre on (0,0,0), HU clipped like A1
  Lung  : Mask_Lung_unpadded.npy (same grid)
  Geom  : arms/A1_oracle_dirlab/Geometry.xml (the DIR-Lab test geometry, 680 projections), thinned to every
          --angle-step-th projection to keep the pooled set near A2's size (A2: 55,080 pairs for 9 patients).

  cd "DIR EXPERIMENTS"
  LEARN-GUI/.venv/bin/python scripts/prepare_a2_tcia.py --scan 1 --gpu 0
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

DIR_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIR_EXP / "scripts"))
import prepare_a1_case as A1P  # noqa: E402  (run_downsample, run_drr, run_compress, run_dvf, run_prep, ensure_layout)
from reorient_dirlab import HU_HI, HU_LO  # noqa: E402

TCIA = Path("/media/abhishek/3CCA3CADCA3C6574/TCIA_4D-Lung/packed_r3_hu")
A1_GEOM = DIR_EXP / "arms" / "A1_oracle_dirlab" / "Geometry.xml"
ARM = DIR_EXP / "arms" / "A2_tcia"
SPACING = 2.0


def thin_geometry(src: Path, dst: Path, step: int) -> int:
    tree = ET.parse(src)
    root = tree.getroot()
    projs = root.findall("Projection")
    for i, p in enumerate(projs):
        if i % step:
            root.remove(p)
    tree.write(dst)
    return len(root.findall("Projection"))


def write_mha(arr_zyx: np.ndarray, path: Path) -> None:
    import SimpleITK as sitk

    img = sitk.GetImageFromArray(np.ascontiguousarray(arr_zyx))
    img.SetSpacing((SPACING, SPACING, SPACING))
    img.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    img.SetOrigin(tuple(-(n - 1) * SPACING / 2.0 for n in img.GetSize()))  # grid centre on (0,0,0), as A1 R3
    sitk.WriteImage(img, str(path), True)


def stage_scan(n: int, staged: Path, geom_xml: Path) -> None:
    src = TCIA / f"S{n}" / "all"
    cts = [src / f"CT_{k:02d}.npy" for k in range(1, 11)]
    missing = [c.name for c in cts if not c.is_file()]
    if missing:
        raise SystemExit(f"S{n}: missing {missing}")
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)
    for k, c in enumerate(cts, start=1):
        write_mha(np.clip(np.load(c).astype(np.float32).squeeze(), HU_LO, HU_HI), staged / f"GTVol_{k:02d}.mha")
    lung = (np.load(src / "Mask_Lung_unpadded.npy").squeeze() > 0).astype(np.uint8)
    write_mha(lung, staged / "Mask_Lung.mha")
    write_mha(np.ones_like(lung), staged / "Mask_Body.mha")
    proj = staged / "Proj"
    proj.mkdir()
    shutil.copy2(geom_xml, proj / "Geometry.xml")
    n_proj = len(ET.parse(geom_xml).findall("Projection"))
    (proj / "RespBin.csv").write_text("\n".join(str((i % 10) + 1) for i in range(n_proj)) + "\n")
    logging.info("Staged TCIA S%d (10 GTVol, lung %d vox, %d projs) -> %s", n, int(lung.sum()), n_proj, staged)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scan", type=int, required=True, help="TCIA scan number (S1..S82)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--angle-step", type=int, default=9, help="keep every Nth of the 680 A1 projections")
    args = ap.parse_args()
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    scan_id = f"TCIA_S{args.scan:02d}"
    run_root = ARM / "runs" / scan_id
    staged = ARM / "data" / "staged" / scan_id
    if run_root.exists():
        shutil.rmtree(run_root)
    (run_root / "logs").mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(), logging.FileHandler(run_root / "logs" / "prepare_a2_tcia.log", mode="w")],
                        force=True)
    t0 = time.perf_counter()
    geom = ARM / "data" / f"Geometry_A1_step{args.angle_step}.xml"
    geom.parent.mkdir(parents=True, exist_ok=True)
    if not geom.is_file():
        logging.info("thinned A1 geometry: %d projections", thin_geometry(A1_GEOM, geom, args.angle_step))
    stage_scan(args.scan, staged, geom)
    train = A1P.ensure_layout(run_root, staged, scan_id, wipe_train=True)
    A1P._load_spare_d2m()(staged, train)  # GTVol -> CT_XX.mha (no R3: TCIA is already in the R3 frame)
    train_geom = train / "Proj" / "Geometry.xml"
    logging.info("=== DOWNSAMPLE (HU preserved) === sub_CT %d", A1P.run_downsample(train))
    logging.info("=== DRR ===")
    A1P.run_drr(train, train_geom)
    logging.info("=== COMPRESS === bins %d", A1P.run_compress(train))
    logging.info("=== DVF (fixed=sub_CT_06, masked) ===")
    A1P.run_dvf(train)
    logging.info("=== PREP_TRAIN ===")
    A1P.run_prep(run_root, scan_id, train_geom, with_test=False)
    mt = run_root / "ModelTraining" / "train" / scan_id
    logging.info("Done in %.1f min -> %s", (time.perf_counter() - t0) / 60, mt)
    return 0 if mt.is_dir() else 1


if __name__ == "__main__":
    raise SystemExit(main())
