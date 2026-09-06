#!/usr/bin/env python3
"""Build Elastix DVF library on packed DIR-Lab volumes (10×10 pairs per patient).

Reads CT_01..10 + Mask_Lung from Experiment1/packed/P*_DIR/all/ (µ, 128³).
Writes ref_to_tgt_pair.npy — same layout as ClinicalExperiments.

  cd PopulationStudy/DIR-Experiments/Experiment1
  /path/to/LEARN-GUI/.venv/bin/python scripts/prepare_dir_dvf_library.py --patient P1_DIR
  ... --patients P1_DIR,...,P10_DIR --skip_existing
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

E1 = Path(__file__).resolve().parents[1]
DIR = E1.parent
PACKED = E1 / "packed"
CE_SCRIPTS = DIR.parent / "ClinicalExperiments" / "scripts"
REPO = DIR.parent.parent
PARAM_DEFAULT = REPO / "My v1.0" / "configs" / "elastix_bspline_masked.txt"

sys.path.insert(0, str(CE_SCRIPTS))
from prepare_clinical_dvf_library import register_pair  # noqa: E402


def process_patient(
    pid: str,
    packed_root: Path,
    param_file: Path,
    skip_existing: bool,
    max_pairs: int | None,
):
    all_dir = packed_root / pid / "all"
    if not all_dir.is_dir():
        raise FileNotFoundError(all_dir)

    cts = [np.load(all_dir / f"CT_{i:02d}.npy").astype(np.float32) for i in range(1, 11)]
    mask = (np.load(all_dir / "Mask_Lung.npy") > 0).astype(np.uint8)
    print(f"[{pid}] packed CT {cts[0].shape}  mask_vox={int(mask.sum())}", flush=True)

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
            if n % 25 == 0 or n == len(jobs):
                print(f"[{pid}] [{n}/{len(jobs)}] skip {name}", flush=True)
            continue
        if ref == tgt:
            dvf = np.zeros(cts[0].shape + (3,), dtype=np.float32)
            n_id += 1
            print(f"[{pid}] [{n}/{len(jobs)}] identity {name}", flush=True)
        else:
            print(f"[{pid}] [{n}/{len(jobs)}] Elastix {name} …", flush=True)
            tic = time.time()
            dvf = register_pair(cts[tgt - 1], cts[ref - 1], mask, param_file)
            print(f"[{pid}]     {time.time()-tic:.0f}s", flush=True)
            n_done += 1
        np.save(out_path, dvf.astype(np.float32))

    print(
        f"[{pid}] DONE {time.time()-t0:.0f}s  reg={n_done} id={n_id} skip={n_skip} → {all_dir}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patient", default="P1_DIR")
    ap.add_argument("--patients", default="", help="comma list overrides --patient")
    ap.add_argument("--packed-root", type=Path, default=PACKED)
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
    print(f"DIR DVF library  n={len(patients)}  packed={args.packed_root}", flush=True)
    for pid in patients:
        process_patient(pid, args.packed_root, args.param_file, args.skip_existing, args.max_pairs)
    print("All requested patients finished.", flush=True)


if __name__ == "__main__":
    main()
