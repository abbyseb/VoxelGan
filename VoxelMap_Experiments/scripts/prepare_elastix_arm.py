#!/usr/bin/env python3
"""Fresh Elastix VoxelMap prep for a Varian clinical scan (Arm A)."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
VMC = Path(os.environ.get("VOXELMAP_CLINICAL_ROOT", "/home/abhishek/Documents/VoxelMap_Clinical"))
LEARN = Path(os.environ.get("LEARN_GUI_ROOT", "/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python"))
PY = LEARN / ".venv/bin/python"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-id", default="CV_P3_V_01")
    ap.add_argument("--with-test", action="store_true", default=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--clean", action="store_true", default=True)
    args = ap.parse_args()

    from run_paths import run_root as rr

    scan_id = args.scan_id
    run_root = rr(scan_id, "elastix")
    pnum = scan_id.split("_")[1][1:]  # "3"
    staged = EXP / "data" / "staged" / f"P{pnum}" / scan_id

    if not staged.is_dir():
        raise SystemExit(f"Staged data missing: {staged}\nRun stage_varian_scan first.")

    if args.clean and run_root.exists():
        print(f"Cleaning {run_root}")
        shutil.rmtree(run_root)
    (run_root / "logs").mkdir(parents=True, exist_ok=True)
    print(f"run_root={run_root}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["VOXELMAP_CLINICAL_ROOT"] = str(VMC)
    env["LEARN_GUI_ROOT"] = str(LEARN)

    cmd = [
        str(PY if PY.is_file() else sys.executable),
        str(VMC / "scripts" / "run_varian_phase2.py"),
        "--scan-id",
        scan_id,
        "--run-root",
        str(run_root),
        "--staged",
        str(staged),
    ]
    if args.with_test:
        cmd.append("--with-test")

    log = run_root / "logs" / "prepare_elastix_arm.log"
    print("CMD:", " ".join(cmd))
    print("LOG:", log)
    with open(log, "w") as f:
        proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
