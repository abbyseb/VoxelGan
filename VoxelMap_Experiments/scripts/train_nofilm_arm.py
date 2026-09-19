#!/usr/bin/env python3
"""Train concatenated NoFiLM VoxelMap on a prepared arm under VoxelMap_Experiments/runs/."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
VMC = Path(os.environ.get("VOXELMAP_CLINICAL_ROOT", "/home/abhishek/Documents/VoxelMap_Clinical"))
LEARN = Path(os.environ.get("LEARN_GUI_ROOT", "/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python"))
PY = LEARN / ".venv/bin/python"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm",
        required=True,
        choices=["elastix", "synth", "synth_g160_a1_dec", "synth_e3_both"],
    )
    ap.add_argument("--scan-id", default="CV_P3_V_01")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    from run_paths import resolve_arm, run_root as rr

    arm = resolve_arm(args.arm)
    run_root = rr(args.scan_id, arm)
    data = run_root / "ModelTraining" / "train" / args.scan_id
    if not (data / "TargetProjections").is_dir():
        raise SystemExit(f"Missing ModelTraining data: {data}")

    ckpt_dir = run_root / "checkpoints_nofilm"
    plots_dir = run_root / "plots_nofilm"
    (run_root / "logs").mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_path = ckpt_dir / f"{arm}_{args.scan_id}_concat_nofilm.pt"
    log = run_root / "logs" / "train_nofilm.log"

    python = PY if PY.is_file() else Path(sys.executable)
    cmd = [
        str(python),
        str(VMC / "ml" / "trainer.py"),
        "--data_dirs",
        str(data),
        "--architecture",
        "concatenated",
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        "1e-5",
        "--val_split",
        "0.1",
        "--device",
        "cuda",
        "--num_workers",
        str(args.num_workers),
        "--checkpoint_dir",
        str(ckpt_dir),
        "--plots_dir",
        str(plots_dir),
        "--save_path",
        str(save_path),
        # No --use_film → NoFiLM
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["VOXELMAP_CLINICAL_ROOT"] = str(VMC)
    env["LEARN_GUI_ROOT"] = str(LEARN)

    print("CMD:", " ".join(cmd))
    print("LOG:", log)
    with open(log, "w") as f:
        f.write("CMD: " + " ".join(cmd) + "\n")
        f.flush()
        return subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
