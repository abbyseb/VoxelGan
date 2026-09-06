#!/usr/bin/env python3
"""Run train-pair and/or breathing-sweep eval for a VoxelMap_Experiments arm."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
VMC = Path(os.environ.get("VOXELMAP_CLINICAL_ROOT", "/home/abhishek/Documents/VoxelMap_Clinical"))
LEARN = Path(os.environ.get("LEARN_GUI_ROOT", "/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python"))
PY = LEARN / ".venv/bin/python"


def run(cmd: list[str], env: dict, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write("CMD: " + " ".join(cmd) + "\n")
        f.flush()
        return subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm",
        required=True,
        choices=["elastix", "synth", "synth_g160_a1_dec", "synth_e3_both"],
    )
    ap.add_argument("--scan-id", default="CV_P3_V_01")
    ap.add_argument("--mode", default="both", choices=["train", "sweep", "both"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all (train eval only)")
    args = ap.parse_args()

    from run_paths import resolve_arm, run_root as rr

    arm = resolve_arm(args.arm)
    run_root = rr(args.scan_id, arm)
    ckpt_dir = run_root / "checkpoints_nofilm"
    # Prefer full trainer checkpoint (has config); best.pt is sometimes a raw state_dict.
    named = sorted(ckpt_dir.glob(f"{arm}_*_concat_nofilm.pt"))
    candidates = named + [ckpt_dir / "last.pt", ckpt_dir / "best.pt"]
    ckpt = next((p for p in candidates if p.is_file()), None)
    if ckpt is None:
        raise SystemExit(f"Missing checkpoint under {ckpt_dir}")
    print(f"Using checkpoint: {ckpt}", flush=True)

    train_data = run_root / "ModelTraining" / "train" / args.scan_id
    test_data = run_root / "ModelTraining" / "test" / args.scan_id
    for p in (train_data, test_data):
        if not (p / "TargetProjections").is_dir():
            raise SystemExit(f"Missing ModelTraining data: {p}")

    python = PY if PY.is_file() else Path(sys.executable)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["VOXELMAP_CLINICAL_ROOT"] = str(VMC)
    env["LEARN_GUI_ROOT"] = str(LEARN)

    log_dir = run_root / "logs"
    rc = 0

    if args.mode in ("sweep", "both"):
        out = run_root / "eval_sweep"
        log = log_dir / "eval_sweep.log"
        cmd = [
            str(python),
            str(VMC / "ml" / "sweep_evaluator.py"),
            "--checkpoint",
            str(ckpt),
            "--data_dir",
            str(test_data),
            "--output_dir",
            str(out),
            "--scan-id",
            f"{args.arm}_{args.scan_id}",
            "--device",
            "cuda",
        ]
        print("Sweep eval:", " ".join(cmd))
        rc = run(cmd, env, log) or rc

    if args.mode in ("train", "both"):
        out = run_root / "eval_train"
        log = log_dir / "eval_train.log"
        cmd = [
            str(python),
            str(VMC / "ml" / "evaluator.py"),
            "--checkpoint",
            str(ckpt),
            "--data_dir",
            str(train_data),
            "--output_dir",
            str(out),
            "--architecture",
            "concatenated",
            "--device",
            "cuda",
        ]
        if args.max_samples > 0:
            cmd += ["--max_samples", str(args.max_samples)]
        print("Train eval:", " ".join(cmd))
        rc = run(cmd, env, log) or rc

    # Aggregate comparison across arms under runs/<scan_id>/
    compare = EXP / "logs" / f"eval_comparison_{args.scan_id}.json"
    rows = {}
    for arm_name in ("elastix", "synth_g160_a1_dec", "synth_e3_both"):
        root = rr(args.scan_id, arm_name)
        row = {"arm": arm_name, "run_root": str(root)}
        tp = root / "eval_train" / "metrics.json"
        sp = root / "eval_sweep" / "metrics.json"
        if tp.is_file():
            row["train"] = json.loads(tp.read_text())["summary"]
        if sp.is_file():
            s = json.loads(sp.read_text())
            row["sweep"] = {
                "n_samples": len(s.get("angles", [])),
                "mean_dice": float(__import__("numpy").mean(s["dice"])),
                "mean_3d_error_mm": float(__import__("numpy").mean(s["shifts_mm"]["3d"])),
                "mean_psnr_db": float(__import__("numpy").mean(s["psnr"])),
                "mean_ssim": float(__import__("numpy").nanmean(s["ssim"])),
            }
        if "train" in row or "sweep" in row:
            rows[arm_name] = row
    if rows:
        compare.parent.mkdir(parents=True, exist_ok=True)
        compare.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"Wrote {compare}")
        print("\n======== SWEEP COMPARISON ========", flush=True)
        for name, row in rows.items():
            if "sweep" not in row:
                continue
            sw = row["sweep"]
            print(
                f"{name:22s}  dice={sw['mean_dice']:.4f}  "
                f"3d_err={sw['mean_3d_error_mm']:.3f} mm  "
                f"psnr={sw['mean_psnr_db']:.2f} dB  ssim={sw['mean_ssim']:.4f}",
                flush=True,
            )

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
