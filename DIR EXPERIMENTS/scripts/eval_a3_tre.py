#!/usr/bin/env python3
"""A3 TRE: synth-trained VoxelMap ckpt on DIR A1 ModelTraining (real DRRs + landmarks)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

DIR_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIR_EXP / "scripts"))

from eval_a1_tre import (  # noqa: E402
    infer_voxelmap_dvf_phase01,
    load_dvf_zyx3,
    tre_t00_t50,
    tre_t50_t00,
)

A1 = DIR_EXP / "arms" / "A1_oracle_dirlab"
A3 = DIR_EXP / "arms" / "A3_synth_conditioned"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", type=int, required=True, choices=range(1, 11))
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument(
        "--run-suffix",
        default=None,
        help="A3 run folder suffix (e.g. muhist → runs/DIR_C01_muhist)",
    )
    ap.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Override A3 run root (default: arms/A3.../runs/DIR_C0N[_suffix])",
    )
    ap.add_argument(
        "--r3",
        action="store_true",
        help="DVF/landmarks on R3-reoriented A1 DIR grid",
    )
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("DIRLAB_ROOT", str(DIR_EXP / "data" / "dirlab_packs"))

    scan_id = f"DIR_C{args.case:02d}"
    if args.run_root is not None:
        run = args.run_root
    else:
        run_name = f"{scan_id}_{args.run_suffix}" if args.run_suffix else scan_id
        run = A3 / "runs" / run_name
    ckpt = args.ckpt or (run / "checkpoints_nofilm" / "best.pt")
    if not ckpt.is_file():
        raise SystemExit(f"Missing ckpt: {ckpt}")

    a1_root = A1 / "runs" / scan_id
    mt = a1_root / "ModelTraining" / "train" / scan_id
    train = a1_root / scan_id / "train"
    if not mt.is_dir():
        raise SystemExit(f"Missing A1 ModelTraining: {mt}")

    out_dir = run / "tre"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = "r3" if args.r3 else "native"
    print(f"A3 Case {args.case} | TRE | frame={frame} | run={run.name} | ckpt={ckpt.name} | data={mt}")

    results = {
        "arm": "A3_synth_conditioned",
        "case": args.case,
        "sets": ["75", "300"],
        "scan_id": scan_id,
        "frame": frame,
        "run": str(run),
        "checkpoint": str(ckpt),
        "data_dir": str(mt),
        "stride": args.stride,
        "note": "VoxelMap trained on G160 synth 4D; TRE on A1 DIR ModelTraining + landmarks",
        "arms": {},
    }

    def _store_arm(label: str, path: Path | None, dvf: np.ndarray, extra: dict | None = None):
        arm = results["arms"].setdefault(label, {})
        if path is not None:
            arm["path"] = str(path)
        if extra:
            arm.update(extra)
        for which in ("75", "300"):
            r00 = tre_t00_t50(dvf, args.case, which, r3=args.r3)
            r50 = tre_t50_t00(dvf, args.case, which, r3=args.r3)
            arm[which] = {"T00_T50": r00, "T50_T00": r50}
            if which == "75":
                arm["T00_T50"] = r00
                arm["T50_T00"] = r50
            print(
                f"{label:12s} [{which:>3s}] T00→T50 TRE={r00['registered']['mean']:.3f} mm "
                f"(identity {r00['identity']['mean']:.3f}, Δ {r00['improvement_mm']:+.3f})"
            )

    for label, path in [
        ("elastix_mha", train / "DVF_sub_01.mha"),
        ("elastix_npy", mt / "DVFs" / "DVF_01_mha.npy"),
    ]:
        if path.is_file():
            _store_arm(label, path, load_dvf_zyx3(path))

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dvf_vm, n = infer_voxelmap_dvf_phase01(ckpt, mt, device, stride=args.stride)
    np.save(out_dir / "voxelmap_dvf_phase01_mean.npy", dvf_vm.astype(np.float32))
    extra = {"n_proj_averaged": n, "checkpoint": str(ckpt)}
    gt_path = mt / "DVFs" / "DVF_01_mha.npy"
    if gt_path.is_file():
        gt = load_dvf_zyx3(gt_path)
        extra["l1_vs_elastix_zyx"] = float(np.mean(np.abs(dvf_vm - gt)))
        print(f"{'voxelmap':12s}       L1 vs elastix: {extra['l1_vs_elastix_zyx']:.4f}")
    _store_arm("voxelmap", None, dvf_vm, extra=extra)
    print(f"{'voxelmap':12s}       averaged over {n} projs (stride={args.stride})")

    out_json = out_dir / "tre_summary.json"
    out_json.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
