#!/usr/bin/env python3
"""A2 zero-shot TRE: SPARE MC pooled VoxelMap ckpt on DIR A1 ModelTraining.

Reuses A1 DIR case dirs (projections / volumes) and the same TRE bridge as
eval_a1_tre.py. Writes under arms/A2_generic_spare/runs/DIR_C0N/tre/.

  cd "DIR EXPERIMENTS"
  LEARN-GUI/.venv/bin/python scripts/eval_a2_tre.py --case 1 --gpu 0
"""
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
A2 = DIR_EXP / "arms" / "A2_generic_spare"
DEFAULT_CKPT = A2 / "checkpoints" / "a2_spare_mc_val_prior_p1to9_concat_nofilm.pt"
# Pre-R3 archive (SPARE-trained weights — still valid for zero-shot)
FALLBACK_CKPT = (
    DIR_EXP
    / "arms"
    / "Incorrect DRR"
    / "A2_generic_spare"
    / "checkpoints"
    / "a2_spare_mc_val_prior_p1to9_concat_nofilm.pt"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", type=int, required=True, choices=range(1, 11))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument(
        "--r3",
        action="store_true",
        help="DVF/landmarks on R3-reoriented A1 DIR grid (required for current A1 runs)",
    )
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("DIRLAB_ROOT", str(DIR_EXP / "data" / "dirlab_packs"))

    scan_id = f"DIR_C{args.case:02d}"
    a1_root = A1 / "runs" / scan_id
    mt = a1_root / "ModelTraining" / "train" / scan_id
    train = a1_root / scan_id / "train"
    ckpt = (args.ckpt or (DEFAULT_CKPT if DEFAULT_CKPT.is_file() else FALLBACK_CKPT)).resolve()
    if not ckpt.is_file():
        raise SystemExit(f"Missing ckpt: {ckpt}")
    if not mt.is_dir():
        raise SystemExit(f"Missing A1 ModelTraining: {mt}")

    out_root = A2 / "runs" / scan_id
    out_dir = out_root / "tre"
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = "r3" if args.r3 else "native"
    print(
        f"A2 Case {args.case} | zero-shot TRE (75+300) | frame={frame} | "
        f"ckpt={ckpt.name} | data={mt}"
    )

    results = {
        "arm": "A2_generic_spare",
        "case": args.case,
        "sets": ["75", "300"],
        "scan_id": scan_id,
        "frame": frame,
        "checkpoint": str(ckpt),
        "data_dir": str(mt),
        "stride": args.stride,
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
