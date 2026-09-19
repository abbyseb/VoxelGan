#!/usr/bin/env python3
"""Smoke: G160 synth DVF_sub_01 TRE vs A1 Elastix on R3 DIR (no VoxelMap train).

  cd "DIR EXPERIMENTS"
  LEARN-GUI/.venv/bin/python scripts/eval_a3_synth_vs_elastix.py --case 1 --r3
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

from eval_a1_tre import load_dvf_zyx3, tre_t00_t50, tre_t50_t00  # noqa: E402

A1 = DIR_EXP / "arms" / "A1_oracle_dirlab"
A3 = DIR_EXP / "arms" / "A3_synth_conditioned"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", type=int, required=True, choices=range(1, 11))
    ap.add_argument("--r3", action="store_true", default=True)
    ap.add_argument("--no-r3", action="store_true")
    ap.add_argument("--run-suffix", default=None)
    args = ap.parse_args()
    r3 = not args.no_r3

    os.environ.setdefault("DIRLAB_ROOT", str(DIR_EXP / "data" / "dirlab_packs"))
    scan_id = f"DIR_C{args.case:02d}"
    run_name = f"{scan_id}_{args.run_suffix}" if args.run_suffix else scan_id
    a3_train = A3 / "runs" / run_name / scan_id / "train"
    a1_train = A1 / "runs" / scan_id / scan_id / "train"

    synth = a3_train / "DVF_sub_01.mha"
    elastix = a1_train / "DVF_sub_01.mha"
    if not synth.is_file():
        raise SystemExit(f"Missing synth DVF: {synth}")
    if not elastix.is_file():
        raise SystemExit(f"Missing A1 Elastix DVF: {elastix}")

    out_dir = A3 / "runs" / run_name / "tre_synth_oracle"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = "r3" if r3 else "native"
    print(f"A3 synth-vs-Elastix | case={args.case} | frame={frame}")
    print(f"  synth   {synth}")
    print(f"  elastix {elastix}")

    results = {
        "arm": "A3_synth_oracle",
        "case": args.case,
        "frame": frame,
        "sets": ["75", "300"],
        "synth_dvf": str(synth),
        "elastix_dvf": str(elastix),
        "arms": {},
    }

    def store(label: str, path: Path):
        dvf = load_dvf_zyx3(path)
        arm = results["arms"].setdefault(label, {"path": str(path)})
        for which in ("75", "300"):
            r00 = tre_t00_t50(dvf, args.case, which, r3=r3)
            r50 = tre_t50_t00(dvf, args.case, which, r3=r3)
            arm[which] = {"T00_T50": r00, "T50_T00": r50}
            print(
                f"{label:12s} [{which:>3s}] T00→T50 TRE={r00['registered']['mean']:.3f} mm "
                f"(id {r00['identity']['mean']:.3f}, Δid {r00['improvement_mm']:+.3f}) | "
                f"T50→T00 TRE={r50['registered']['mean']:.3f}"
            )

    store("elastix_a1", elastix)
    store("synth_g160", synth)

    # L1 in sub-voxels (same grid)
    e = load_dvf_zyx3(elastix)
    s = load_dvf_zyx3(synth)
    if e.shape == s.shape:
        l1 = float(np.mean(np.abs(s - e)))
        results["l1_synth_vs_elastix_zyx"] = l1
        print(f"{'L1 synth vs Elastix':12s}       {l1:.4f} (sub-vox)")

    out = out_dir / "tre_summary.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
