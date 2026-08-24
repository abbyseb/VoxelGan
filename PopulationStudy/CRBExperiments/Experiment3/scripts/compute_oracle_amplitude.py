#!/usr/bin/env python3
"""Compute oracle amplitudes.json for CRBExperiments Experiment 1."""
from __future__ import annotations

import json
import sys
from pathlib import Path

E1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E1))

from utilities.amplitude import compute_table
from utilities.seed_utils import load_seed_config


def main():
    cfg = load_seed_config()
    train = list(cfg["train_patients"])
    table = compute_table(train)
    out = E1 / "amplitudes.json"
    out.write_text(json.dumps(table, indent=2))
    print(f"wrote {out}")
    print(f"A_train_mean_mm={table['A_train_mean_mm']:.4f} spacing={table['spacing_mm']}")
    for pid in sorted(table["patients"], key=lambda p: -table["patients"][p]["A_p_mm"]):
        r = table["patients"][pid]
        print(
            f"  {pid}: A_mm={r['A_p_mm']:.3f} A_vox={r['A_p']:.3f} "
            f"r_p={r['r_p']:.3f} log_r={r['log_r_p']:.3f} pair={r['extreme_pair']}"
        )


if __name__ == "__main__":
    main()
