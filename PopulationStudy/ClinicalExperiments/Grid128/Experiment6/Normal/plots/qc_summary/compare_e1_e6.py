#!/usr/bin/env python3
"""Print E1 vs E6 clinical QC comparison (run after qc_clinical.py)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

E6 = Path(__file__).resolve().parents[2]
CE = E6.parents[1]
scans = [f"CV_P{i}_V_01" for i in range(1, 6)] + [f"CE_P{i}_V_01" for i in range(1, 6)]


def load_e1(arch, scan, mu=False):
    vend = "elekta" if scan.startswith("CE") else "varian"
    p = scan.split("_")[1]
    root = CE / "Experiment1" / ("EXPT5_SETTINGS" if mu else "")
    f = root / f"{arch}CRB/plots/qc_{vend}/{p}/{scan}/summary.json"
    return json.loads(f.read_text()) if f.is_file() else None


def load_e6(arch, scan, norm):
    vend = "elekta" if scan.startswith("CE") else "varian"
    p = scan.split("_")[1]
    f = E6 / f"{arch}CRB/plots/qc_{vend}/{norm}/{p}/{scan}/summary.json"
    return json.loads(f.read_text()) if f.is_file() else None


def main():
    lines = ["arch\tnorm\tscan\tE1_cos\tE6_cos\tdelta_cos\tE1_beat\tE6_beat\tdelta_beat"]
    for arch in ("Encoder", "Decoder", "Both"):
        for norm, mu in (("minmax", False), ("mu", True)):
            for s in scans:
                e1, e6 = load_e1(arch, s, mu=mu), load_e6(arch, s, norm)
                if not e1 or not e6:
                    continue
                lines.append(
                    f"{arch}\t{norm}\t{s}\t{e1['mean_cos']:.4f}\t{e6['mean_cos']:.4f}\t"
                    f"{e6['mean_cos']-e1['mean_cos']:+.4f}\t{e1['beat_zero_pct']:.1f}\t"
                    f"{e6['beat_zero_pct']:.1f}\t{e6['beat_zero_pct']-e1['beat_zero_pct']:+.1f}"
                )
    out = E6 / "plots/qc_summary/compare_e1_e6.tsv"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
