#!/usr/bin/env python3
"""Shared run-path helpers for VoxelMap_Experiments.

Layout:
  runs/<scan_id>/<arm>/
    e.g. runs/CV_P3_V_01/elastix/
         runs/CV_P3_V_01/synth_g160_a1_dec/
         runs/CV_P3_V_01/synth_e3_both/   (legacy)
"""
from __future__ import annotations

from pathlib import Path

EXP = Path(__file__).resolve().parents[1]

# Canonical arm folder names under runs/<scan_id>/
ARMS = {
    "elastix": "elastix",
    "synth": "synth_g160_a1_dec",  # default synth = current best
    "synth_g160_a1_dec": "synth_g160_a1_dec",
    "synth_e3_both": "synth_e3_both",
}


def run_root(scan_id: str, arm: str) -> Path:
    folder = ARMS.get(arm, arm)
    return EXP / "runs" / scan_id / folder


def resolve_arm(arm: str) -> str:
    if arm not in ARMS and arm not in ARMS.values():
        raise SystemExit(f"Unknown arm {arm!r}. Choose from: {sorted(set(ARMS)|set(ARMS.values()))}")
    return ARMS.get(arm, arm)
