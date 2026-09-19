#!/usr/bin/env python3
"""P2: G160 conditioning sensitivity — same phase, different patient mid-CTs.

Expect if collapsed: high pairwise cos (same direction), different |u|.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

DIR_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIR_EXP / "scripts"))
from prepare_a3_dir_case import (  # noqa: E402
    SYNTH_SPEC,
    dir_hu_to_mu_for_synth,
    load_generator,
    norm_mu,
    resize_volume_zyx,
)

OUT = DIR_EXP / "arms" / "A3_synth_conditioned" / "results"


def load_mid_ct(path: Path) -> tuple[np.ndarray, str]:
    a = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
    kind = "mu" if float(a.max()) < 1.0 else "hu"
    return a, kind


def to_g160_input(vol: np.ndarray, kind: str, hu_mode: str = "default") -> np.ndarray:
    if kind == "hu":
        mu, _ = dir_hu_to_mu_for_synth(vol, hu_mode)
    else:
        mu = vol
    return norm_mu(resize_volume_zyx(mu, int(SYNTH_SPEC["infer_size"])))


def cos_mot(a: np.ndarray, b: np.ndarray, pct: float = 50.0) -> float:
    """a,b: (3,D,H,W)."""
    am = np.sqrt((a**2).sum(0))
    bm = np.sqrt((b**2).sum(0))
    m = am > np.percentile(am, pct)
    m &= bm > 1e-6
    return float(((a * b).sum(0)[m] / (am[m] * bm[m])).mean())


def main() -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"P2 device={device}")

    cases = [
        (
            "MC_P1",
            Path(
                "/home/abhishek/Voxel_GAN/VoxelMap_Experiments/runs/MC_V_P1_NS_01/"
                "synth_g160_a1_dec/MC_V_P1_NS_01/train/CT_06.mha"
            ),
            "mu",
            "default",
        ),
        (
            "CV_P3",
            Path(
                "/home/abhishek/Voxel_GAN/VoxelMap_Experiments/runs/CV_P3_V_01/"
                "elastix/CV_P3_V_01/train/CT_06.mha"
            ),
            "auto",
            "default",
        ),
        (
            "DIR_C01",
            Path(
                "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/arms/A1_oracle_dirlab/"
                "runs/DIR_C01/DIR_C01/train/CT_06.mha"
            ),
            "hu",
            "default",
        ),
        (
            "DIR_C01_muhist",
            Path(
                "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/arms/A1_oracle_dirlab/"
                "runs/DIR_C01/DIR_C01/train/CT_06.mha"
            ),
            "hu",
            "hist_match",
        ),
        (
            "DIR_C08",
            Path(
                "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/arms/A1_oracle_dirlab/"
                "runs/DIR_C08/DIR_C08/train/CT_06.mha"
            ),
            "hu",
            "default",
        ),
    ]

    g = load_generator(SYNTH_SPEC, device)
    ref_ph = torch.tensor([5], dtype=torch.long, device=device)
    tgt_ph = torch.tensor([0], dtype=torch.long, device=device)  # phase 01

    fields: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}
    for name, path, kind, hu_mode in cases:
        if not path.is_file():
            print(f"SKIP {name}: missing {path}")
            continue
        vol, detected = load_mid_ct(path)
        if kind == "auto":
            kind = detected
        x = to_g160_input(vol, kind, hu_mode)
        xt = torch.from_numpy(x[None, None]).to(device)
        with torch.no_grad():
            dvf = g(xt, ref_ph, tgt_ph)[0].cpu().numpy()  # (3,160,160,160)
        mag = np.sqrt((dvf**2).sum(0))
        fields[name] = dvf
        meta[name] = {
            "path": str(path),
            "kind": kind,
            "hu_mode": hu_mode,
            "mean_|u|": float(mag.mean()),
            "p50_|u|": float(np.median(mag)),
            "absmean_xyz": [float(np.abs(dvf[i]).mean()) for i in range(3)],
        }
        print(
            f"{name:16s} |u|={meta[name]['mean_|u|']:.4f} "
            f"xyz_abs={meta[name]['absmean_xyz']}"
        )
        np.save(OUT / f"p2_dvf_{name}_phase01.npy", dvf.astype(np.float32))

    names = list(fields)
    pairs = {}
    print("\nPairwise cos_mot50 (1 = same direction):")
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            c = cos_mot(fields[a], fields[b], 50)
            pairs[f"{a}_vs_{b}"] = c
            print(f"  {a:16s} vs {b:16s}: {c:.4f}")

    # Collapse signature: high mean pairwise cos among SPARE-like + DIR
    mean_pair = float(np.mean(list(pairs.values()))) if pairs else float("nan")
    verdict = (
        "COLLAPSE_LIKELY"
        if mean_pair > 0.7
        else ("MIXED" if mean_pair > 0.35 else "PATIENT_SPECIFIC_DIRECTION")
    )

    out = {
        "probe": "P2_g160_patient_sensitivity",
        "phase": "01_from_06",
        "device": str(device),
        "per_case": meta,
        "pairwise_cos_mot50": pairs,
        "mean_pairwise_cos": mean_pair,
        "verdict": verdict,
        "note": (
            "High pairwise cos + different |u| ⇒ shared motion template (amplitude-only conditioning). "
            "Low cos ⇒ direction changes with patient (conditioning works for direction, may still be wrong vs Elastix)."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "p2_patient_sensitivity.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nverdict={verdict} mean_pairwise_cos={mean_pair:.4f}")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
