#!/usr/bin/env python3
"""E5: QC-only robust CT intensity normalization sweep (E3 Both).

Baseline QC uses global min–max. E5 replaces that with percentile clips and
optional µ air/water anchors — no retrain.

  PYTHONPATH=Experiment3 python ClinicalExperiments/scripts/e5_intensity_norm_qc.py --gpu 1

Success: Elekta mean cos ≥ 0.28; hard patients rise more.
Kill: intensity distance to SPARE drops but cos does not.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

CE = Path(__file__).resolve().parents[1]
E1 = CE / "Experiment1"
E3 = CE / "Experiment3"
IE1 = CE.parent / "InitialExperiments" / "Experiment1"
sys.path.insert(0, str(IE1))
sys.path.insert(0, str(E3))

from losses.losses import DVFLoss  # noqa: E402
from networks.generator_crb_both import UNetCRBBoth  # noqa: E402

CKPT = E3 / "BothCRB/weights/crb_both_mse_cyclic_fov_aug_generator.pth"
OUT = CE / "plots" / "e5_intensity_norm"
PCT_PROFILE = (1, 5, 25, 50, 75, 95, 99)

# Sweep
NORMS = [
    ("minmax", None),  # baseline
    ("p0.5_p99.5", (0.5, 99.5)),
    ("p1_p99", (1.0, 99.0)),
    ("p2_p98", (2.0, 98.0)),
    ("p5_p95", (5.0, 95.0)),
    ("mu_air_water", "mu"),  # air=0, water=0.02
]


def norm_minmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi <= lo else (x - lo) / (hi - lo)


def norm_percentile(x: np.ndarray, plo: float, phi: float) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, [plo, phi]).astype(np.float32)
    if hi <= lo:
        return np.zeros_like(x)
    y = np.clip(x, lo, hi)
    return (y - lo) / (hi - lo)


def norm_mu_air_water(x: np.ndarray, air: float = 0.0, water: float = 0.02) -> np.ndarray:
    """Map µ≈air→0, µ≈water→1; clip to [0, 1.5] then squash to [0,1] via /1.5."""
    x = x.astype(np.float32)
    y = (x - air) / max(water - air, 1e-8)
    y = np.clip(y, 0.0, 1.5) / 1.5
    return y


def apply_norm(x: np.ndarray, name: str, spec) -> np.ndarray:
    if name == "minmax":
        return norm_minmax(x)
    if name == "mu_air_water":
        return norm_mu_air_water(x)
    plo, phi = spec
    return norm_percentile(x, plo, phi)


def profile(x: np.ndarray) -> np.ndarray:
    return np.percentile(x, PCT_PROFILE).astype(np.float64)


def to_cdhw(dvf: np.ndarray) -> np.ndarray:
    if dvf.ndim == 4 and dvf.shape[-1] == 3:
        return np.moveaxis(dvf, -1, 0)
    return dvf


def metrics(pred, gt, mask, dvf_l1):
    m = mask > 0.5
    gt_t = torch.from_numpy(gt)[None].float()
    pr_t = torch.from_numpy(pred)[None].float()
    mk_t = torch.from_numpy(mask)[None, None].float()
    l1 = float(dvf_l1.loss(gt_t, pr_t, mk_t).item())
    l1z = float(dvf_l1.loss(gt_t, torch.zeros_like(gt_t), mk_t).item())
    if m.any() and float(np.abs(gt).max()) > 0:
        a, b = pred[:, m], gt[:, m]
        cos = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-8))
        k = float(np.linalg.norm(pred[:, m]) / (np.linalg.norm(gt[:, m]) + 1e-8))
    else:
        cos, k = float("nan"), float("nan")
    return dict(l1=l1, l1_zero=l1z, ratio=l1 / (l1z + 1e-8), cos=cos, k=k)


def summarize(rows):
    directed = [r for r in rows if r["ref"] != r["tgt"]]
    return {
        "n_directed": len(directed),
        "mean_L1": float(np.mean([r["l1"] for r in directed])),
        "mean_L1_over_zero": float(np.mean([r["ratio"] for r in directed])),
        "mean_cos": float(np.nanmean([r["cos"] for r in directed])),
        "mean_k": float(np.nanmean([r["k"] for r in directed])),
        "beat_zero_pct": float(100.0 * sum(r["ratio"] < 1.0 for r in directed) / max(len(directed), 1)),
    }


def load_spare_ref_profiles() -> dict[str, np.ndarray]:
    """Build SPARE-like reference profiles from Varian CV (closest in-domain clinical).

    True SPARE CTs aren't packed as CT_*.npy in pooled; Varian pack intensities
    are the practical in-domain reference for this QC-only study.
    """
    refs = {name: [] for name, _ in NORMS}
    for i in range(1, 6):
        raw = np.load(E1 / "data" / f"CV_P{i}_V_01" / "all" / "CT_01.npy")
        for name, spec in NORMS:
            refs[name].append(profile(apply_norm(raw, name, spec)))
    return {k: np.mean(v, axis=0) for k, v in refs.items()}


def run_scan(scan: str, g, device, dvf_l1, spare_prof: dict):
    data_dir = E1 / "data" / scan / "all"
    mask = (np.load(data_dir / "Mask_Lung.npy") > 0).astype(np.float32)
    stems = [f"{i:02d}_to_{j:02d}" for i in range(1, 11) for j in range(1, 11)]
    raw_cts = {ph: np.load(data_dir / f"CT_{ph:02d}.npy").astype(np.float32) for ph in range(1, 11)}

    out = {}
    for name, spec in NORMS:
        rows = []
        # intensity distance of phase-01 to Varian-ref under this norm
        p01 = apply_norm(raw_cts[1], name, spec)
        dist = float(np.linalg.norm(profile(p01) - spare_prof[name]))
        with torch.no_grad():
            for stem in stems:
                ref, tgt = int(stem[:2]), int(stem[-2:])
                ct_r = apply_norm(raw_cts[ref], name, spec)
                if ref == tgt:
                    gt = np.zeros((3,) + ct_r.shape, np.float32)
                else:
                    gt = to_cdhw(np.load(data_dir / f"{stem}_pair.npy").astype(np.float32))
                ref_t = torch.from_numpy(ct_r)[None, None].to(device)
                pred = g(
                    ref_t,
                    torch.tensor([ref - 1], device=device),
                    torch.tensor([tgt - 1], device=device),
                )[0].cpu().numpy()
                m = metrics(pred, gt, mask, dvf_l1)
                rows.append(dict(pair=stem, ref=ref, tgt=tgt, **m))
        summ = summarize(rows)
        summ["intensity_dist_to_varian_ref"] = dist
        out[name] = summ
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument(
        "--scans",
        default=",".join(
            [f"CV_P{i}_V_01" for i in range(1, 6)] + [f"CE_P{i}_V_01" for i in range(1, 6)]
        ),
    )
    args = ap.parse_args()
    scans = [s.strip() for s in args.scans.split(",") if s.strip()]

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    g = UNetCRBBoth(im_size=128, n_phases=10).to(device)
    assert g.cond_dim == 4
    g.load_state_dict(torch.load(CKPT, map_location=device))
    g.eval()
    dvf_l1 = DVFLoss()

    spare_prof = load_spare_ref_profiles()
    OUT.mkdir(parents=True, exist_ok=True)
    results = {"ckpt": str(CKPT), "norms": [n for n, _ in NORMS], "scans": {}}

    print(f"E5 intensity-norm QC  ckpt={CKPT.name}  device={device}", flush=True)
    print(f"norms={[n for n,_ in NORMS]}", flush=True)

    for scan in scans:
        print(f"\n=== {scan} ===", flush=True)
        summ = run_scan(scan, g, device, dvf_l1, spare_prof)
        results["scans"][scan] = summ
        for name, _ in NORMS:
            s = summ[name]
            print(
                f"  {name:14s}  L1/z={s['mean_L1_over_zero']:.3f}  cos={s['mean_cos']:.3f}  "
                f"k={s['mean_k']:.3f}  beat={s['beat_zero_pct']:.0f}%  "
                f"dist={s['intensity_dist_to_varian_ref']:.3f}",
                flush=True,
            )

    print("\n======== COHORT MEANS ========", flush=True)
    for vendor, prefix in (("varian", "CV_"), ("elekta", "CE_")):
        subset = [s for s in scans if s.startswith(prefix)]
        print(f"\n-- {vendor} (n={len(subset)}) --")
        for name, _ in NORMS:
            cos = np.mean([results["scans"][s][name]["mean_cos"] for s in subset])
            beat = np.mean([results["scans"][s][name]["beat_zero_pct"] for s in subset])
            dist = np.mean(
                [results["scans"][s][name]["intensity_dist_to_varian_ref"] for s in subset]
            )
            ratio = np.mean(
                [results["scans"][s][name]["mean_L1_over_zero"] for s in subset]
            )
            print(
                f"  {name:14s}  L1/z={ratio:.3f}  cos={cos:.3f}  beat={beat:.0f}%  dist={dist:.3f}",
                flush=True,
            )
        if vendor == "elekta":
            hard = [s for s in subset if any(f"P{i}" in s for i in (1, 2, 3))]
            easy = [s for s in subset if any(f"P{i}" in s for i in (4, 5))]
            print("  hard CE_P1–P3 / easy CE_P4–P5 cos:")
            for name, _ in NORMS:
                ch = np.mean([results["scans"][s][name]["mean_cos"] for s in hard])
                ce = np.mean([results["scans"][s][name]["mean_cos"] for s in easy])
                print(f"  {name:14s}  hard={ch:.3f}  easy={ce:.3f}", flush=True)

    # verdict
    base = np.mean([results["scans"][s]["minmax"]["mean_cos"] for s in scans if s.startswith("CE_")])
    print("\n======== VERDICT vs success (Elekta cos≥0.28) ========", flush=True)
    best_name, best_cos = "minmax", base
    for name, _ in NORMS:
        cos = float(
            np.mean([results["scans"][s][name]["mean_cos"] for s in scans if s.startswith("CE_")])
        )
        dist = float(
            np.mean(
                [
                    results["scans"][s][name]["intensity_dist_to_varian_ref"]
                    for s in scans
                    if s.startswith("CE_")
                ]
            )
        )
        flag = "PASS" if cos >= 0.28 else "fail"
        print(f"  {name:14s}  elekta_cos={cos:.3f}  dist={dist:.3f}  [{flag}]", flush=True)
        if cos > best_cos:
            best_cos, best_name = cos, name
    base_dist = float(
        np.mean(
            [
                results["scans"][s]["minmax"]["intensity_dist_to_varian_ref"]
                for s in scans
                if s.startswith("CE_")
            ]
        )
    )
    best_dist = float(
        np.mean(
            [
                results["scans"][s][best_name]["intensity_dist_to_varian_ref"]
                for s in scans
                if s.startswith("CE_")
            ]
        )
    )
    if best_cos < 0.28 and best_dist < base_dist - 1e-6 and best_cos < base + 0.01:
        print(
            f"KILL-ish: best={best_name} lowered dist {base_dist:.3f}→{best_dist:.3f} "
            f"but cos {base:.3f}→{best_cos:.3f} (no real lift)",
            flush=True,
        )
    elif best_cos >= 0.28:
        print(f"SUCCESS: {best_name} Elekta cos={best_cos:.3f}", flush=True)
    else:
        print(
            f"No pass: best={best_name} cos={best_cos:.3f} (need ≥0.28); "
            f"delta vs minmax={best_cos - base:+.3f}",
            flush=True,
        )

    out_json = OUT / "e5_summary.json"
    out_json.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
