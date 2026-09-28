#!/usr/bin/env python3
"""Build TCIA3.1 manifest: MAE train + oversample **big peak breath** on 01→06.

Val split unchanged. High scans = lung **max |u|** or **p95 |u|** on Elastix 01→06
≥ thresholds (default max≥12 mm → ~56/82 scans). Very-high tier (default max≥20 mm)
gets more repeats.

  cd …/Grid160/TCIA3.1
  python scripts/build_oversampled_manifest.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

T1 = Path(__file__).resolve().parents[1]
DEFAULT_POOLED = Path("/media/abhishek/3CCA3CADCA3C6574/TCIA_4D-Lung/synth_g160_r3/pooled")

_PATIENT_RE = re.compile(r"^(S\d+)_")


def patient_tag(pair_name: str) -> str:
    m = _PATIENT_RE.match(pair_name)
    if not m:
        raise ValueError(f"unexpected pair name: {pair_name}")
    return m.group(1)


def is_0106_pair(pair_name: str) -> bool:
    return "_01_to_06_" in pair_name


def lung_u0106_stats_mm(im_dir: Path, patient: str) -> dict[str, float]:
    """Lung |u| on 01→06 pair: mean, p95, max (mm @ 2 mm iso)."""
    pair = im_dir / f"{patient}_01_to_06_pair.npy"
    lung_path = im_dir / f"{patient}_Mask_Lung.npy"
    if not pair.is_file() or not lung_path.is_file():
        raise FileNotFoundError(f"missing {pair} or {lung_path}")
    lung = np.load(lung_path) > 0
    dvf = np.load(pair, mmap_mode="r")
    if dvf.ndim == 4 and dvf.shape[0] == 3:
        ux = np.asarray(dvf[0][lung], dtype=np.float64)
        uy = np.asarray(dvf[1][lung], dtype=np.float64)
        uz = np.asarray(dvf[2][lung], dtype=np.float64)
        mag = np.sqrt(ux * ux + uy * uy + uz * uz)
    else:
        v = np.asarray(dvf[lung], dtype=np.float64)
        mag = np.linalg.norm(v, axis=-1)
    if mag.size == 0:
        return {"mean_mm": 0.0, "p95_mm": 0.0, "max_mm": 0.0}
    return {
        "mean_mm": float(mag.mean()),
        "p95_mm": float(np.percentile(mag, 95)),
        "max_mm": float(mag.max()),
    }


def load_base_manifest(pooled: Path) -> dict:
    for p in (T1 / "data" / "manifest_base.json", pooled / "manifest.json"):
        if p.is_file():
            return json.loads(p.read_text())
    raise FileNotFoundError(
        "No base manifest — run TCIA3 build_pooled_dataset.py or copy pooled/manifest.json"
    )


def scan_tier(
    stats: dict[str, float],
    *,
    min_max_mm: float,
    min_p95_mm: float,
    very_high_max_mm: float,
) -> str:
    mx, p95 = stats["max_mm"], stats["p95_mm"]
    high = mx >= min_max_mm - 1e-9 or p95 >= min_p95_mm - 1e-9
    if not high:
        return "normal"
    if mx >= very_high_max_mm - 1e-9:
        return "very_high"
    return "high"


def expand_train_pairs(
    train_pairs: list[str],
    tiers: dict[str, str],
    *,
    repeat_high: int,
    repeat_very_high: int,
    extra_0106_high: int,
    extra_0106_very_high: int,
) -> tuple[list[str], dict]:
    out: list[str] = []
    tier_counts = {"normal": 0, "high": 0, "very_high": 0}
    for p, t in tiers.items():
        tier_counts[t] = tier_counts.get(t, 0) + 1

    high_scans = sorted(p for p, t in tiers.items() if t == "high")
    very_high_scans = sorted(p for p, t in tiers.items() if t == "very_high")

    for name in train_pairs:
        tag = patient_tag(name)
        t = tiers[tag]
        rep = 1
        extra_0106 = 0
        if t == "high":
            rep = repeat_high
            extra_0106 = extra_0106_high
        elif t == "very_high":
            rep = repeat_very_high
            extra_0106 = extra_0106_very_high
        if is_0106_pair(name):
            rep += extra_0106
        out.extend([name] * rep)

    stats = {
        "criterion": "u0106_lung_peak",
        "repeat_high": repeat_high,
        "repeat_very_high": repeat_very_high,
        "extra_0106_high": extra_0106_high,
        "extra_0106_very_high": extra_0106_very_high,
        "n_scans_normal": tier_counts.get("normal", 0),
        "n_scans_high": tier_counts.get("high", 0),
        "n_scans_very_high": tier_counts.get("very_high", 0),
        "n_scans_oversampled": tier_counts.get("high", 0) + tier_counts.get("very_high", 0),
        "high_scans": high_scans,
        "very_high_scans": very_high_scans,
        "n_base_train_pairs": len(train_pairs),
        "n_train_pairs_oversampled": len(out),
        "oversample_factor_mean": len(out) / max(len(train_pairs), 1),
    }
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pooled", type=Path, default=DEFAULT_POOLED)
    ap.add_argument("--seed-json", type=Path, default=T1 / "seed.json")
    args = ap.parse_args()

    cfg = json.loads(args.seed_json.read_text())
    base = load_base_manifest(args.pooled)
    train_dir = Path(base["pooled_train_dir"])
    if not train_dir.is_dir():
        raise SystemExit(f"pooled train dir missing: {train_dir}")

    min_max = float(cfg.get("oversample_u0106_min_max_mm", 12.0))
    min_p95 = float(cfg.get("oversample_u0106_min_p95_mm", 12.0))
    very_max = float(cfg.get("oversample_u0106_very_high_max_mm", 20.0))

    patients = sorted({patient_tag(n) for n in base["train_pairs"] + base["val_pairs"]})
    u0106: dict[str, dict[str, float]] = {}
    tiers: dict[str, str] = {}
    for p in patients:
        u0106[p] = lung_u0106_stats_mm(train_dir, p)
        tiers[p] = scan_tier(
            u0106[p],
            min_max_mm=min_max,
            min_p95_mm=min_p95,
            very_high_max_mm=very_max,
        )

    train_os, os_stats = expand_train_pairs(
        list(base["train_pairs"]),
        tiers,
        repeat_high=int(cfg.get("oversample_high_scan_repeat", 4)),
        repeat_very_high=int(cfg.get("oversample_very_high_scan_repeat", 6)),
        extra_0106_high=int(cfg.get("oversample_pair_0106_extra_high", 3)),
        extra_0106_very_high=int(cfg.get("oversample_pair_0106_extra_very_high", 5)),
    )
    os_stats["min_max_mm"] = min_max
    os_stats["min_p95_mm"] = min_p95
    os_stats["very_high_max_mm"] = very_max

    means = [v["mean_mm"] for v in u0106.values()]
    maxes = [v["max_mm"] for v in u0106.values()]
    p95s = [v["p95_mm"] for v in u0106.values()]
    motion_report = {
        "pair": "01→06 Elastix",
        "per_scan": {k: {kk: round(vv, 4) for kk, vv in sorted(u0106[k].items())} for k in sorted(u0106)},
        "mean_of_scan_means_mm": float(np.mean(means)),
        "max_of_scan_maxes_mm": float(max(maxes)),
        "n_scans_max_ge_12": int(sum(1 for v in maxes if v >= 12.0)),
        "n_scans_p95_ge_12": int(sum(1 for v in p95s if v >= 12.0)),
    }

    manifest = dict(base)
    manifest["seed"] = cfg
    manifest["experiment"] = cfg.get("experiment", "G160-TCIA3.1")
    manifest["train_pairs"] = train_os
    manifest["n_train_pairs"] = len(train_os)
    manifest["val_pairs"] = list(base["val_pairs"])
    manifest["n_val_pairs"] = len(base["val_pairs"])
    manifest["oversample"] = os_stats
    manifest["u0106_motion"] = motion_report
    manifest["scan_tiers"] = tiers

    data_dir = T1 / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    if not (data_dir / "manifest_base.json").is_file():
        (data_dir / "manifest_base.json").write_text(json.dumps(base, indent=2) + "\n")
    out = data_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"wrote {out}", flush=True)
    print(
        f"  train pairs {os_stats['n_base_train_pairs']} → {os_stats['n_train_pairs_oversampled']} "
        f"(×{os_stats['oversample_factor_mean']:.2f})",
        flush=True,
    )
    print(
        f"  tiers: normal={os_stats['n_scans_normal']} high={os_stats['n_scans_high']} "
        f"very_high={os_stats['n_scans_very_high']} "
        f"(max≥{min_max} or p95≥{min_p95} mm; very_high max≥{very_max})",
        flush=True,
    )
    print(
        f"  repeats: high={os_stats['repeat_high']}× very_high={os_stats['repeat_very_high']}×; "
        f"01→06 extra +{os_stats['extra_0106_high']}/+{os_stats['extra_0106_very_high']}",
        flush=True,
    )
    print(
        f"  library: {motion_report['n_scans_max_ge_12']} scans with 01→06 lung max≥12 mm",
        flush=True,
    )


if __name__ == "__main__":
    main()
