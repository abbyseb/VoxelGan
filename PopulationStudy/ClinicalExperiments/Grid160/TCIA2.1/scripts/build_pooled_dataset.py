#!/usr/bin/env python3
"""Build patient-prefixed pooled symlink tree for TCIA packed_iso → G160 train.

Layout (matches SPARE / Iso-E2 convention so FOVAugPhasePairDataset works):

  S01_CT_01.npy … S01_CT_10.npy
  S01_Mask_Lung.npy
  S01_01_to_02_pair.npy

  cd …/Grid160/TCIA1
  python scripts/build_pooled_dataset.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

T1 = Path(__file__).resolve().parents[1]
DEFAULT_PACKED = Path("/media/abhishek/3CCA3CADCA3C6574/TCIA_4D-Lung/packed_r3_hu")
DEFAULT_POOLED = Path("/media/abhishek/3CCA3CADCA3C6574/TCIA_4D-Lung/synth_g160_r3/pooled")


def scan_tag(scan_id: str) -> str:
    """S1 → S01 (zero-padded so lexical sort matches numeric order)."""
    n = int(scan_id[1:])
    return f"S{n:02d}"


def abs_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def link_scan(scan_id: str, src_root: Path, dest_dir: Path) -> list[str]:
    tag = scan_tag(scan_id)
    src_dir = src_root / scan_id / "all"
    if not src_dir.is_dir():
        raise FileNotFoundError(src_dir)
    pair_names: list[str] = []
    abs_symlink(src_dir / "Mask_Lung.npy", dest_dir / f"{tag}_Mask_Lung.npy")
    for p in range(1, 11):
        # Prefer µ CTs for training (A3/G160 style); fall back to HU
        mu = src_dir / f"CT_{p:02d}_mu.npy"
        hu = src_dir / f"CT_{p:02d}.npy"
        abs_symlink(mu if mu.is_file() else hu, dest_dir / f"{tag}_CT_{p:02d}.npy")
    for ref in range(1, 11):
        for tgt in range(1, 11):
            src = src_dir / f"{ref:02d}_to_{tgt:02d}_pair.npy"
            if not src.is_file():
                raise FileNotFoundError(src)
            name = f"{tag}_{ref:02d}_to_{tgt:02d}_pair.npy"
            abs_symlink(src, dest_dir / name)
            pair_names.append(name)
    return pair_names


def split_val(train_by_scan: dict[str, list[str]], seed: int, frac: float):
    val, train = [], []
    for i, sid in enumerate(sorted(train_by_scan.keys(), key=lambda x: int(x[1:]))):
        rng = random.Random(seed + i)
        names = list(train_by_scan[sid])
        rng.shuffle(names)
        n_val = max(1, int(round(len(names) * frac)))
        val.extend(names[:n_val])
        train.extend(names[n_val:])
    return sorted(train), sorted(val)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packed", type=Path, default=DEFAULT_PACKED)
    ap.add_argument("--pooled", type=Path, default=DEFAULT_POOLED)
    ap.add_argument("--seed-json", type=Path, default=T1 / "seed.json")
    args = ap.parse_args()

    cfg = json.loads(args.seed_json.read_text())
    seed = int(cfg["val_split_seed"])
    frac = float(cfg["val_fraction"])

    scans = sorted(
        [p.name for p in args.packed.iterdir() if p.is_dir() and p.name.startswith("S")],
        key=lambda n: int(n[1:]),
    )
    if len(scans) != 82:
        raise SystemExit(f"expected 82 scans under {args.packed}, got {len(scans)}")

    train_dir = args.pooled / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    train_by_scan: dict[str, list[str]] = {}
    for sid in scans:
        train_by_scan[sid] = link_scan(sid, args.packed, train_dir)
        print(f"linked {sid} → {scan_tag(sid)} ({len(train_by_scan[sid])} pairs)", flush=True)

    train_pairs, val_pairs = split_val(train_by_scan, seed, frac)

    manifest = {
        "seed": cfg,
        "grid": "TCIA packed_r3_hu 2mm 160³ (R3; Elastix=HU, train=µ)",
        "packed_root": str(args.packed),
        "pooled_train_dir": str(train_dir),
        "n_train_pairs": len(train_pairs),
        "n_val_pairs": len(val_pairs),
        "n_test_pairs": 0,
        "train_scans": scans,
        "train_patients": [scan_tag(s) for s in scans],  # dataset uses this key name
        "holdout_patients": [],
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "test_pairs": [],
        "matching_rule": (
            "Sxx_rr_to_tt_pair.npy loads only Sxx_CT_rr.npy, Sxx_CT_tt.npy, "
            "Sxx_Mask_Lung.npy."
        ),
    }
    out = args.pooled / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    # also mirror under TCIA1/data for convenience
    local = T1 / "data"
    local.mkdir(parents=True, exist_ok=True)
    (local / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"wrote {out}\n"
        f"  train {len(train_pairs)} + val {len(val_pairs)} "
        f"(of {sum(len(v) for v in train_by_scan.values())} pairs, {len(scans)} scans)"
    )


if __name__ == "__main__":
    main()
