#!/usr/bin/env python3
"""Write TCIA4/data/manifest.json with a patient holdout.

Reads the existing pooled pair list. Does not rebuild or edit the pooled
tree, and does not touch TCIA2 / TCIA3 manifests.
"""

from __future__ import annotations

import json
from pathlib import Path

T4 = Path(__file__).resolve().parents[1]
SOURCE = T4.parent / "TCIA3" / "data" / "manifest.json"


def scan_tag(scan_id: str) -> str:
    return f"S{int(scan_id[1:]):02d}"


def main() -> None:
    cfg = json.loads((T4 / "seed.json").read_text())
    src = json.loads(SOURCE.read_text())
    holdout = list(cfg["holdout_scans"])
    hold_tags = {scan_tag(s) for s in holdout}
    all_pairs = list(src["train_pairs"]) + list(src["val_pairs"])
    train_pairs, val_pairs = [], []
    for name in all_pairs:
        tag = name.split("_", 1)[0]
        (val_pairs if tag in hold_tags else train_pairs).append(name)
    train_pairs.sort()
    val_pairs.sort()
    scans = list(src["train_scans"])
    train_scans = [s for s in scans if s not in set(holdout)]
    manifest = {
        "seed": cfg,
        "grid": "TCIA packed_r3_hu 2mm 160³ full volume (R3; Elastix=HU, train=µ)",
        "packed_root": src["packed_root"],
        "pooled_train_dir": src["pooled_train_dir"],
        "n_train_pairs": len(train_pairs),
        "n_val_pairs": len(val_pairs),
        "n_test_pairs": 0,
        "train_scans": train_scans,
        "train_patients": [scan_tag(s) for s in train_scans],
        "holdout_scans": holdout,
        "holdout_patients": [scan_tag(s) for s in holdout],
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "test_pairs": [],
        "matching_rule": src["matching_rule"],
        "split": "patient holdout; val pairs are unseen scans, not a same-patient pair split",
    }
    out_dir = T4 / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"wrote {out}\n"
        f"  train {len(train_pairs)} pairs / {len(train_scans)} scans\n"
        f"  val   {len(val_pairs)} pairs / {len(holdout)} holdout scans: {holdout}"
    )


if __name__ == "__main__":
    main()
