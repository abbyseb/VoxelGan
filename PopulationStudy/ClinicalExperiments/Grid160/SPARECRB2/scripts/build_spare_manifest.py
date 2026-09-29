#!/usr/bin/env python3
"""Symlink SPARE data_iso into one pooled folder and write a patient holdout.

Does not copy volumes and does not touch TCIA data.
"""

from __future__ import annotations

import json
from pathlib import Path

VS = Path(__file__).resolve().parents[1]


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def main() -> None:
    cfg = json.loads((VS / "seed.json").read_text())
    root = Path(cfg["spare_root"])
    holdout = set(cfg["holdout_scans"])
    val = set(cfg["val_scans"])
    if holdout & val:
        raise ValueError(f"patient in both holdout and val: {holdout & val}")
    pooled = VS / "data" / "pooled"
    pooled.mkdir(parents=True, exist_ok=True)

    scans = sorted(
        [p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("P")],
        key=lambda n: int(n[1:]),
    )
    train_pairs, val_pairs, test_pairs = [], [], []
    for sid in scans:
        src = root / sid / "all"
        if not (src / "CT_01.npy").is_file():
            raise FileNotFoundError(src / "CT_01.npy")
        link(src / "Mask_Lung.npy", pooled / f"{sid}_Mask_Lung.npy")
        for ph in range(1, 11):
            link(src / f"CT_{ph:02d}.npy", pooled / f"{sid}_CT_{ph:02d}.npy")
        for ref in range(1, 11):
            for tgt in range(1, 11):
                name = f"{sid}_{ref:02d}_to_{tgt:02d}_pair.npy"
                link(src / f"{ref:02d}_to_{tgt:02d}_pair.npy", pooled / name)
                (test_pairs if sid in holdout else val_pairs if sid in val else train_pairs).append(name)
        print(f"linked {sid}", flush=True)

    train_pairs.sort()
    val_pairs.sort()
    test_pairs.sort()
    train_scans = [s for s in scans if s not in holdout and s not in val]
    manifest = {
        "seed": cfg,
        "grid": "SPARE data_iso 2mm 160³ full volume",
        "pooled_train_dir": str(pooled),
        "n_train_pairs": len(train_pairs),
        "n_val_pairs": len(val_pairs),
        "n_test_pairs": len(test_pairs),
        "train_scans": train_scans,
        "train_patients": train_scans,
        "val_scans": [s for s in scans if s in val],
        "val_patients": [s for s in scans if s in val],
        "holdout_scans": [s for s in scans if s in holdout],
        "holdout_patients": [s for s in scans if s in holdout],
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "test_pairs": test_pairs,
        "split": (
            "train "
            + ", ".join(train_scans)
            + " | val "
            + ", ".join(s for s in scans if s in val)
            + " | test "
            + ", ".join(s for s in scans if s in holdout)
        ),
    }
    out = VS / "data" / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"wrote {out}\n"
        f"  train {len(train_pairs)} pairs / {len(train_scans)} scans\n"
        f"  val   {len(val_pairs)} pairs / {manifest['val_scans']}\n"
        f"  test  {len(test_pairs)} pairs / {manifest['holdout_scans']} (not used in training)"
    )


if __name__ == "__main__":
    main()
