"""Build patient-prefixed pooled train/test symlink trees for IsoExperiments E2.

Uses PopulationStudy/data_iso (P0-A common isotropic grid, 2 mm, 160³).

  python scripts/build_pooled_dataset.py           # E1 split (hold-out P3–P5)
  python scripts/build_pooled_dataset.py --full    # all P1–P9
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

E2 = Path(__file__).resolve().parents[1]
PS = E2.parents[1]  # PopulationStudy
SRC = PS / "data_iso"


def patient_tag(pid: str) -> str:
    return f"P{int(pid[1:]):02d}"


def abs_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def link_patient(pid: str, dest_dir: Path) -> list[str]:
    tag = patient_tag(pid)
    src_dir = SRC / pid / "all"
    pair_names = []
    abs_symlink(src_dir / "Mask_Lung.npy", dest_dir / f"{tag}_Mask_Lung.npy")
    for p in range(1, 11):
        abs_symlink(src_dir / f"CT_{p:02d}.npy", dest_dir / f"{tag}_CT_{p:02d}.npy")
    for ref in range(1, 11):
        for tgt in range(1, 11):
            src = src_dir / f"{ref:02d}_to_{tgt:02d}_pair.npy"
            name = f"{tag}_{ref:02d}_to_{tgt:02d}_pair.npy"
            abs_symlink(src, dest_dir / name)
            pair_names.append(name)
    return pair_names


def split_val(train_by_patient: dict[str, list[str]], seed: int, frac: float):
    val, train = [], []
    for i, pid in enumerate(sorted(train_by_patient.keys(), key=lambda x: int(x[1:]))):
        rng = random.Random(seed + i)
        names = list(train_by_patient[pid])
        rng.shuffle(names)
        n_val = max(1, int(round(len(names) * frac)))
        val.extend(names[:n_val])
        train.extend(names[n_val:])
    return sorted(train), sorted(val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--full",
        action="store_true",
        help="train all P1–P9 (writes data/pooled_full/)",
    )
    args = ap.parse_args()

    seed_path = E2 / ("seed_full.json" if args.full else "seed.json")
    pooled = E2 / "data" / ("pooled_full" if args.full else "pooled")
    cfg = json.loads(seed_path.read_text())
    train_pids = cfg["train_patients"]
    hold_pids = cfg["holdout_patients"]
    seed = int(cfg["val_split_seed"])
    frac = float(cfg["val_fraction"])

    train_dir = pooled / "train"
    test_dir = pooled / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    train_by_patient = {}
    for pid in train_pids:
        train_by_patient[pid] = link_patient(pid, train_dir)

    test_pairs = []
    for pid in hold_pids:
        test_pairs.extend(link_patient(pid, test_dir))

    train_pairs, val_pairs = split_val(train_by_patient, seed, frac)

    manifest = {
        "seed": cfg,
        "grid": "data_iso 2mm 160³",
        "pooled_train_dir": str(train_dir),
        "pooled_test_dir": str(test_dir),
        "n_train_pairs": len(train_pairs),
        "n_val_pairs": len(val_pairs),
        "n_test_pairs": len(test_pairs),
        "train_patients": train_pids,
        "holdout_patients": hold_pids,
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "test_pairs": sorted(test_pairs),
        "matching_rule": (
            "Pxx_rr_to_tt_pair.npy loads only Pxx_CT_rr.npy, Pxx_CT_tt.npy, "
            "Pxx_Mask_Lung.npy. DataLoader shuffles indices; __getitem__ never "
            "crosses patients."
        ),
    }
    out = pooled / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"wrote {out}\n"
        f"  SRC={SRC}\n"
        f"  train patients {train_pids}: {len(train_pairs)} train + {len(val_pairs)} val "
        f"(of {sum(len(v) for v in train_by_patient.values())} pooled)\n"
        f"  hold-out {hold_pids}: {len(test_pairs)} test pairs\n"
        f"  identity included"
    )


if __name__ == "__main__":
    main()
