"""Point Experiment2 at Experiment1 pooled data (same split/seeds).

E2 only changes phase encoding; SPARE pair trees are identical to E1.
Creates/refreshes `data/pooled` → `../Experiment1/data/pooled` and prints
the E1 manifest summary.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

E2 = Path(__file__).resolve().parents[1]
E1 = E2.parent / "Experiment1"
POOLED = E2 / "data" / "pooled"
E1_MANIFEST = E1 / "data" / "pooled" / "manifest.json"


def main():
    if not E1_MANIFEST.is_file():
        raise SystemExit(
            f"missing {E1_MANIFEST}\n"
            "Build Experiment1 pooled data first:\n"
            "  cd ../Experiment1 && python scripts/build_pooled_dataset.py"
        )
    E2.joinpath("data").mkdir(parents=True, exist_ok=True)
    if POOLED.is_symlink() or POOLED.exists():
        if POOLED.is_symlink():
            POOLED.unlink()
        elif POOLED.is_dir():
            raise SystemExit(f"{POOLED} exists as a real directory; remove it to use E1 symlink")
    POOLED.symlink_to(os.path.relpath(E1 / "data" / "pooled", start=POOLED.parent))

    # Clinical scan symlinks (QC)
    for scan in [f"CV_P{i}_V_01" for i in range(1, 6)]:
        src = E1 / "data" / scan
        dst = E2 / "data" / scan
        if not src.is_dir():
            continue
        if dst.is_symlink() or dst.exists():
            if dst.is_symlink():
                dst.unlink()
            else:
                continue
        dst.symlink_to(os.path.relpath(src, start=dst.parent))

    man = json.loads(E1_MANIFEST.read_text())
    print(
        f"linked {POOLED} → Experiment1 pooled\n"
        f"  train {man['n_train_pairs']} + val {man['n_val_pairs']} "
        f"(patients {man['train_patients']})\n"
        f"  E2 will train with cyclic phase encoding (cond_dim=4)"
    )


if __name__ == "__main__":
    main()
