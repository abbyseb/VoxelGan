#!/usr/bin/env python3
"""Left-right mirror averaging (test-time augmentation) on the verified 2 mm DIR path.

For each model:  A = net(ct);  B = net(mirror_x(ct)) -> mirror back, negate dx;
                 u = (A + B) / 2.   Also scores an ensemble of non-fine-tuned runs.

No training, weights only read. Uses eval_dir_tcia3_iso2mm_v2 geometry (round-trip verified).

  cd "DIR EXPERIMENTS"
  CUDA_VISIBLE_DEVICES=1 python scripts/mirror_tta_iso2mm.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

DIR_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIR_EXP / "scripts"))
import eval_dir_tcia3_iso2mm_v2 as v2  # noqa: E402
from rescore_oracle_iso2mm_v3 import load_ckpt  # noqa: E402

G160 = v2.VOXEL / "PopulationStudy/ClinicalExperiments/Grid160"
MODELS = {
    "TCIA3_ep100": G160 / "TCIA3/DecoderCRB/checkpoints/epoch_100.pt",
    "TCIA3.5_ep92": G160 / "TCIA3.5/DecoderCRB/checkpoints/epoch_092.pt",
    "MagFT_ep30": G160 / "TCIA3_magFT/DecoderCRB/checkpoints/epoch_030.pt",  # reference only
}
ENSEMBLE = ["TCIA3_ep100", "TCIA3.5_ep92"]  # base models only, no fine-tunes


def predict_mirror(g, device, mu):
    a = v2.predict(g, device, mu, flip=False)
    b = v2.predict(g, device, np.ascontiguousarray(mu[:, :, ::-1]), flip=False)
    b = np.ascontiguousarray(b[:, :, ::-1, :])
    b[..., 0] *= -1.0  # left-right component changes sign
    return a, 0.5 * (a + b)


def tre(u, case, pack):
    return (
        v2.score_set(u, case, "75", pack)["registered"]["mean"],
        v2.score_set(u, case, "300", pack)["registered"]["mean"],
    )


def main() -> None:
    import argparse
    global MODELS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", action="append", default=[],
                    help="LABEL=PATH, repeatable; replaces the built-in model list")
    args = ap.parse_args()
    if args.ckpt:
        MODELS = {c.split("=", 1)[0]: Path(c.split("=", 1)[1]) for c in args.ckpt}
    os.environ.setdefault("DIRLAB_ROOT", str(DIR_EXP / "data" / "dirlab_packs"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cases = list(range(1, 11))
    packs = {c: v2.pack_case(c) for c in cases}

    fields = {}  # (model, variant) -> {case: u}
    for name, path in MODELS.items():
        if not path.is_file():
            print(f"skip {name}: missing {path}")
            continue
        g = load_ckpt(path, device)
        plain, mirr = {}, {}
        for c in cases:
            plain[c], mirr[c] = predict_mirror(g, device, packs[c]["mu"])
        fields[(name, "plain")] = plain
        fields[(name, "mirror")] = mirr
        del g
        torch.cuda.empty_cache()

    ens = [m for m in ENSEMBLE if (m, "plain") in fields]
    if len(ens) >= 2:
        for variant in ("plain", "mirror"):
            fields[("ENS_" + "+".join(ens), variant)] = {
                c: np.mean([fields[(m, variant)][c] for m in ens], axis=0) for c in cases
            }

    rows = {}
    for (name, variant), f in fields.items():
        per = {c: tre(f[c], c, packs[c]) for c in cases}
        rows[f"{name}|{variant}"] = {
            "tre75": float(np.mean([v[0] for v in per.values()])),
            "tre300": float(np.mean([v[1] for v in per.values()])),
            "per_case_tre300": {str(c): v[1] for c, v in per.items()},
        }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = v2.OUT_DIR / f"mirror_tta_iso2mm_{stamp}.json"
    out.write_text(json.dumps({"when": datetime.now(timezone.utc).astimezone().isoformat(),
                               "rows": rows}, indent=2) + "\n")

    print("\n============ MIRROR TTA (mm) ============")
    print(f"{'model':32s} {'variant':7s} TRE75  TRE300  C08")
    for k in sorted(rows, key=lambda k: rows[k]["tre300"]):
        name, variant = k.split("|")
        r = rows[k]
        print(f"{name:32s} {variant:7s} {r['tre75']:.3f}  {r['tre300']:.3f}  {r['per_case_tre300']['8']:.2f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
