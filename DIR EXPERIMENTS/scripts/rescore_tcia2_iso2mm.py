#!/usr/bin/env python3
"""Re-score TCIA2 (UNetCRBDecoder, MSE, TCIA, FOV aug) on the corrected DIR path.

Same data and augmentation as TCIA3, loss MSE instead of MAE, so:
  TCIA2 vs SPARE_A1_dec  -> effect of training data (same loss, same aug)
  TCIA2 vs TCIA3         -> effect of the loss (same data)
Reuses the verified v2 geometry and the helpers in rescore_spare_iso2mm.py.
TCIA models use the TCIA frame directly (no layout change). Weights only read.

  cd "DIR EXPERIMENTS"
  CUDA_VISIBLE_DEVICES=1 python scripts/rescore_tcia2_iso2mm.py
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
from rescore_spare_iso2mm import load_model, predict_model_layout  # noqa: E402

G160 = v2.VOXEL / "PopulationStudy/ClinicalExperiments/Grid160"
MODELS = [
    ("TCIA2_ep100", G160 / "TCIA2/DecoderCRB/checkpoints/epoch_100.pt"),
    ("TCIA2_bestval_weights", G160 / "TCIA2/DecoderCRB/weights/crb_dec_mse_iso_g160_tcia_r3_a1_generator.pth"),
    ("TCIA3_ep100", G160 / "TCIA3/DecoderCRB/checkpoints/epoch_100.pt"),  # reference, 4.25
]
IDENT = ((0, 1, 2), (0, 0, 0))


def main() -> None:
    os.environ.setdefault("DIRLAB_ROOT", str(DIR_EXP / "data" / "dirlab_packs"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cases = list(range(1, 11))
    packs = {c: v2.pack_case(c) for c in cases}
    rows = {}
    for label, path in MODELS:
        if not path.is_file():
            print(f"skip {label}: missing {path}")
            continue
        g = load_model(path, "dec", device)
        for mirror in (False, True):
            per75, per300 = {}, {}
            for c in cases:
                u = predict_model_layout(g, device, packs[c]["mu"], *IDENT, mirror)
                per75[str(c)] = v2.score_set(u, c, "75", packs[c])["registered"]["mean"]
                per300[str(c)] = v2.score_set(u, c, "300", packs[c])["registered"]["mean"]
            key = f"{label}|{'mirror' if mirror else 'plain'}"
            rows[key] = {"tre75": float(np.mean(list(per75.values()))),
                         "tre300": float(np.mean(list(per300.values()))),
                         "per_case_tre300": per300, "ckpt": str(path)}
            print(f"{key:32s} TRE75 {rows[key]['tre75']:.3f}  TRE300 {rows[key]['tre300']:.3f}  "
                  f"C08 {per300['8']:.2f}", flush=True)
        del g
        torch.cuda.empty_cache()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = v2.OUT_DIR / f"rescore_tcia2_iso2mm_{stamp}.json"
    out.write_text(json.dumps({"when": datetime.now(timezone.utc).astimezone().isoformat(),
                               "rows": rows}, indent=2) + "\n")
    print("\n============ TCIA2 on corrected DIR (mm) ============")
    for k in sorted(rows, key=lambda k: rows[k]["tre300"]):
        r = rows[k]
        print(f"{k:32s} TRE75 {r['tre75']:.3f}  TRE300 {r['tre300']:.3f}  C08 {r['per_case_tre300']['8']:.2f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
