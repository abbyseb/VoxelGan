#!/usr/bin/env python3
"""Is the network's DIR error mostly a MAGNITUDE (breath-depth) error?

For each case, the network field u is multiplied by one scalar k and rescored:
  plain     k = 1
  ratio     k = mean|true motion| / mean|net motion| at the T00 landmarks
  lsq       k = argmin_k sum |k*d_net - d_true|^2 = sum(d_net.d_true)/sum(d_net.d_net)
  lsq_all   one k for all cases (the same shared scale; not per patient)
Both per-case k are ORACLE (they use the T50 landmarks) - a diagnostic, not a result.
If lsq TRE falls far below plain, the direction is right and only the size is missing.
The remaining lsq TRE is the shape error the network would still have.

  cd "DIR EXPERIMENTS"
  CUDA_VISIBLE_DEVICES=1 /home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python \
      scripts/scale_oracle_test.py --ckpt TCIA3=<.../epoch_100.pt> [--cases 1 2 ... 10]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

DIR_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIR_EXP / "scripts"))
import eval_dir_tcia3_iso2mm_v2 as v2  # noqa: E402
from dirlab_tre import landmarks_300  # noqa: E402
from rescore_oracle_iso2mm_v3 import load_ckpt  # noqa: E402

VOX = 2.0  # mm per iso voxel


def tre(pred: np.ndarray, l50: np.ndarray) -> float:
    return float(np.linalg.norm((pred - l50) * VOX, axis=1).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="LABEL=PATH")
    ap.add_argument("--cases", type=int, nargs="*", default=list(range(1, 11)))
    args = ap.parse_args()
    label, path = args.ckpt.split("=", 1)
    os.environ.setdefault("DIRLAB_ROOT", str(DIR_EXP / "data" / "dirlab_packs"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    g = load_ckpt(Path(path), device)

    rows, store = {}, {}
    for case in args.cases:
        pack = v2.pack_case(case)
        l00 = v2.official_to_iso(landmarks_300(case, "T00"), pack)
        l50 = v2.official_to_iso(landmarks_300(case, "T50"), pack)
        u = v2.predict(g, device, pack["mu"], flip=False)
        d_net = v2.sample_u(u, l00)          # landmark displacement, voxels
        d_true = l50 - l00
        k_ratio = np.linalg.norm(d_true, axis=1).mean() / max(np.linalg.norm(d_net, axis=1).mean(), 1e-6)
        k_lsq = float((d_net * d_true).sum() / max((d_net * d_net).sum(), 1e-9))
        cos = float(((d_net * d_true).sum(1) /
                     (np.linalg.norm(d_net, axis=1) * np.linalg.norm(d_true, axis=1) + 1e-9)).mean())
        rows[case] = {"identity": tre(l00, l50), "plain": tre(l00 + d_net, l50),
                      "ratio": tre(l00 + k_ratio * d_net, l50), "lsq": tre(l00 + k_lsq * d_net, l50),
                      "k_ratio": float(k_ratio), "k_lsq": k_lsq, "mean_cos": cos}
        store[case] = (d_net, d_true, l00, l50)
        r = rows[case]
        print(f"C{case:02d} id {r['identity']:.2f}  plain {r['plain']:.2f}  ratio {r['ratio']:.2f} "
              f"(k {k_ratio:.2f})  lsq {r['lsq']:.2f} (k {k_lsq:.2f})  cos {cos:.2f}", flush=True)

    num = sum((s[0] * s[1]).sum() for s in store.values())
    den = sum((s[0] * s[0]).sum() for s in store.values())
    k_all = float(num / den)
    for case, (d_net, _, l00, l50) in store.items():
        rows[case]["lsq_all"] = tre(l00 + k_all * d_net, l50)

    print(f"\n===== {label}: TRE300 mean over {len(rows)} cases (mm) =====")
    for key in ("identity", "plain", "lsq_all", "ratio", "lsq"):
        print(f"{key:9s} {np.mean([r[key] for r in rows.values()]):.3f}")
    print(f"shared k (lsq_all) = {k_all:.2f}")
    print("per-case k_lsq:", " ".join(f"C{c:02d}={r['k_lsq']:.2f}" for c, r in rows.items()))

    out = v2.OUT_DIR / f"scale_oracle_{label}_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({"ckpt": path, "k_all": k_all, "rows": rows}, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
