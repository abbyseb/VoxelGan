#!/usr/bin/env python3
"""Two tests of whether TCIA3 is a real population synthesiser (verified v2 DIR path).

Test 1  "Does it use the anatomy?"
        For each DIR case, feed the network a DIFFERENT patient's exhale CT
        (a) each of the other 9 DIR cases, averaged, and (b) TCIA S1 phase 06,
        then score the predicted field against the real case's landmarks.
        If TRE hardly changes vs the correct CT, the network ignores anatomy.

Test 2  "Is it better than an average-motion template?"
        Template = mean of the 82 TCIA Elastix fields for the same pair the network
        is asked (phase 06 -> 01), each on its own lung-centred 2 mm 160^3 cube.
        Apply that one fixed field to every DIR case and score it.

All cubes are lung-centred 2 mm 160^3, so fields line up at the lung centre.
Weights and data are only read.

  cd "DIR EXPERIMENTS"
  CUDA_VISIBLE_DEVICES=1 python scripts/anatomy_tests_iso2mm.py
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

POOLED = Path("/media/abhishek/3CCA3CADCA3C6574/TCIA_4D-Lung/synth_g160_r3/pooled/train")
TCIA_S1_CT06 = Path("/media/abhishek/3CCA3CADCA3C6574/TCIA_4D-Lung/packed_r3_hu/S1/all/CT_06_mu.npy")
PAIR = "06_to_01"  # network: ref phase 06 (index 5) -> target 01 (index 0)


def tre300(u, c, pack):
    return v2.score_set(u, c, "300", pack)["registered"]["mean"]


def tre75(u, c, pack):
    return v2.score_set(u, c, "75", pack)["registered"]["mean"]


def build_template() -> tuple[np.ndarray, int]:
    files = sorted(POOLED.glob(f"S*_{PAIR}_pair.npy"))
    if not files:
        raise SystemExit(f"no {PAIR} pair files under {POOLED}")
    acc = None
    for f in files:
        d = np.load(f, mmap_mode="r")
        d = np.moveaxis(np.asarray(d), 0, -1) if d.shape[0] == 3 else np.asarray(d)  # -> (Z,Y,X,3)
        d = d.astype(np.float64)
        acc = d if acc is None else acc + d
    return (acc / len(files)).astype(np.float32), len(files)


def main() -> None:
    os.environ.setdefault("DIRLAB_ROOT", str(DIR_EXP / "data" / "dirlab_packs"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cases = list(range(1, 11))
    packs = {c: v2.pack_case(c) for c in cases}
    g = v2.load_generator(device)  # TCIA3 epoch 100

    # predictions from every DIR CT once
    fields = {c: v2.predict(g, device, packs[c]["mu"], flip=False) for c in cases}
    tcia_mu = np.load(TCIA_S1_CT06).astype(np.float32)
    field_tcia = v2.predict(g, device, tcia_mu, flip=False)

    rows = {}
    # correct anatomy
    rows["own_CT"] = {c: (tre75(fields[c], c, packs[c]), tre300(fields[c], c, packs[c])) for c in cases}
    # Test 1a: other DIR patients' CT (mean over the 9 others)
    other = {}
    for c in cases:
        vals75, vals300 = [], []
        for o in cases:
            if o == c:
                continue
            vals75.append(tre75(fields[o], c, packs[c]))
            vals300.append(tre300(fields[o], c, packs[c]))
        other[c] = (float(np.mean(vals75)), float(np.mean(vals300)))
    rows["other_DIR_CT_mean9"] = other
    # Test 1b: TCIA S1 CT
    rows["TCIA_S1_CT"] = {c: (tre75(field_tcia, c, packs[c]), tre300(field_tcia, c, packs[c])) for c in cases}
    # Test 2: average TCIA template
    tmpl, n = build_template()
    print(f"template = mean of {n} TCIA {PAIR} fields", flush=True)
    rows["TCIA_mean_template"] = {c: (tre75(tmpl, c, packs[c]), tre300(tmpl, c, packs[c])) for c in cases}
    # identity for reference
    zero = np.zeros_like(tmpl)
    rows["do_nothing"] = {c: (tre75(zero, c, packs[c]), tre300(zero, c, packs[c])) for c in cases}

    summary = {}
    for k, per in rows.items():
        summary[k] = {
            "tre75": float(np.mean([v[0] for v in per.values()])),
            "tre300": float(np.mean([v[1] for v in per.values()])),
            "per_case_tre300": {str(c): v[1] for c, v in per.items()},
        }
    own = summary["own_CT"]["per_case_tre300"]
    for k in summary:
        s = summary[k]["per_case_tre300"]
        summary[k]["own_CT_better_in_cases"] = int(sum(own[c] < s[c] for c in own)) if k != "own_CT" else None

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = v2.OUT_DIR / f"anatomy_tests_iso2mm_{stamp}.json"
    out.write_text(json.dumps({"when": datetime.now(timezone.utc).astimezone().isoformat(),
                               "model": str(v2.CKPT), "summary": summary}, indent=2) + "\n")

    print("\n============ ANATOMY TESTS, TCIA3 ep100 (mm) ============")
    print(f"{'input / field':24s} TRE75  TRE300  own-CT better in")
    for k in ("own_CT", "other_DIR_CT_mean9", "TCIA_S1_CT", "TCIA_mean_template", "do_nothing"):
        s = summary[k]
        wins = "" if s["own_CT_better_in_cases"] is None else f"{s['own_CT_better_in_cases']}/10 cases"
        print(f"{k:24s} {s['tre75']:.3f}  {s['tre300']:.3f}  {wins}")
    print("\nper case TRE300:")
    print("case " + " ".join(f"{k[:10]:>11s}" for k in summary))
    for c in cases:
        print(f"C{c:02d}  " + " ".join(f"{summary[k]['per_case_tre300'][str(c)]:11.2f}" for k in summary))
    print("wrote", out)


if __name__ == "__main__":
    main()
