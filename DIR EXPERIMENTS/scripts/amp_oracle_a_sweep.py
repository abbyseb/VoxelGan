#!/usr/bin/env python3
"""Offline amplitude oracle: grid-search scalar a on DIR TRE75 for a frozen Decoder.

For each case: û = TCIA3 synth (06→01), sweep a, TRE75(a·û). Reports best-per-case
and best global-a cohort means — upper bound for a trainable AmpHead.

  cd "DIR EXPERIMENTS"
  LEARN-GUI/.venv/bin/python scripts/amp_oracle_a_sweep.py --gpu 0 \\
    --ckpt ../PopulationStudy/ClinicalExperiments/Grid160/TCIA3/DecoderCRB/checkpoints/epoch_100.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

DIR_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIR_EXP / "scripts"))

from compare_g160_enc_dec_on_dir import (  # noqa: E402
    dvf_to_sub128,
    pull_to_elastix_zyx,
    synth_phase01,
)
from eval_a1_tre import tre_t00_t50  # noqa: E402

A1 = DIR_EXP / "arms" / "A1_oracle_dirlab" / "runs"
OUT = DIR_EXP / "arms" / "A3_synth_conditioned" / "results"
NET = DIR_EXP.parent / "PopulationStudy" / "InitialExperiments" / "Experiment6"
IM_SIZE = 64
DEFAULT_CKPT = (
    DIR_EXP.parent
    / "PopulationStudy/ClinicalExperiments/Grid160/TCIA3/DecoderCRB/checkpoints/epoch_100.pt"
)


def load_decoder(ckpt: Path, device: torch.device):
    if str(NET) not in sys.path:
        sys.path.insert(0, str(NET))
    from networks.generator_crb_dec import UNetCRBDecoder

    g = UNetCRBDecoder(im_size=IM_SIZE, n_phases=10)
    raw = torch.load(str(ckpt), map_location=device, weights_only=False)
    state = raw.get("generator", raw.get("state_dict", raw)) if isinstance(raw, dict) else raw
    g.load_state_dict(state, strict=True)
    g.to(device).eval()
    return g


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--cases", type=int, nargs="*", default=list(range(1, 11)))
    ap.add_argument(
        "--a-min", type=float, default=0.8, help="Inclusive lower bound of a sweep"
    )
    ap.add_argument("--a-max", type=float, default=2.5, help="Inclusive upper bound")
    ap.add_argument("--a-step", type=float, default=0.1)
    ap.add_argument(
        "--out",
        type=Path,
        default=OUT / "amp_oracle_a_sweep_tcia3_ep100.json",
    )
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("DIRLAB_ROOT", str(DIR_EXP / "data" / "dirlab_packs"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = args.ckpt.resolve()
    if not ckpt.is_file():
        raise SystemExit(f"missing ckpt: {ckpt}")

    a_grid = np.round(
        np.arange(args.a_min, args.a_max + 0.5 * args.a_step, args.a_step), 4
    ).tolist()
    print(
        f"device={device} ckpt={ckpt.name} a∈[{args.a_min},{args.a_max}] step={args.a_step} "
        f"({len(a_grid)} vals) cases={args.cases}",
        flush=True,
    )

    g = load_decoder(ckpt, device)
    per_case = []

    for case in args.cases:
        sid = f"DIR_C{case:02d}"
        train = A1 / sid / sid / "train"
        ct06_p = train / "CT_06.mha"
        sub06_p = train / "sub_CT_06.mha"
        if not ct06_p.is_file() or not sub06_p.is_file():
            print(f"skip {sid}: missing A1 files", flush=True)
            continue

        ct06 = sitk.GetArrayFromImage(sitk.ReadImage(str(ct06_p))).astype(np.float32)
        sub_ref = sitk.ReadImage(str(sub06_p))
        u_infer, _, _ = synth_phase01(g, ct06, device)
        base = dvf_to_sub128(pull_to_elastix_zyx(u_infer), sub_ref)  # (128,128,128,3)

        curve = []
        best_a, best_tre = None, 1e9
        for a in a_grid:
            tre = tre_t00_t50(base * float(a), case, "75", r3=True)
            t = float(tre["registered"]["mean"])
            id_ = float(tre["identity"]["mean"])
            curve.append({"a": float(a), "tre75_mm": t, "identity_mm": id_})
            if t < best_tre:
                best_tre, best_a = t, float(a)

        tre_a1 = next(c["tre75_mm"] for c in curve if abs(c["a"] - 1.0) < 1e-6)
        row = {
            "case": case,
            "scan_id": sid,
            "tre75_a1": tre_a1,
            "best_a": best_a,
            "tre75_best_a": best_tre,
            "delta_vs_a1_mm": tre_a1 - best_tre,
            "curve": curve,
        }
        per_case.append(row)
        print(
            f"{sid} a=1 → {tre_a1:.2f} mm | best a={best_a:.1f} → {best_tre:.2f} mm "
            f"(Δ {tre_a1 - best_tre:+.2f})",
            flush=True,
        )

    # Cohort: a=1, best-per-case, and best single global a
    tre_a1 = np.array([r["tre75_a1"] for r in per_case], float)
    tre_best = np.array([r["tre75_best_a"] for r in per_case], float)

    global_scores = []
    for a in a_grid:
        vals = []
        for r in per_case:
            vals.append(next(c["tre75_mm"] for c in r["curve"] if abs(c["a"] - a) < 1e-6))
        vals = np.asarray(vals, float)
        global_scores.append(
            {
                "a": float(a),
                "mean_tre75_mm": float(vals.mean()),
                "std_tre75_mm": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            }
        )
    best_global = min(global_scores, key=lambda s: s["mean_tre75_mm"])

    out = {
        "protocol": "DIR-Lab 75-pt T00→T50 R3; TCIA3 synth 06→01; û_elastix × a; offline oracle",
        "note": "Upper bound for AmpHead: best-per-case a vs best global a vs a=1",
        "when": datetime.now(timezone.utc).astimezone().isoformat(),
        "ckpt": str(ckpt),
        "a_grid": a_grid,
        "per_case": per_case,
        "cohort": {
            "a1": {
                "mean_tre75_mm": float(tre_a1.mean()),
                "std_tre75_mm": float(tre_a1.std(ddof=1)),
            },
            "best_per_case": {
                "mean_tre75_mm": float(tre_best.mean()),
                "std_tre75_mm": float(tre_best.std(ddof=1)),
                "mean_best_a": float(np.mean([r["best_a"] for r in per_case])),
            },
            "best_global_a": best_global,
            "global_a_curve": global_scores,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")

    print("\n=== cohort TRE75 (mm) ===", flush=True)
    print(
        f"  a=1            {out['cohort']['a1']['mean_tre75_mm']:.3f} ± "
        f"{out['cohort']['a1']['std_tre75_mm']:.3f}",
        flush=True,
    )
    print(
        f"  best-per-case  {out['cohort']['best_per_case']['mean_tre75_mm']:.3f} ± "
        f"{out['cohort']['best_per_case']['std_tre75_mm']:.3f}  "
        f"(mean a*={out['cohort']['best_per_case']['mean_best_a']:.2f})",
        flush=True,
    )
    print(
        f"  best global a  {best_global['mean_tre75_mm']:.3f} ± {best_global['std_tre75_mm']:.3f}  "
        f"(a={best_global['a']:.1f})",
        flush=True,
    )
    gain = out["cohort"]["a1"]["mean_tre75_mm"] - out["cohort"]["best_per_case"]["mean_tre75_mm"]
    print(f"  gain best-per-case vs a=1: {gain:+.3f} mm", flush=True)
    print(f"  ≤4 mm reachable? {'YES' if out['cohort']['best_per_case']['mean_tre75_mm'] <= 4.0 else 'NO (not with scalar a alone)'}", flush=True)
    print("Wrote", args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
