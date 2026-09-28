#!/usr/bin/env python3
"""Synth-oracle TRE75 (R3, CT_06→01, Elastix DVF) for a G160 Decoder checkpoint.

  cd "DIR EXPERIMENTS"
  CUDA_VISIBLE_DEVICES=0 LEARN-GUI/.venv/bin/python \\
    ../PopulationStudy/ClinicalExperiments/Grid160/TCIA3/scripts/qc_synth_oracle_tre75.py \\
    --gpu 0 --tag tcia3_preemptive \\
    --ckpt ../PopulationStudy/ClinicalExperiments/Grid160/TCIA3/DecoderCRB/weights/crb_dec_mae_iso_g160_tcia_r3_a1_generator.pth
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

DIR_EXP = Path("/home/abhishek/Voxel_GAN/DIR EXPERIMENTS")
sys.path.insert(0, str(DIR_EXP / "scripts"))

from compare_g160_enc_dec_on_dir import (  # noqa: E402
    cos_mot,
    dvf_to_sub128,
    pearson_ncc,
    pull_to_elastix_zyx,
    synth_phase01,
)
from eval_a1_tre import load_dvf_zyx3, tre_t00_t50  # noqa: E402

T3 = Path(__file__).resolve().parents[1]
E1 = T3.parent / "Experiment1"
NET = DIR_EXP.parent / "PopulationStudy" / "InitialExperiments" / "Experiment6"
A1 = DIR_EXP / "arms" / "A1_oracle_dirlab" / "runs"
OUT = T3 / "DecoderCRB" / "plots" / "qc_dir_oracle"
IM_SIZE = 64
SPARE = E1 / "DecoderCRB/weights/crb_dec_mse_iso_g160_fov_full_generator.pth"
TCIA2 = (
    T3.parent / "TCIA2" / "DecoderCRB/weights/crb_dec_mse_iso_g160_tcia_r3_a1_generator.pth"
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


def eval_model(name: str, ckpt: Path, device: torch.device, cases: list[int]) -> dict:
    g = load_decoder(ckpt, device)
    rows = []
    vals = []
    for case in cases:
        sid = f"DIR_C{case:02d}"
        train = A1 / sid / sid / "train"
        ct06_p, ct01_p, sub06_p = train / "CT_06.mha", train / "CT_01.mha", train / "sub_CT_06.mha"
        el_p = A1 / sid / "ModelTraining" / "train" / sid / "DVFs" / "DVF_01_mha.npy"
        ct06 = sitk.GetArrayFromImage(sitk.ReadImage(str(ct06_p))).astype(np.float32)
        ct01 = sitk.GetArrayFromImage(sitk.ReadImage(str(ct01_p))).astype(np.float32)
        sub_ref = sitk.ReadImage(str(sub06_p))
        el_c = np.moveaxis(load_dvf_zyx3(el_p), -1, 0)
        u_infer, _, warped = synth_phase01(g, ct06, device)
        dvf_sub = dvf_to_sub128(pull_to_elastix_zyx(u_infer), sub_ref)
        tre75 = tre_t00_t50(dvf_sub, case, "75", r3=True)
        tre300 = tre_t00_t50(dvf_sub, case, "300", r3=True)
        pred_c = np.moveaxis(dvf_sub, -1, 0)
        sl = tuple(slice(0, min(a, b)) for a, b in zip(warped.shape, ct01.shape))
        ncc, pearson = pearson_ncc(warped[sl], ct01[sl])
        m = {
            "tre75_mm": tre75["registered"]["mean"],
            "tre300_mm": tre300["registered"]["mean"],
            "identity_mm": tre75["identity"]["mean"],
            "delta_id_mm": tre75["improvement_mm"],
            "cos_vs_elastix": cos_mot(pred_c, el_c),
            "ct_ncc_vs_real01": ncc,
        }
        rows.append({"case": case, **m})
        vals.append(m["tre75_mm"])
        print(
            f"{sid} {name:22s} TRE75={m['tre75_mm']:.2f} TRE300={m['tre300_mm']:.2f} "
            f"Δid={m['delta_id_mm']:+.2f} cos={m['cos_vs_elastix']:.3f}",
            flush=True,
        )
    a = np.asarray(vals, float)
    a300 = np.asarray([r["tre300_mm"] for r in rows], float)
    return {
        "ckpt": str(ckpt),
        "per_case": rows,
        "mean_tre75_mm": float(a.mean()),
        "std_tre75_mm": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "mean_tre300_mm": float(a300.mean()),
        "std_tre300_mm": float(a300.std(ddof=1)) if len(a300) > 1 else 0.0,
        "n": len(a),
        "values": vals,
        "values_tre300": a300.tolist(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--note", default="preemptive synth oracle TRE75")
    ap.add_argument("--cases", type=int, nargs="*", default=list(range(1, 11)))
    ap.add_argument("--out", type=Path, default=OUT / "tre75_synth_oracle_preemptive.json")
    ap.add_argument("--skip-refs", action="store_true", help="Skip SPARE/TCIA2 reference oracles")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("DIRLAB_ROOT", str(DIR_EXP / "data" / "dirlab_packs"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = args.ckpt if args.ckpt.is_absolute() else (T3 / args.ckpt).resolve()
    if not ckpt.is_file():
        raise SystemExit(f"missing ckpt: {ckpt}")

    print(f"device {device} | tag={args.tag} | ckpt={ckpt.name}", flush=True)
    primary = eval_model(args.tag, ckpt, device, args.cases)

    out = {
        "protocol": "DIR-Lab 75+300-pt T00→T50 R3; synth from CT_06; pull→elastix sub128",
        "note": args.note,
        "when": datetime.now(timezone.utc).astimezone().isoformat(),
        "summary": {args.tag: primary},
        "per_model": {args.tag: primary["per_case"]},
    }
    if not args.skip_refs:
        for ref_name, ref_ckpt in [("spare_g160_a1", SPARE), ("tcia2_best_final", TCIA2)]:
            if ref_ckpt.is_file() and ref_name not in out["summary"]:
                print(f"\n--- reference {ref_name} ---", flush=True)
                out["summary"][ref_name] = eval_model(ref_name, ref_ckpt, device, args.cases)

    OUT.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print("\n=== mean TRE (mm) ===", flush=True)
    for k, s in out["summary"].items():
        print(
            f"  {k:22s} TRE75={s['mean_tre75_mm']:.3f}±{s['std_tre75_mm']:.3f}  "
            f"TRE300={s.get('mean_tre300_mm', float('nan')):.3f}±{s.get('std_tre300_mm', float('nan')):.3f}  "
            f"(n={s['n']})",
            flush=True,
        )
    print("Wrote", args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
