#!/usr/bin/env python3
"""DIR TRE75: TCIA3 ep100 ± FeatAmpHead (mid-CT + |û| → a).

  cd "DIR EXPERIMENTS"
  LEARN-GUI/.venv/bin/python scripts/eval_amp_head_feat_dir_tre.py --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

DIR_EXP = Path(__file__).resolve().parents[1]
ROOT = DIR_EXP.parent
sys.path.insert(0, str(DIR_EXP / "scripts"))
sys.path.insert(0, str(ROOT / "PopulationStudy/ClinicalExperiments/Grid160/TCIA3/scripts"))

from amp_oracle_a_sweep import DEFAULT_CKPT, load_decoder  # noqa: E402
from compare_g160_enc_dec_on_dir import (  # noqa: E402
    dvf_to_sub128,
    pull_to_elastix_zyx,
    synth_phase01,
)
from eval_a1_tre import tre_t00_t50  # noqa: E402
from train_amp_head_feat import (  # noqa: E402
    A_MAX,
    A_MIN,
    FeatAmpHead,
    feature_vector,
)

A1 = DIR_EXP / "arms" / "A1_oracle_dirlab" / "runs"
FEAT_CKPT = (
    ROOT
    / "PopulationStudy/ClinicalExperiments/Grid160/TCIA3/AmpHeadFeat/amp_head_feat_best.pt"
)
OUT = DIR_EXP / "arms" / "A3_synth_conditioned" / "results"


def lung_mask_from_ct(ct: np.ndarray) -> torch.Tensor:
    """Rough lung mask on µ/HU-ish volume for feature stats (DIR CT_06 is HU)."""
    # DIR packs are HU; Decoder path converts elsewhere — for mask, HU air/lung:
    m = (ct < -200) & (ct > -1000)
    return torch.from_numpy(m.astype(np.float32))[None, None]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dec-ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--feat-ckpt", type=Path, default=FEAT_CKPT)
    ap.add_argument("--cases", type=int, nargs="*", default=list(range(1, 11)))
    ap.add_argument(
        "--out",
        type=Path,
        default=OUT / "amp_head_feat_dir_tre75_tcia3_ep100.json",
    )
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("DIRLAB_ROOT", str(DIR_EXP / "data" / "dirlab_packs"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not args.feat_ckpt.is_file():
        raise SystemExit(f"missing FeatAmpHead ckpt: {args.feat_ckpt}")

    raw = torch.load(str(args.feat_ckpt), map_location=device, weights_only=False)
    head = FeatAmpHead(hidden=int(raw.get("hidden", 64))).to(device)
    head.load_state_dict(raw["amp_head"])
    head.eval()
    g = load_decoder(args.dec_ckpt.resolve(), device)

    # Need raw Decoder û in network space for features — synth_phase01 returns
    # post-processed DVF. Replicate feature extraction on sub128 elastix û for DIR:
    # use |û| stats from the same grid we TRE on + lung from CT_06 HU.
    rows = []
    for case in args.cases:
        sid = f"DIR_C{case:02d}"
        train = A1 / sid / sid / "train"
        ct06 = sitk.GetArrayFromImage(sitk.ReadImage(str(train / "CT_06.mha"))).astype(
            np.float32
        )
        sub_ref = sitk.ReadImage(str(train / "sub_CT_06.mha"))
        u_infer, _, _ = synth_phase01(g, ct06, device)
        base = dvf_to_sub128(pull_to_elastix_zyx(u_infer), sub_ref).astype(np.float64)

        # Features on 128³ elastix-grid û (fair: only mid-CT + Decoder output)
        u_t = torch.from_numpy(base.astype(np.float32)).permute(3, 0, 1, 2)[None].to(
            device
        )  # (1,3,Z,Y,X)
        # CT on sub128 for lung frac — use sub_CT_06
        sub_ct = sitk.GetArrayFromImage(sub_ref).astype(np.float32)
        mask = lung_mask_from_ct(sub_ct).to(device)
        # reference_ct as (1,1,Z,Y,X); scale HU to roughly µ-like not required for mean cue
        ref = torch.from_numpy(sub_ct)[None, None].to(device)
        # DIR inhale-exhale: phase 5→0 in 0..9 indexing used by Decoder (06→01)
        feat = feature_vector(ref, mask, u_t, ref_phase=5, tgt_phase=0)
        with torch.no_grad():
            a = float(head(torch.tensor([feat], dtype=torch.float32, device=device)).item())

        tre1 = float(tre_t00_t50(base, case, "75", r3=True)["registered"]["mean"])
        # SI-only scale (ch1) — matches mag/dir oracle finding; also report isotropic
        u_si = base.copy()
        u_si[..., 1] *= a
        u_iso = base * a
        tre_si = float(tre_t00_t50(u_si, case, "75", r3=True)["registered"]["mean"])
        tre_iso = float(tre_t00_t50(u_iso, case, "75", r3=True)["registered"]["mean"])
        row = {
            "case": case,
            "scan_id": sid,
            "a": a,
            "feat": feat,
            "tre75_a1": tre1,
            "tre75_si_a": tre_si,
            "tre75_iso_a": tre_iso,
        }
        rows.append(row)
        print(
            f"{sid} a={a:.3f} | a=1 {tre1:.2f} | SI×a {tre_si:.2f} (Δ {tre1-tre_si:+.2f}) | "
            f"iso×a {tre_iso:.2f} (Δ {tre1-tre_iso:+.2f})",
            flush=True,
        )

    def cohort(key):
        v = np.array([r[key] for r in rows], float)
        return float(v.mean()), float(v.std(ddof=1))

    m1, s1 = cohort("tre75_a1")
    msi, ssi = cohort("tre75_si_a")
    miso, siso = cohort("tre75_iso_a")
    print("\n=== cohort TRE75 ===")
    print(f"  a=1       {m1:.3f} ± {s1:.3f}")
    print(f"  SI×a      {msi:.3f} ± {ssi:.3f}  (gain {m1-msi:+.3f})")
    print(f"  iso×a     {miso:.3f} ± {siso:.3f}  (gain {m1-miso:+.3f})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "protocol": "FeatAmpHead mid-CT+|û|; DIR TRE75 R3; SI-scale vs isotropic",
                "feat_ckpt": str(args.feat_ckpt),
                "dec_ckpt": str(args.dec_ckpt.resolve()),
                "per_case": rows,
                "cohort": {
                    "a1": {"mean": m1, "std": s1},
                    "si_a": {"mean": msi, "std": ssi},
                    "iso_a": {"mean": miso, "std": siso},
                },
            },
            indent=2,
        )
        + "\n"
    )
    print("Wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
