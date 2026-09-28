#!/usr/bin/env python3
"""DIR synth-oracle TRE75: TCIA3 Decoder ± phase AmpHead.

  cd "DIR EXPERIMENTS"
  LEARN-GUI/.venv/bin/python scripts/eval_amp_head_dir_tre.py --gpu 0
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
import torch.nn as nn

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
T3 = DIR_EXP.parent / "PopulationStudy/ClinicalExperiments/Grid160/TCIA3"
DEFAULT_G = T3 / "DecoderCRB/checkpoints/epoch_100.pt"
DEFAULT_AMP = T3 / "AmpHead/amp_head_best.pt"
IM_SIZE = 64


class AmpHead(nn.Module):
    def __init__(self, hidden: int = 64, a_min: float = 0.8, a_max: float = 2.5):
        super().__init__()
        self.a_min, self.a_max = a_min, a_max
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, ref_phase: torch.Tensor, tgt_phase: torch.Tensor) -> torch.Tensor:
        x = torch.stack([ref_phase.float() / 9.0, tgt_phase.float() / 9.0], dim=-1)
        z = self.net(x).squeeze(-1)
        return self.a_min + (self.a_max - self.a_min) * torch.sigmoid(z)


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


def load_amp(ckpt: Path, device: torch.device) -> AmpHead:
    raw = torch.load(str(ckpt), map_location=device, weights_only=False)
    head = AmpHead(
        hidden=int(raw.get("hidden", 64)),
        a_min=float(raw.get("a_min", 0.8)),
        a_max=float(raw.get("a_max", 2.5)),
    )
    head.load_state_dict(raw["amp_head"], strict=True)
    head.to(device).eval()
    return head


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ckpt-g", type=Path, default=DEFAULT_G)
    ap.add_argument("--ckpt-amp", type=Path, default=DEFAULT_AMP)
    ap.add_argument("--cases", type=int, nargs="*", default=list(range(1, 11)))
    ap.add_argument(
        "--out",
        type=Path,
        default=OUT / "amp_head_dir_tre75_tcia3_ep100.json",
    )
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("DIRLAB_ROOT", str(DIR_EXP / "data" / "dirlab_packs"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    g = load_decoder(args.ckpt_g.resolve(), device)
    amp = load_amp(args.ckpt_amp.resolve(), device)

    # synth_phase01 uses ref_ph=5, tgt_ph=0 (CT_06 → CT_01)
    ref_ph = torch.tensor([5], dtype=torch.long, device=device)
    tgt_ph = torch.tensor([0], dtype=torch.long, device=device)
    with torch.no_grad():
        a_pred = float(amp(ref_ph, tgt_ph).item())
    print(f"AmpHead a(φ=5→0) = {a_pred:.4f}", flush=True)

    rows = []
    for case in args.cases:
        sid = f"DIR_C{case:02d}"
        train = A1 / sid / sid / "train"
        ct06 = sitk.GetArrayFromImage(sitk.ReadImage(str(train / "CT_06.mha"))).astype(
            np.float32
        )
        sub_ref = sitk.ReadImage(str(train / "sub_CT_06.mha"))
        u_infer, _, _ = synth_phase01(g, ct06, device)
        base = dvf_to_sub128(pull_to_elastix_zyx(u_infer), sub_ref)

        tre1 = tre_t00_t50(base, case, "75", r3=True)
        tre_a = tre_t00_t50(base * a_pred, case, "75", r3=True)
        row = {
            "case": case,
            "scan_id": sid,
            "a_amphead": a_pred,
            "tre75_a1": float(tre1["registered"]["mean"]),
            "tre75_amphead": float(tre_a["registered"]["mean"]),
            "identity_mm": float(tre1["identity"]["mean"]),
            "delta_vs_a1_mm": float(tre1["registered"]["mean"] - tre_a["registered"]["mean"]),
        }
        rows.append(row)
        print(
            f"{sid} a=1 → {row['tre75_a1']:.2f} | AmpHead a={a_pred:.2f} → {row['tre75_amphead']:.2f} "
            f"(Δ {row['delta_vs_a1_mm']:+.2f})",
            flush=True,
        )

    t1 = np.array([r["tre75_a1"] for r in rows], float)
    ta = np.array([r["tre75_amphead"] for r in rows], float)
    out = {
        "protocol": "DIR-Lab 75-pt T00→T50 R3; TCIA3 ep100 synth 06→01; AmpHead phase-only a",
        "when": datetime.now(timezone.utc).astimezone().isoformat(),
        "ckpt_g": str(args.ckpt_g.resolve()),
        "ckpt_amp": str(args.ckpt_amp.resolve()),
        "a_for_06_to_01": a_pred,
        "note": "Phase-only AmpHead → same a for all cases on 06→01 pair",
        "per_case": rows,
        "cohort": {
            "a1": {"mean_tre75_mm": float(t1.mean()), "std_tre75_mm": float(t1.std(ddof=1))},
            "amphead": {
                "mean_tre75_mm": float(ta.mean()),
                "std_tre75_mm": float(ta.std(ddof=1)),
            },
            "gain_mm": float(t1.mean() - ta.mean()),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print("\n=== cohort TRE75 ===", flush=True)
    print(
        f"  a=1       {out['cohort']['a1']['mean_tre75_mm']:.3f} ± {out['cohort']['a1']['std_tre75_mm']:.3f}",
        flush=True,
    )
    print(
        f"  AmpHead   {out['cohort']['amphead']['mean_tre75_mm']:.3f} ± {out['cohort']['amphead']['std_tre75_mm']:.3f}  "
        f"(a={a_pred:.3f})",
        flush=True,
    )
    print(f"  gain      {out['cohort']['gain_mm']:+.3f} mm", flush=True)
    print("Wrote", args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
