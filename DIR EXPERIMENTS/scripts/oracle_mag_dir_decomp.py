#!/usr/bin/env python3
"""Offline magnitude-vs-direction oracle for a frozen Decoder on DIR TRE75.

For each case, with û = synth 06→01 and u_el = Elastix DVF_sub_01 (same 128³ grid):
  base       û
  elastix    u_el                                  (sanity: ~A1)
  mag_el     (û/|û|)·|u_el|    direction ours, size perfect per voxel
  dir_el     (u_el/|u_el|)·|û| size ours, direction perfect per voxel
  smooth_s   s·û, s = G_σ|u_el| / G_σ|û|           low-frequency scale map
  axis_si    û with SI component scaled by best a_z (others 1)

  cd "DIR EXPERIMENTS"
  LEARN-GUI/.venv/bin/python scripts/oracle_mag_dir_decomp.py --gpu 0
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
from scipy.ndimage import gaussian_filter

DIR_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIR_EXP / "scripts"))

from amp_oracle_a_sweep import DEFAULT_CKPT, load_decoder  # noqa: E402
from compare_g160_enc_dec_on_dir import (  # noqa: E402
    dvf_to_sub128,
    pull_to_elastix_zyx,
    synth_phase01,
)
from eval_a1_tre import load_dvf_zyx3, tre_t00_t50  # noqa: E402

A1 = DIR_EXP / "arms" / "A1_oracle_dirlab" / "runs"
OUT = DIR_EXP / "arms" / "A3_synth_conditioned" / "results"
EPS = 1e-3


def tre75(u: np.ndarray, case: int) -> float:
    return float(tre_t00_t50(u, case, "75", r3=True)["registered"]["mean"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--cases", type=int, nargs="*", default=list(range(1, 11)))
    ap.add_argument("--sigmas", type=float, nargs="*", default=[4.0, 8.0, 16.0])
    ap.add_argument("--out", type=Path, default=OUT / "oracle_mag_dir_decomp_tcia3_ep100.json")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("DIRLAB_ROOT", str(DIR_EXP / "data" / "dirlab_packs"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    g = load_decoder(args.ckpt.resolve(), device)

    a_grid = np.round(np.arange(0.8, 3.01, 0.1), 2)
    rows = []
    for case in args.cases:
        sid = f"DIR_C{case:02d}"
        train = A1 / sid / sid / "train"
        ct06 = sitk.GetArrayFromImage(sitk.ReadImage(str(train / "CT_06.mha"))).astype(np.float32)
        sub_ref = sitk.ReadImage(str(train / "sub_CT_06.mha"))
        u_hat, _, _ = synth_phase01(g, ct06, device)
        u_hat = dvf_to_sub128(pull_to_elastix_zyx(u_hat), sub_ref).astype(np.float64)
        u_el = load_dvf_zyx3(train / "DVF_sub_01.mha")

        m_hat = np.linalg.norm(u_hat, axis=-1, keepdims=True)
        m_el = np.linalg.norm(u_el, axis=-1, keepdims=True)
        d_hat = np.where(m_hat > EPS, u_hat / np.maximum(m_hat, EPS), 0.0)
        d_el = np.where(m_el > EPS, u_el / np.maximum(m_el, EPS), 0.0)

        r = {"case": case, "scan_id": sid}
        r["base"] = tre75(u_hat, case)
        r["elastix"] = tre75(u_el, case)
        r["mag_el"] = tre75(d_hat * m_el, case)
        r["dir_el"] = tre75(d_el * m_hat, case)

        for s in args.sigmas:
            num = gaussian_filter(m_el[..., 0], s)
            den = gaussian_filter(m_hat[..., 0], s)
            scale = np.clip(num / np.maximum(den, EPS), 0.5, 4.0)[..., None]
            r[f"smooth_s{int(s)}"] = tre75(u_hat * scale, case)

        # SI axis is z in (z,y,x,3) arrays; component order follows the DVF's xyz channels.
        best = (1e9, None, None)
        for ch in range(3):
            for a in a_grid:
                u = u_hat.copy()
                u[..., ch] *= a
                t = tre75(u, case)
                if t < best[0]:
                    best = (t, ch, float(a))
        r["axis_best"] = best[0]
        r["axis_best_ch"] = best[1]
        r["axis_best_a"] = best[2]

        lung = m_el[..., 0] > 1.0
        r["mag_ratio_p50"] = float(np.median(m_hat[..., 0][lung] / np.maximum(m_el[..., 0][lung], EPS)))
        r["cos_mean"] = float(np.mean(np.sum(d_hat * d_el, axis=-1)[lung]))
        rows.append(r)
        print(
            f"{sid} base {r['base']:.2f} | El {r['elastix']:.2f} | mag_el {r['mag_el']:.2f} | "
            f"dir_el {r['dir_el']:.2f} | "
            + " ".join(f"s{int(s)} {r[f'smooth_s{int(s)}']:.2f}" for s in args.sigmas)
            + f" | axis ch{r['axis_best_ch']}×{r['axis_best_a']} {r['axis_best']:.2f}"
            f" | ratio {r['mag_ratio_p50']:.2f} cos {r['cos_mean']:.2f}",
            flush=True,
        )

    keys = ["base", "elastix", "mag_el", "dir_el"] + [f"smooth_s{int(s)}" for s in args.sigmas] + ["axis_best"]
    cohort = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    print("\n=== cohort TRE75 (mm) ===")
    for k in keys:
        print(f"  {k:12s} {cohort[k]:.3f}  (Δ vs base {cohort['base'] - cohort[k]:+.3f})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "when": datetime.now(timezone.utc).astimezone().isoformat(),
        "ckpt": str(args.ckpt.resolve()),
        "protocol": "DIR-Lab 75-pt T00→T50 R3; offline oracles (use Elastix; not deployable)",
        "per_case": rows,
        "cohort": cohort,
    }, indent=2) + "\n")
    print("Wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
