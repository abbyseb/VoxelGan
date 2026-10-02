#!/usr/bin/env python3
"""Step 1 + 2 on the corrected DIR geometry (v2 chain, no flip).

Part A  Re-score old synthesiser checkpoints (TCIA3, MagFT, TCIA4, TCIA3.5, TCIA3.1)
        on DIR-Lab with the verified 2 mm / lung-centred / min-max input.
Part B  Magnitude-vs-direction oracle on TCIA3 ep100 (uses Elastix: a ceiling,
        NOT a deployable result).

Reuses every geometry function from eval_dir_tcia3_iso2mm_v2.py, whose
round-trip gate passed (C01 1.262 vs 1.263, C08 3.770 vs 3.771).

  cd "DIR EXPERIMENTS"
  CUDA_VISIBLE_DEVICES=1 python scripts/rescore_oracle_iso2mm_v3.py
  # options: --skip-oracle  --skip-rescore  --cases 1 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

DIR_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIR_EXP / "scripts"))
import eval_dir_tcia3_iso2mm_v2 as v2  # noqa: E402

G160 = v2.VOXEL / "PopulationStudy/ClinicalExperiments/Grid160"
CKPTS = [
    ("TCIA3_ep100", G160 / "TCIA3/DecoderCRB/checkpoints/epoch_100.pt", 1.0),
    ("TCIA3_ep100_x1.04_AmpHead", G160 / "TCIA3/DecoderCRB/checkpoints/epoch_100.pt", 1.042),
    ("MagFT_ep12", G160 / "TCIA3_magFT/DecoderCRB/checkpoints/epoch_012.pt", 1.0),
    ("MagFT_ep27", G160 / "TCIA3_magFT/DecoderCRB/checkpoints/epoch_027.pt", 1.0),
    ("MagFT_ep30", G160 / "TCIA3_magFT/DecoderCRB/checkpoints/epoch_030.pt", 1.0),
    ("MagMatch_ep01", G160 / "TCIA3_magMatch/DecoderCRB/checkpoints/epoch_001.pt", 1.0),
    ("MagMatch_ep02", G160 / "TCIA3_magMatch/DecoderCRB/checkpoints/epoch_002.pt", 1.0),
    ("MagMatch_ep03", G160 / "TCIA3_magMatch/DecoderCRB/checkpoints/epoch_003.pt", 1.0),
    ("MagMatch_ep04", G160 / "TCIA3_magMatch/DecoderCRB/checkpoints/epoch_004.pt", 1.0),
    ("MagMatch_ep05", G160 / "TCIA3_magMatch/DecoderCRB/checkpoints/epoch_005.pt", 1.0),
    ("MagMatch_ep06", G160 / "TCIA3_magMatch/DecoderCRB/checkpoints/epoch_006.pt", 1.0),
    ("MagMatch_ep07", G160 / "TCIA3_magMatch/DecoderCRB/checkpoints/epoch_007.pt", 1.0),
    ("MagMatch_ep08", G160 / "TCIA3_magMatch/DecoderCRB/checkpoints/epoch_008.pt", 1.0),
    ("MagMatch_ep09", G160 / "TCIA3_magMatch/DecoderCRB/checkpoints/epoch_009.pt", 1.0),
    ("MagMatch_ep10", G160 / "TCIA3_magMatch/DecoderCRB/checkpoints/epoch_010.pt", 1.0),
    (
        "MagMatch_weights",
        G160 / "TCIA3_magMatch/DecoderCRB/weights/crb_dec_mae_magmatch_ft_tcia3ep100_generator.pth",
        1.0,
    ),
    ("TCIA4_ep26", G160 / "TCIA4/DecoderCRB/checkpoints/epoch_026.pt", 1.0),
    ("TCIA3.5_ep92", G160 / "TCIA3.5/DecoderCRB/checkpoints/epoch_092.pt", 1.0),
    ("TCIA3.5_ep100", G160 / "TCIA3.5/DecoderCRB/checkpoints/epoch_100.pt", 1.0),
    ("TCIA3.1_ep74", G160 / "TCIA3.1/DecoderCRB/checkpoints/epoch_074.pt", 1.0),
]
OUT = v2.OUT_DIR / "rescore_oracle_iso2mm_v3.json"
EPS = 1e-3
A_GRID = np.round(np.arange(0.8, 3.01, 0.1), 2)


def load_ckpt(path: Path, device):
    net_py = os.environ.get("TCIA_NET_PY", "").strip()
    if net_py:
        import importlib.util
        spec = importlib.util.spec_from_file_location("tcia_net_override", net_py)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        UNetCRBDecoder = mod.UNetCRBDecoder
    else:
        if str(v2.NET) not in sys.path:
            sys.path.insert(0, str(v2.NET))
        from networks.generator_crb_dec import UNetCRBDecoder

    g = UNetCRBDecoder(im_size=v2.SIZE, n_phases=10)
    raw = torch.load(str(path), map_location=device, weights_only=False)
    if isinstance(raw, dict):
        for k in ("generator", "state_dict", "model"):
            if k in raw and isinstance(raw[k], dict):
                raw = raw[k]
                break
    g.load_state_dict(raw, strict=True)
    return g.to(device).eval()


def tre(dvf, case, pack):
    r75 = v2.score_set(dvf, case, "75", pack)["registered"]["mean"]
    r300 = v2.score_set(dvf, case, "300", pack)["registered"]["mean"]
    return r75, r300


def summarize(rows: dict) -> dict:
    t75 = np.array([r[0] for r in rows.values()])
    t300 = np.array([r[1] for r in rows.values()])
    return {
        "tre75_mean": float(t75.mean()),
        "tre300_mean": float(t300.mean()),
        "tre300_std": float(t300.std(ddof=1)) if len(t300) > 1 else 0.0,
        "per_case": {str(c): {"tre75": v[0], "tre300": v[1]} for c, v in rows.items()},
    }


def oracle_case(u_hat, u_el, pack, case, sigmas=(4.0, 8.0)):
    m_hat = np.linalg.norm(u_hat, axis=-1, keepdims=True)
    m_el = np.linalg.norm(u_el, axis=-1, keepdims=True)
    d_hat = np.where(m_hat > EPS, u_hat / np.maximum(m_hat, EPS), 0.0)
    d_el = np.where(m_el > EPS, u_el / np.maximum(m_el, EPS), 0.0)
    out = {
        "base": tre(u_hat, case, pack),
        "elastix": tre(u_el, case, pack),
        "mag_el": tre((d_hat * m_el).astype(np.float32), case, pack),
        "dir_el": tre((d_el * m_hat).astype(np.float32), case, pack),
    }
    for s in sigmas:
        num = gaussian_filter(m_el[..., 0], s)
        den = gaussian_filter(m_hat[..., 0], s)
        scale = np.clip(num / np.maximum(den, EPS), 0.5, 4.0)[..., None]
        out[f"smooth_s{int(s)}"] = tre((u_hat * scale).astype(np.float32), case, pack)
    # one head-foot (dy, channel 1) scale per case, chosen on TRE75 (oracle)
    best = None
    for a in A_GRID:
        u = u_hat.copy()
        u[..., 1] *= a
        r = tre(u, case, pack)
        if best is None or r[0] < best[1][0]:
            best = (float(a), r)
    out["si_scale"] = best[1]
    lung = pack["lung_iso"] & (m_el[..., 0] > 0.5)
    diag = {
        "si_scale_a": best[0],
        "mag_ratio_p50": float(np.median(m_hat[..., 0][lung] / np.maximum(m_el[..., 0][lung], EPS))),
        "cos_mean": float(np.mean(np.sum(d_hat * d_el, axis=-1)[lung])),
    }
    return out, diag


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=int, nargs="*", default=list(range(1, 11)))
    ap.add_argument("--skip-oracle", action="store_true")
    ap.add_argument("--skip-rescore", action="store_true")
    ap.add_argument("--ckpt", action="append", default=[],
                    help="LABEL=PATH, repeatable; replaces the built-in checkpoint list")
    args = ap.parse_args()
    global CKPTS
    if args.ckpt:
        CKPTS = [(c.split("=", 1)[0], Path(c.split("=", 1)[1]), 1.0) for c in args.ckpt]

    os.environ.setdefault("DIRLAB_ROOT", str(DIR_EXP / "data" / "dirlab_packs"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("packing cases…", flush=True)
    packs = {c: v2.pack_case(c) for c in args.cases}

    report = {
        "geometry": "v2: 2 mm, 160³ lung-centred, per-scan µ min-max, no flip, verified landmark chain",
        "cases": args.cases,
        "when": datetime.now(timezone.utc).astimezone().isoformat(),
        "rescore": {},
    }

    # -------- Part A
    if not args.skip_rescore:
        for label, path, scale in CKPTS:
            if not path.is_file():
                print(f"[A] SKIP {label}: missing {path}", flush=True)
                report["rescore"][label] = {"missing": str(path)}
                continue
            try:
                g = load_ckpt(path, device)
            except Exception as e:  # noqa: BLE001
                print(f"[A] SKIP {label}: load failed ({e})", flush=True)
                report["rescore"][label] = {"load_error": str(e)}
                continue
            rows = {}
            for c in args.cases:
                u = v2.predict(g, device, packs[c]["mu"], flip=False) * scale
                rows[c] = tre(u, c, packs[c])
            s = summarize(rows)
            s["ckpt"] = str(path)
            s["scale"] = scale
            report["rescore"][label] = s
            c8 = s["per_case"].get("8", {}).get("tre300", float("nan"))
            print(f"[A] {label:28s} TRE75 {s['tre75_mean']:.3f}  TRE300 {s['tre300_mean']:.3f}  C08 {c8:.2f}", flush=True)
            del g
            torch.cuda.empty_cache()

    # -------- Part B
    if not args.skip_oracle:
        g = load_ckpt(CKPTS[0][1], device)
        per_case, diags = {}, {}
        for c in args.cases:
            u_hat = v2.predict(g, device, packs[c]["mu"], flip=False)
            u_el = v2.elastix_on_iso(packs[c])
            per_case[c], diags[c] = oracle_case(u_hat, u_el, packs[c], c)
            o = per_case[c]
            print(
                f"[B] C{c:02d} base {o['base'][1]:.2f} | El {o['elastix'][1]:.2f} | "
                f"mag_el {o['mag_el'][1]:.2f} | dir_el {o['dir_el'][1]:.2f} | "
                f"s4 {o['smooth_s4'][1]:.2f} s8 {o['smooth_s8'][1]:.2f} | "
                f"SI×{diags[c]['si_scale_a']} {o['si_scale'][1]:.2f} | "
                f"ratio {diags[c]['mag_ratio_p50']:.2f} cos {diags[c]['cos_mean']:.2f}   (TRE300)",
                flush=True,
            )
        keys = list(next(iter(per_case.values())).keys())
        cohort = {
            k: {
                "tre75": float(np.mean([per_case[c][k][0] for c in per_case])),
                "tre300": float(np.mean([per_case[c][k][1] for c in per_case])),
            }
            for k in keys
        }
        report["oracle"] = {
            "note": "Uses Elastix per case. Ceiling analysis only, not a result.",
            "cohort": cohort,
            "per_case": {str(c): {k: {"tre75": v[0], "tre300": v[1]} for k, v in per_case[c].items()}
                         for c in per_case},
            "diag": {str(c): d for c, d in diags.items()},
        }

    global OUT
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT = OUT.with_name(f"{OUT.stem}_{stamp}.json")  # never overwrite an earlier run
    OUT.write_text(json.dumps(report, indent=2) + "\n")

    print("\n================ SUMMARY (mm) ================")
    if report["rescore"]:
        print("Part A: re-scored checkpoints (baseline TCIA3 ep100 = 4.25 TRE300)")
        ok = [(k, v) for k, v in report["rescore"].items() if "tre300_mean" in v]
        for k, v in sorted(ok, key=lambda kv: kv[1]["tre300_mean"]):
            c8 = v["per_case"].get("8", {}).get("tre300", float("nan"))
            print(f"  {k:28s} TRE75 {v['tre75_mean']:.3f}  TRE300 {v['tre300_mean']:.3f}  C08 {c8:.2f}")
    if "oracle" in report:
        print("Part B: oracle on TCIA3 ep100 (Elastix-assisted, ceiling only)")
        for k, v in report["oracle"]["cohort"].items():
            print(f"  {k:12s} TRE75 {v['tre75']:.3f}  TRE300 {v['tre300']:.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
