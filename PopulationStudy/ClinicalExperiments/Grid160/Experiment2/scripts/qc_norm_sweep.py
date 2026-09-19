#!/usr/bin/env python3
"""G160-E2: inference-only CT intensity norm sweep across A0 / A0h / A1 checkpoints.

Metrics only (no panels). Recipe from Grid128 E5.

  cd PopulationStudy/ClinicalExperiments/Grid160/Experiment2
  PYTHONPATH=. python scripts/qc_norm_sweep.py --gpu 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

E2 = Path(__file__).resolve().parents[1]
G160 = E2.parent
E0 = G160 / "Experiment0"
E1 = G160 / "Experiment1"
CE1 = G160.parent / "Experiment1"
IE6 = G160.parents[1] / "InitialExperiments" / "Experiment6"
sys.path.insert(0, str(IE6))

from losses.losses import DVFLoss  # noqa: E402
from networks.generator_crb import UNetCRB  # noqa: E402
from networks.generator_crb_both import UNetCRBBoth  # noqa: E402
from networks.generator_crb_dec import UNetCRBDecoder  # noqa: E402

ARCH = {"encoder": UNetCRB, "decoder": UNetCRBDecoder, "both": UNetCRBBoth}
ARCH_DIR = {"encoder": "EncoderCRB", "decoder": "DecoderCRB", "both": "BothCRB"}
IM_SIZE = 64

# (arm_id, root, arch -> ckpt filename)
CHECKPOINTS = {
    "a0_full": (
        E0,
        {
            "encoder": "crb_enc_mse_iso_g160_a0_full_generator.pth",
            "decoder": "crb_dec_mse_iso_g160_a0_full_generator.pth",
            "both": "crb_both_mse_iso_g160_a0_full_generator.pth",
        },
    ),
    "a0h": (
        E0,
        {
            "encoder": "crb_enc_mse_iso_g160_a0h_generator.pth",
            "decoder": "crb_dec_mse_iso_g160_a0h_generator.pth",
            "both": "crb_both_mse_iso_g160_a0h_generator.pth",
        },
    ),
    "a1_full": (
        E1,
        {
            "encoder": "crb_enc_mse_iso_g160_fov_full_generator.pth",
            "decoder": "crb_dec_mse_iso_g160_fov_full_generator.pth",
            "both": "crb_both_mse_iso_g160_fov_full_generator.pth",
        },
    ),
}

NORMS = [
    ("minmax", None),
    ("p0.5_p99.5", (0.5, 99.5)),
    ("p1_p99", (1.0, 99.0)),
    ("p2_p98", (2.0, 98.0)),
    ("p5_p95", (5.0, 95.0)),
    ("mu_air_water", "mu"),
]
PCT_PROFILE = (1, 5, 25, 50, 75, 95, 99)
DEFAULT_SCANS = [f"CV_P{i}_V_01" for i in range(1, 6)] + [f"CE_P{i}_V_01" for i in range(1, 6)]


def norm_minmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi <= lo else (x - lo) / (hi - lo)


def norm_percentile(x: np.ndarray, plo: float, phi: float) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, [plo, phi]).astype(np.float32)
    if hi <= lo:
        return np.zeros_like(x)
    y = np.clip(x, lo, hi)
    return (y - lo) / (hi - lo)


def norm_mu_air_water(x: np.ndarray, air: float = 0.0, water: float = 0.02) -> np.ndarray:
    x = x.astype(np.float32)
    y = (x - air) / max(water - air, 1e-8)
    return np.clip(y, 0.0, 1.5) / 1.5


def apply_norm(x: np.ndarray, name: str, spec) -> np.ndarray:
    if name == "minmax":
        return norm_minmax(x)
    if name == "mu_air_water":
        return norm_mu_air_water(x)
    plo, phi = spec
    return norm_percentile(x, plo, phi)


def profile(x: np.ndarray) -> np.ndarray:
    return np.percentile(x, PCT_PROFILE).astype(np.float64)


def to_cdhw(dvf: np.ndarray) -> np.ndarray:
    if dvf.ndim == 4 and dvf.shape[-1] == 3:
        return np.moveaxis(dvf, -1, 0)
    return dvf


def metrics(pred, gt, mask, dvf_l1):
    m = mask > 0.5
    gt_t = torch.from_numpy(gt)[None].float()
    pr_t = torch.from_numpy(pred)[None].float()
    mk_t = torch.from_numpy(mask)[None, None].float()
    l1 = float(dvf_l1.loss(gt_t, pr_t, mk_t).item())
    l1z = float(dvf_l1.loss(gt_t, torch.zeros_like(gt_t), mk_t).item())
    if m.any() and float(np.abs(gt).max()) > 0:
        a, b = pred[:, m], gt[:, m]
        cos = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-8))
        k = float(np.linalg.norm(pred[:, m]) / (np.linalg.norm(gt[:, m]) + 1e-8))
    else:
        cos, k = float("nan"), float("nan")
    return dict(l1=l1, l1_zero=l1z, ratio=l1 / (l1z + 1e-8), cos=cos, k=k)


def summarize(rows):
    directed = [r for r in rows if r["ref"] != r["tgt"]]
    return {
        "n_directed": len(directed),
        "mean_L1": float(np.mean([r["l1"] for r in directed])),
        "mean_L1_over_zero": float(np.mean([r["ratio"] for r in directed])),
        "mean_cos": float(np.nanmean([r["cos"] for r in directed])),
        "mean_k": float(np.nanmean([r["k"] for r in directed])),
        "beat_zero_pct": float(100.0 * sum(r["ratio"] < 1.0 for r in directed) / max(len(directed), 1)),
    }


def load_varian_ref_profiles() -> dict[str, np.ndarray]:
    refs = {name: [] for name, _ in NORMS}
    for i in range(1, 6):
        raw = np.load(CE1 / "data" / f"CV_P{i}_V_01" / "all" / "CT_01.npy")
        for name, spec in NORMS:
            refs[name].append(profile(apply_norm(raw, name, spec)))
    return {k: np.mean(v, axis=0) for k, v in refs.items()}


def run_scan(scan: str, g, device, dvf_l1, spare_prof: dict, norms: list):
    data_dir = CE1 / "data" / scan / "all"
    mask = (np.load(data_dir / "Mask_Lung.npy") > 0).astype(np.float32)
    stems = [f"{i:02d}_to_{j:02d}" for i in range(1, 11) for j in range(1, 11)]
    raw_cts = {ph: np.load(data_dir / f"CT_{ph:02d}.npy").astype(np.float32) for ph in range(1, 11)}

    out = {}
    for name, spec in norms:
        rows = []
        p01 = apply_norm(raw_cts[1], name, spec)
        dist = float(np.linalg.norm(profile(p01) - spare_prof[name]))
        with torch.no_grad():
            for stem in stems:
                ref, tgt = int(stem[:2]), int(stem[-2:])
                ct_r = apply_norm(raw_cts[ref], name, spec)
                if ref == tgt:
                    gt = np.zeros((3,) + ct_r.shape, np.float32)
                else:
                    gt = to_cdhw(np.load(data_dir / f"{stem}_pair.npy").astype(np.float32))
                ref_t = torch.from_numpy(ct_r)[None, None].to(device)
                pred = g(
                    ref_t,
                    torch.tensor([ref - 1], device=device),
                    torch.tensor([tgt - 1], device=device),
                )[0].cpu().numpy()
                m = metrics(pred, gt, mask, dvf_l1)
                rows.append(dict(pair=stem, ref=ref, tgt=tgt, **m))
        summ = summarize(rows)
        summ["intensity_dist_to_varian_ref"] = dist
        out[name] = summ
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--arms", default="a0_full,a0h,a1_full")
    ap.add_argument("--arches", default="encoder,decoder,both")
    ap.add_argument("--scans", default=",".join(DEFAULT_SCANS))
    ap.add_argument(
        "--norms",
        default=",".join(n for n, _ in NORMS),
        help="comma list of norm names",
    )
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    arches = [a.strip() for a in args.arches.split(",") if a.strip()]
    scans = [s.strip() for s in args.scans.split(",") if s.strip()]
    want = {n.strip() for n in args.norms.split(",") if n.strip()}
    norms = [(n, s) for n, s in NORMS if n in want]

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dvf_l1 = DVFLoss()
    spare_prof = load_varian_ref_profiles()

    out_root = E2 / "plots" / "qc_norm_sweep"
    out_root.mkdir(parents=True, exist_ok=True)

    results = {
        "experiment": "G160-E2",
        "norms": [n for n, _ in norms],
        "scans": scans,
        "arms": {},
    }
    tsv = ["arm\tarch\tnorm\tvendor\tscan\tcos\tL1_over_zero\tbeat\tdist"]

    print(
        f"G160-E2 norm sweep  device={device}  arms={arms}  arches={arches}  "
        f"norms={[n for n,_ in norms]}",
        flush=True,
    )

    for arm in arms:
        if arm not in CHECKPOINTS:
            print(f"WARN unknown arm {arm}", flush=True)
            continue
        root, ckpt_map = CHECKPOINTS[arm]
        results["arms"][arm] = {}
        for arch in arches:
            ckpt_name = ckpt_map.get(arch)
            if not ckpt_name:
                continue
            ckpt_path = root / ARCH_DIR[arch] / "weights" / ckpt_name
            if not ckpt_path.is_file():
                print(f"WARN skip {arm}/{arch}: missing {ckpt_path}", flush=True)
                continue
            g = ARCH[arch](im_size=IM_SIZE, n_phases=10).to(device)
            g.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
            g.eval()
            print(f"\nloaded {arm} {arch} ← {ckpt_path.name}", flush=True)
            results["arms"][arm][arch] = {"ckpt": str(ckpt_path), "scans": {}}

            for scan in scans:
                print(f"  [{arm}|{arch}|{scan}]", flush=True)
                summ = run_scan(scan, g, device, dvf_l1, spare_prof, norms)
                results["arms"][arm][arch]["scans"][scan] = summ
                vendor = "elekta" if scan.startswith("CE_") else "varian"
                for name, _ in norms:
                    s = summ[name]
                    print(
                        f"    {name:14s}  cos={s['mean_cos']:.3f}  L1/z={s['mean_L1_over_zero']:.3f}  "
                        f"beat={s['beat_zero_pct']:.0f}%  dist={s['intensity_dist_to_varian_ref']:.3f}",
                        flush=True,
                    )
                    tsv.append(
                        f"{arm}\t{arch}\t{name}\t{vendor}\t{scan}\t"
                        f"{s['mean_cos']:.4f}\t{s['mean_L1_over_zero']:.4f}\t"
                        f"{s['beat_zero_pct']:.1f}\t{s['intensity_dist_to_varian_ref']:.4f}"
                    )

            # free GPU before next model
            del g
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print("\n======== G160-E2 COHORT MEANS ========", flush=True)
    cohort_lines = ["arm\tarch\tnorm\tvendor\tcos\tL1_over_zero\tbeat\tdist"]
    for arm in results["arms"]:
        for arch in results["arms"][arm]:
            scans_d = results["arms"][arm][arch]["scans"]
            for vendor, pref in (("varian", "CV_"), ("elekta", "CE_")):
                subset = [s for s in scans_d if s.startswith(pref)]
                if not subset:
                    continue
                for name, _ in norms:
                    cos = float(np.mean([scans_d[s][name]["mean_cos"] for s in subset]))
                    beat = float(np.mean([scans_d[s][name]["beat_zero_pct"] for s in subset]))
                    ratio = float(np.mean([scans_d[s][name]["mean_L1_over_zero"] for s in subset]))
                    dist = float(
                        np.mean([scans_d[s][name]["intensity_dist_to_varian_ref"] for s in subset])
                    )
                    print(
                        f"{arm:8s} {arch:8s} {name:14s} {vendor:7s}  "
                        f"cos={cos:.3f}  L1/z={ratio:.3f}  beat={beat:.0f}%  dist={dist:.3f}",
                        flush=True,
                    )
                    cohort_lines.append(
                        f"{arm}\t{arch}\t{name}\t{vendor}\t{cos:.4f}\t{ratio:.4f}\t{beat:.1f}\t{dist:.4f}"
                    )

    # Best Elekta cos per arm/arch
    print("\n======== BEST ELEKTA NORM PER MODEL ========", flush=True)
    for arm in results["arms"]:
        for arch in results["arms"][arm]:
            scans_d = results["arms"][arm][arch]["scans"]
            elekta = [s for s in scans_d if s.startswith("CE_")]
            if not elekta:
                continue
            best_name, best_cos = None, -1.0
            for name, _ in norms:
                cos = float(np.mean([scans_d[s][name]["mean_cos"] for s in elekta]))
                if cos > best_cos:
                    best_cos, best_name = cos, name
            base = float(np.mean([scans_d[s]["minmax"]["mean_cos"] for s in elekta]))
            print(
                f"{arm:8s} {arch:8s}  best={best_name:14s}  elekta_cos={best_cos:.3f}  "
                f"(minmax={base:.3f}, Δ={best_cos - base:+.3f})",
                flush=True,
            )

    tag = "_".join(arms) + "_" + "_".join(arches)
    (out_root / f"summary_{tag}.json").write_text(json.dumps(results, indent=2) + "\n")
    (out_root / f"metrics_{tag}.tsv").write_text("\n".join(tsv) + "\n")
    (out_root / f"cohort_{tag}.tsv").write_text("\n".join(cohort_lines) + "\n")
    print(f"\nwrote {out_root / f'summary_{tag}.json'}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
