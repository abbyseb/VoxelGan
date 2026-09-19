#!/usr/bin/env python3
"""Clinical zero-shot QC for Grid160 G160-A0 / A0h checkpoints.

  cd PopulationStudy/ClinicalExperiments/Grid160/Experiment0
  PYTHONPATH=. python scripts/qc_clinical.py --arms a0_full,a0h --norm both --no-panels --gpu 0
  PYTHONPATH=. python scripts/qc_clinical.py --arms a0_full,a0h --norm both --panels-only --gpu 0
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

E0 = Path(__file__).resolve().parents[1]
CE1 = E0.parents[1] / "Experiment1"
IE6 = E0.parents[3] / "InitialExperiments" / "Experiment6"
sys.path.insert(0, str(IE6))

from losses.losses import DVFLoss  # noqa: E402
from networks.generator_crb import UNetCRB  # noqa: E402
from networks.generator_crb_both import UNetCRBBoth  # noqa: E402
from networks.generator_crb_dec import UNetCRBDecoder  # noqa: E402
from utilities.warp import warp  # noqa: E402

ARCH = {"encoder": UNetCRB, "decoder": UNetCRBDecoder, "both": UNetCRBBoth}
ARCH_DIR = {"encoder": "EncoderCRB", "decoder": "DecoderCRB", "both": "BothCRB"}
ARM_CKPT = {
    "a0_full": {
        "encoder": "crb_enc_mse_iso_g160_a0_full_generator.pth",
        "decoder": "crb_dec_mse_iso_g160_a0_full_generator.pth",
        "both": "crb_both_mse_iso_g160_a0_full_generator.pth",
    },
    "a0h": {
        "encoder": "crb_enc_mse_iso_g160_a0h_generator.pth",
        "decoder": "crb_dec_mse_iso_g160_a0h_generator.pth",
        "both": "crb_both_mse_iso_g160_a0h_generator.pth",
    },
}
DEFAULT_SCANS = [f"CV_P{i}_V_01" for i in range(1, 6)] + [f"CE_P{i}_V_01" for i in range(1, 6)]
SAMPLE_PANEL_PAIRS = {"01_to_02", "01_to_06", "01_to_10", "06_to_01", "06_to_06", "03_to_07", "05_to_08"}
ALL_PAIR_STEMS = [f"{i:02d}_to_{j:02d}" for i in range(1, 11) for j in range(1, 11)]
IM_SIZE = 64


def panel_stems(mode: str) -> list[str]:
    if mode == "all":
        return ALL_PAIR_STEMS
    if mode == "sample":
        return sorted(SAMPLE_PANEL_PAIRS)
    raise ValueError(f"unknown panel mode: {mode}")


def norm_minmax(x):
    x = x.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi <= lo else (x - lo) / (hi - lo)


def norm_mu(x, air=0.0, water=0.02):
    x = x.astype(np.float32)
    y = (x - air) / max(water - air, 1e-8)
    return np.clip(y, 0.0, 1.5) / 1.5


def to_cdhw(dvf):
    if dvf.ndim == 4 and dvf.shape[-1] == 3:
        return np.moveaxis(dvf, -1, 0)
    return dvf


def lung_mid_z(mask):
    zs = np.where(mask.any(axis=(1, 2)))[0]
    return int(zs[len(zs) // 2]) if zs.size else mask.shape[0] // 2


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
    return l1, l1z, l1 / (l1z + 1e-8), cos, k


def save_panel(out, ct_r, ct_t, warped, gt, pred, mask, z, title, l1, cos, k):
    gt_mag = np.linalg.norm(gt, axis=0)
    pr_mag = np.linalg.norm(pred, axis=0)
    err = np.linalg.norm(pred - gt, axis=0) * mask
    img_err = np.abs(ct_t - warped) * mask
    vmax = max(float(np.percentile(np.concatenate([gt_mag[mask > 0], pr_mag[mask > 0]]), 99)), 1e-3)
    fig, axs = plt.subplots(2, 4, figsize=(14, 7))
    panels = [
        (axs[0, 0], ct_r[z], "ref CT", "gray", 0, 1),
        (axs[0, 1], ct_t[z], "target CT", "gray", 0, 1),
        (axs[0, 2], warped[z], "warp(ref,pred)", "gray", 0, 1),
        (axs[0, 3], img_err[z], "|target-warp|·lung", "hot", 0, None),
        (axs[1, 0], gt_mag[z], "|Elastix|", "magma", 0, vmax),
        (axs[1, 1], pr_mag[z], "|pred|", "magma", 0, vmax),
        (axs[1, 2], err[z], "|pred-GT|·lung", "hot", 0, None),
    ]
    for ax, im, ttl, cmap, vmin, vmax_ in panels:
        kw = dict(cmap=cmap, origin="upper", aspect="equal")
        if vmin is not None:
            kw["vmin"] = vmin
        if vmax_ is not None:
            kw["vmax"] = vmax_
        elif ttl.startswith("|"):
            pos = im[im > 0]
            kw["vmax"] = float(np.percentile(pos, 99)) if pos.size else 1.0
        h = ax.imshow(im, **kw)
        ax.set_title(ttl, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(h, ax=ax, fraction=0.046, pad=0.04)
    axs[1, 3].axis("off")
    fig.suptitle(f"{title}  L1={l1:.3f}  cos={cos:.2f}  k={k:.2f}  z={z}", fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def run_panels_only(arm, arch, scan, g, device, dvf_l1, norm_fn, norm_name, panel_pairs, force, ckpt_path):
    vendor = "elekta" if scan.startswith("CE_") else "varian"
    m = re.match(r"C[VE]_(P\d+)_", scan)
    patient = m.group(1) if m else "unknown"
    data_dir = CE1 / "data" / scan / "all"
    out_dir = E0 / ARCH_DIR[arch] / "plots" / f"qc_{vendor}" / arm / norm_name / patient / scan
    summary_path = out_dir / "summary.json"
    if not summary_path.is_file():
        print(f"WARN no summary for {arm} {arch} {scan} {norm_name} — run metrics first", flush=True)
        return None
    summary = json.loads(summary_path.read_text())
    pair_metrics = {r["pair"]: r for r in summary.get("pairs", [])}

    mask = (np.load(data_dir / "Mask_Lung.npy") > 0).astype(np.float32)
    z = lung_mid_z(mask)
    n_saved = 0
    with torch.no_grad():
        for stem in sorted(panel_pairs):
            png = out_dir / f"{stem}.png"
            if png.is_file() and not force:
                continue
            ref, tgt = int(stem[:2]), int(stem[-2:])
            ct_r = norm_fn(np.load(data_dir / f"CT_{ref:02d}.npy"))
            ct_t = norm_fn(np.load(data_dir / f"CT_{tgt:02d}.npy"))
            if ref == tgt:
                gt = np.zeros((3,) + ct_r.shape, np.float32)
            else:
                gt = to_cdhw(np.load(data_dir / f"{stem}_pair.npy").astype(np.float32))
            ref_t = torch.from_numpy(ct_r)[None, None].to(device)
            pred_t = g(
                ref_t,
                torch.tensor([ref - 1], device=device),
                torch.tensor([tgt - 1], device=device),
            )
            pred = pred_t[0].cpu().numpy()
            warped = warp(ref_t, pred_t)[0, 0].cpu().numpy()
            pm = pair_metrics.get(stem)
            if pm:
                l1, cos, k = pm["l1"], pm["cos"], pm["k"]
            else:
                l1, _, _, cos, k = metrics(pred, gt, mask, dvf_l1)
            save_panel(
                png,
                ct_r, ct_t, warped, gt, pred, mask, z,
                f"{scan} {stem} (G160-{arm} {arch} · {norm_name})",
                l1, cos, k,
            )
            n_saved += 1
    print(f"[{arm}|{arch}|{scan}|{norm_name}] panels saved={n_saved} → {out_dir}", flush=True)
    return summary


def run_one(arm, arch, scan, g, device, dvf_l1, norm_fn, norm_name, panel_pairs, no_panels, force, ckpt_path):
    vendor = "elekta" if scan.startswith("CE_") else "varian"
    m = re.match(r"C[VE]_(P\d+)_", scan)
    patient = m.group(1) if m else "unknown"
    data_dir = CE1 / "data" / scan / "all"
    out_dir = E0 / ARCH_DIR[arch] / "plots" / f"qc_{vendor}" / arm / norm_name / patient / scan
    if (out_dir / "summary.json").is_file() and not force:
        print(f"skip {arm} {arch} {scan} {norm_name} (exists)", flush=True)
        return json.loads((out_dir / "summary.json").read_text())

    mask = (np.load(data_dir / "Mask_Lung.npy") > 0).astype(np.float32)
    z = lung_mid_z(mask)
    stems = [f"{i:02d}_to_{j:02d}" for i in range(1, 11) for j in range(1, 11)]
    rows = []
    with torch.no_grad():
        for i, stem in enumerate(stems, 1):
            ref, tgt = int(stem[:2]), int(stem[-2:])
            ct_r = norm_fn(np.load(data_dir / f"CT_{ref:02d}.npy"))
            ct_t = norm_fn(np.load(data_dir / f"CT_{tgt:02d}.npy"))
            if ref == tgt:
                gt = np.zeros((3,) + ct_r.shape, np.float32)
            else:
                gt = to_cdhw(np.load(data_dir / f"{stem}_pair.npy").astype(np.float32))
            ref_t = torch.from_numpy(ct_r)[None, None].to(device)
            pred_t = g(
                ref_t,
                torch.tensor([ref - 1], device=device),
                torch.tensor([tgt - 1], device=device),
            )
            pred = pred_t[0].cpu().numpy()
            warped = warp(ref_t, pred_t)[0, 0].cpu().numpy()
            l1, l1z, ratio, cos, k = metrics(pred, gt, mask, dvf_l1)
            rows.append(dict(pair=stem, ref=ref, tgt=tgt, l1=l1, l1_zero=l1z, ratio=ratio, cos=cos, k=k))
            if not no_panels:
                save_panel(
                    out_dir / f"{stem}.png",
                    ct_r, ct_t, warped, gt, pred, mask, z,
                    f"{scan} {stem} (G160-{arm} {arch} · {norm_name})",
                    l1, cos, k,
                )
            if i % 25 == 0:
                print(f"  [{arm}|{arch}|{scan}|{norm_name}] {i}/100", flush=True)

    directed = [r for r in rows if r["ref"] != r["tgt"]]
    summary = {
        "scan": scan,
        "arm": arm,
        "arch": arch,
        "norm": norm_name,
        "experiment": f"G160-{arm}",
        "ckpt": str(ckpt_path),
        "n_pairs": len(rows),
        "n_directed": len(directed),
        "mean_L1": float(np.mean([r["l1"] for r in directed])),
        "mean_L1_over_zero": float(np.mean([r["ratio"] for r in directed])),
        "mean_cos": float(np.nanmean([r["cos"] for r in directed])),
        "mean_k": float(np.nanmean([r["k"] for r in directed])),
        "beat_zero_pct": float(100.0 * sum(r["ratio"] < 1.0 for r in directed) / max(len(directed), 1)),
        "pairs": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with open(out_dir / "metrics.tsv", "w") as f:
        f.write("pair\tnote\tL1_pred\tL1_zero\tL1_over_zero\tcos\tk\n")
        for r in rows:
            note = "identity" if r["ref"] == r["tgt"] else "pair"
            f.write(
                f"{r['pair']}\t{note}\t{r['l1']:.6f}\t{r['l1_zero']:.6f}\t"
                f"{r['ratio']:.6f}\t{r['cos']}\t{r['k']}\n"
            )
    print(
        f"[{arm}|{arch}|{scan}|{norm_name}] cos={summary['mean_cos']:.3f} "
        f"beat={summary['beat_zero_pct']:.0f}% → {out_dir}",
        flush=True,
    )
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--arms", default="a0_full,a0h")
    ap.add_argument("--arches", default="encoder,decoder,both")
    ap.add_argument("--norm", default="both", choices=("minmax", "mu", "both"))
    ap.add_argument("--scans", default=",".join(DEFAULT_SCANS))
    ap.add_argument("--no-panels", action="store_true")
    ap.add_argument("--panels-only", action="store_true", help="write PNGs from existing metrics")
    ap.add_argument(
        "--panel-pairs",
        default="all",
        choices=("all", "sample"),
        help="all=100 phase pairs (default); sample=7 QC pairs",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    arches = [a.strip() for a in args.arches.split(",") if a.strip()]
    scans = [s.strip() for s in args.scans.split(",") if s.strip()]
    norms = ["minmax", "mu"] if args.norm == "both" else [args.norm]
    norm_fns = {"minmax": norm_minmax, "mu": norm_mu}
    stems_for_panels = panel_stems(args.panel_pairs)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dvf_l1 = DVFLoss()
    print(
        f"Grid160-A0 QC  device={device}  arms={arms}  arches={arches}  norms={norms}"
        f"  panels_only={args.panels_only}  panel_pairs={args.panel_pairs} ({len(stems_for_panels)})",
        flush=True,
    )

    if args.panels_only:
        for arm in arms:
            for arch in arches:
                ckpt_name = ARM_CKPT[arm][arch]
                ckpt_path = E0 / ARCH_DIR[arch] / "weights" / ckpt_name
                if not ckpt_path.is_file():
                    print(f"WARN skip {arm}/{arch}: missing {ckpt_path}", flush=True)
                    continue
                g = ARCH[arch](im_size=IM_SIZE, n_phases=10).to(device)
                g.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
                g.eval()
                print(f"loaded {arm} {arch} {ckpt_name}", flush=True)
                for norm_name in norms:
                    for scan in scans:
                        run_panels_only(
                            arm, arch, scan, g, device, dvf_l1,
                            norm_fns[norm_name], norm_name, stems_for_panels,
                            args.force, ckpt_path,
                        )
        print("DONE (panels only)", flush=True)
        return

    all_summ = {}
    lines = ["experiment\tarm\tarch\tnorm\tvendor\tscan\tcos\tL1_over_zero\tbeat"]
    for arm in arms:
        all_summ[arm] = {}
        for arch in arches:
            ckpt_name = ARM_CKPT[arm][arch]
            ckpt_path = E0 / ARCH_DIR[arch] / "weights" / ckpt_name
            if not ckpt_path.is_file():
                print(f"WARN skip {arm}/{arch}: missing {ckpt_path}", flush=True)
                continue
            g = ARCH[arch](im_size=IM_SIZE, n_phases=10).to(device)
            g.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
            g.eval()
            print(f"loaded {arm} {arch} {ckpt_name}", flush=True)
            all_summ[arm][arch] = {}
            for norm_name in norms:
                all_summ[arm][arch][norm_name] = {}
                for scan in scans:
                    s = run_one(
                        arm, arch, scan, g, device, dvf_l1,
                        norm_fns[norm_name], norm_name, stems_for_panels,
                        args.no_panels, args.force, ckpt_path,
                    )
                    all_summ[arm][arch][norm_name][scan] = {
                        k: s[k]
                        for k in ("mean_L1", "mean_L1_over_zero", "mean_cos", "mean_k", "beat_zero_pct")
                    }

    print("\n======== GRID160 A0 CLINICAL SUMMARY ========", flush=True)
    for arm in all_summ:
        for arch in all_summ[arm]:
            for norm_name in all_summ[arm][arch]:
                for vendor, pref in (("varian", "CV_"), ("elekta", "CE_")):
                    subset = [s for s in all_summ[arm][arch][norm_name] if s.startswith(pref)]
                    if not subset:
                        continue
                    cos = float(np.mean([all_summ[arm][arch][norm_name][s]["mean_cos"] for s in subset]))
                    beat = float(np.mean([all_summ[arm][arch][norm_name][s]["beat_zero_pct"] for s in subset]))
                    ratio = float(np.mean([all_summ[arm][arch][norm_name][s]["mean_L1_over_zero"] for s in subset]))
                    print(
                        f"{arm:8s} {arch:8s} {norm_name:6s} {vendor:7s}  cos={cos:.3f}  "
                        f"L1/z={ratio:.3f}  beat={beat:.0f}%",
                        flush=True,
                    )
                    for s in subset:
                        m = all_summ[arm][arch][norm_name][s]
                        lines.append(
                            f"G160-{arm}\t{arm}\t{arch}\t{norm_name}\t{vendor}\t{s}\t"
                            f"{m['mean_cos']:.4f}\t{m['mean_L1_over_zero']:.4f}\t{m['beat_zero_pct']:.1f}"
                        )

    out_root = E0 / "plots" / "qc_summary"
    out_root.mkdir(parents=True, exist_ok=True)
    tag = "_".join(arches) + "_" + "_".join(arms) + "_clinical_" + "_".join(norms)
    (out_root / f"summary_{tag}.json").write_text(json.dumps(all_summ, indent=2) + "\n")
    (out_root / f"metrics_{tag}.tsv").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_root / f'summary_{tag}.json'}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
