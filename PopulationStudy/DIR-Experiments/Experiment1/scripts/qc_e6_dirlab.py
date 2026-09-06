#!/usr/bin/env python3
"""DIR-Experiments E1: clinical-style QC (100 pairs, Elastix GT, panels).

Same recipe as ClinicalExperiments qc_clinical / E6 Normal:
  100 directed pairs, L1/cos vs Elastix, 7 panel PNGs per patient.

Requires prepare_dir_dvf_library.py first (*_pair.npy in packed/.../all/).

  cd PopulationStudy/DIR-Experiments/Experiment1
  PYTHONPATH=../../InitialExperiments/Experiment1 python scripts/qc_e6_dirlab.py \\
      --arches encoder,decoder,both --norm both --gpu 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

E1 = Path(__file__).resolve().parents[1]
PACKED = E1 / "packed"
IE1 = E1.parent.parent / "InitialExperiments" / "Experiment1"
sys.path.insert(0, str(IE1))

from losses.losses import DVFLoss  # noqa: E402
from networks.generator_crb import UNetCRB  # noqa: E402
from networks.generator_crb_both import UNetCRBBoth  # noqa: E402
from networks.generator_crb_dec import UNetCRBDecoder  # noqa: E402
from utilities.warp import warp  # noqa: E402

ARCH = {
    "encoder": UNetCRB,
    "decoder": UNetCRBDecoder,
    "both": UNetCRBBoth,
}
CKPTS = {
    "encoder": E1 / "weights/encoder_e6_normal.pth",
    "decoder": E1 / "weights/decoder_e6_normal.pth",
    "both": E1 / "weights/both_e6_normal.pth",
}
DEFAULT_PATIENTS = [f"P{i}_DIR" for i in range(1, 11)]
DEFAULT_PANEL_PAIRS = {
    "01_to_02", "01_to_06", "01_to_10",
    "06_to_01", "06_to_06", "03_to_07", "05_to_08",
}


def norm_minmax(x):
    x = x.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi <= lo else (x - lo) / (hi - lo)


def norm_mu(x, air=0.0, water=0.02):
    x = x.astype(np.float32)
    y = (x - air) / max(water - air, 1e-8)
    return np.clip(y, 0.0, 1.5) / 1.5


NORMS = {"minmax": norm_minmax, "mu": norm_mu}


def to_cdhw(dvf):
    if dvf.ndim == 4 and dvf.shape[-1] == 3:
        return np.moveaxis(dvf, -1, 0)
    return dvf


def lung_mid_z(mask):
    zs = np.where(mask.any(axis=(1, 2)))[0]
    return int(zs[len(zs) // 2]) if zs.size else mask.shape[0] // 2


def display_axial(im):
    """Rotate 180° so axial slices match SPARE/clinical QC orientation."""
    return np.rot90(im, 2)


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
        (axs[0, 3], img_err[z], "|tgt-warp|·lung", "hot", 0, None),
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
        elif ttl in ("|tgt-warp|·lung", "|pred-GT|·lung"):
            pos = im[im > 0]
            kw["vmax"] = float(np.percentile(pos, 99)) if pos.size else 1.0
        h = ax.imshow(display_axial(im), **kw)
        ax.set_title(ttl, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(h, ax=ax, fraction=0.046, pad=0.04)
    axs[1, 3].axis("off")
    fig.suptitle(f"{title}  L1={l1:.3f}  cos={cos:.2f}  k={k:.2f}  (mid-lung z={z})", fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def run_one(
    arch, pid, g, device, dvf_l1, norm_fn, norm_name, panel_pairs, no_panels, force,
    packed_root: Path,
):
    data_dir = packed_root / pid / "all"
    out_dir = E1 / "plots" / f"qc_{arch}" / norm_name / pid
    if (out_dir / "summary.json").is_file() and not force:
        print(f"skip {arch} {pid} {norm_name} (exists)", flush=True)
        return json.loads((out_dir / "summary.json").read_text())

    pair0 = data_dir / "01_to_02_pair.npy"
    if not pair0.is_file():
        raise FileNotFoundError(
            f"missing {pair0} — run prepare_dir_dvf_library.py for {pid} first"
        )

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
            if not no_panels and stem in panel_pairs:
                save_panel(
                    out_dir / f"{stem}.png",
                    ct_r, ct_t, warped, gt, pred, mask, z,
                    f"{pid} {stem} (E6 {arch} · {norm_name})",
                    l1, cos, k,
                )
            if i % 25 == 0:
                print(f"  [{arch}|{pid}|{norm_name}] {i}/100", flush=True)

    directed = [r for r in rows if r["ref"] != r["tgt"]]
    summary = {
        "patient": pid,
        "arch": arch,
        "norm": norm_name,
        "experiment": "DIR_E1_clinical_style",
        "ckpt": str(CKPTS[arch]),
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
        f"[{arch}|{pid}|{norm_name}] cos={summary['mean_cos']:.3f} "
        f"beat={summary['beat_zero_pct']:.0f}% → {out_dir}",
        flush=True,
    )
    return summary


def run_panels_only(arch, pid, g, device, dvf_l1, norm_fn, norm_name, panel_pairs, force, packed_root):
    data_dir = packed_root / pid / "all"
    out_dir = E1 / "plots" / f"qc_{arch}" / norm_name / pid
    summary_path = out_dir / "summary.json"
    if not summary_path.is_file():
        print(f"WARN no summary for {arch} {pid} {norm_name} — run full QC first", flush=True)
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
                f"{pid} {stem} (E6 {arch} · {norm_name})",
                l1, cos, k,
            )
            n_saved += 1
    print(f"[{arch}|{pid}|{norm_name}] panels saved={n_saved} → {out_dir}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--arch", default="both", choices=sorted(ARCH))
    ap.add_argument("--arches", default=None, help="comma list overrides --arch")
    ap.add_argument("--norm", default="minmax", choices=("minmax", "mu", "both"))
    ap.add_argument("--patients", default=",".join(DEFAULT_PATIENTS))
    ap.add_argument("--packed-root", type=Path, default=PACKED)
    ap.add_argument("--no-panels", action="store_true")
    ap.add_argument("--panels-only", action="store_true", help="rewrite PNGs from existing metrics")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    packed_root = args.packed_root

    arches = [a.strip() for a in (args.arches or args.arch).split(",") if a.strip()]
    patients = [p.strip() for p in args.patients.split(",") if p.strip()]
    norms = ["minmax", "mu"] if args.norm == "both" else [args.norm]

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dvf_l1 = DVFLoss()
    print(f"DIR clinical QC  device={device}  arches={arches}  norms={norms}", flush=True)
    if args.panels_only:
        print("panels-only mode (metrics unchanged)", flush=True)

    all_summ = {}
    lines = ["experiment\tarch\tnorm\tpatient\tcos\tL1_over_zero\tbeat"]
    for arch in arches:
        ckpt = CKPTS[arch]
        if not ckpt.is_file():
            print(f"WARN missing {ckpt}", flush=True)
            continue
        g = ARCH[arch](im_size=128, n_phases=10).to(device)
        g.load_state_dict(torch.load(ckpt, map_location=device))
        g.eval()
        print(f"loaded {arch} {ckpt.name}", flush=True)
        all_summ[arch] = {}
        for norm_name in norms:
            all_summ[arch][norm_name] = {}
            for pid in patients:
                if not (packed_root / pid / "pack_meta.json").is_file():
                    print(f"WARN skip {pid}: not packed", flush=True)
                    continue
                if args.panels_only:
                    s = run_panels_only(
                        arch, pid, g, device, dvf_l1,
                        NORMS[norm_name], norm_name, DEFAULT_PANEL_PAIRS,
                        args.force, packed_root,
                    )
                    if s is None:
                        continue
                else:
                    s = run_one(
                        arch, pid, g, device, dvf_l1,
                        NORMS[norm_name], norm_name, DEFAULT_PANEL_PAIRS,
                        args.no_panels, args.force, packed_root,
                    )
                all_summ[arch][norm_name][pid] = {
                    k: s[k]
                    for k in ("mean_L1", "mean_L1_over_zero", "mean_cos", "mean_k", "beat_zero_pct")
                }
                lines.append(
                    f"DIR\t{arch}\t{norm_name}\t{pid}\t{s['mean_cos']:.4f}\t"
                    f"{s['mean_L1_over_zero']:.4f}\t{s['beat_zero_pct']:.1f}"
                )

    if args.panels_only:
        print("DONE (panels only)", flush=True)
        return

    print("\n======== DIR COHORT SUMMARY ========", flush=True)
    for arch in all_summ:
        for norm_name in all_summ[arch]:
            subset = list(all_summ[arch][norm_name].values())
            if not subset:
                continue
            cos = float(np.mean([x["mean_cos"] for x in subset]))
            beat = float(np.mean([x["beat_zero_pct"] for x in subset]))
            ratio = float(np.mean([x["mean_L1_over_zero"] for x in subset]))
            print(
                f"{arch:8s} {norm_name:6s}  cos={cos:.3f}  L1/z={ratio:.3f}  beat={beat:.0f}%  "
                f"n={len(subset)}",
                flush=True,
            )

    log_dir = E1 / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    tag = "_".join(arches) + "_" + "_".join(norms)
    (log_dir / f"clinical_summary_{tag}.json").write_text(json.dumps(all_summ, indent=2) + "\n")
    (log_dir / f"clinical_metrics_{tag}.tsv").write_text("\n".join(lines) + "\n")
    print(f"wrote {log_dir / f'clinical_metrics_{tag}.tsv'}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
