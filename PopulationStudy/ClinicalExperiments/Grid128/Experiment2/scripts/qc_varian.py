#!/usr/bin/env python3
"""Zero-shot QC for ClinicalExperiments Experiment2 (cyclic).

  cd PopulationStudy/ClinicalExperiments/Experiment2
  PYTHONPATH=. python scripts/qc_varian.py --arch both --scan CV_P1_V_01 --all --gpu 1
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

EXP = Path(__file__).resolve().parents[1]
E1 = EXP.parent / "Experiment1"
IE1 = EXP.parents[1] / "InitialExperiments" / "Experiment1"


def _purge_networks():
    for mod in list(sys.modules):
        if mod == "networks" or mod.startswith("networks."):
            del sys.modules[mod]


def load_arch_classes(ckpt: Path):
    """Match network cond_dim to checkpoint (2=linear IE1, 4=cyclic E2)."""
    sd = torch.load(ckpt, map_location="cpu")
    key = next(k for k in sd if k.endswith("fc.0.weight"))
    cond_dim = int(sd[key].shape[1])
    _purge_networks()
    # Prefer EXP for losses/warp; put network source first.
    sys.path = [p for p in sys.path if Path(p).resolve() not in (EXP.resolve(), IE1.resolve())]
    if cond_dim == 4:
        sys.path.insert(0, str(IE1))
        sys.path.insert(0, str(EXP))  # cyclic
        phase_tag = "cyclic"
    else:
        sys.path.insert(0, str(EXP))
        sys.path.insert(0, str(IE1))  # linear (matches accidental E2 train)
        phase_tag = "linear"
    from networks.generator_crb import UNetCRB  # noqa: E402
    from networks.generator_crb_both import UNetCRBBoth  # noqa: E402
    from networks.generator_crb_dec import UNetCRBDecoder  # noqa: E402
    return {
        "encoder": UNetCRB,
        "decoder": UNetCRBDecoder,
        "both": UNetCRBBoth,
    }, sd, cond_dim, phase_tag


# losses/warp from EXP (or IE1); import after path seed
sys.path.insert(0, str(IE1))
sys.path.insert(0, str(EXP))
from losses.losses import DVFLoss  # noqa: E402
from utilities.warp import warp  # noqa: E402

ARCH_CHOICES = ("encoder", "decoder", "both")

DEFAULT_CKPTS = {
    "encoder": "EncoderCRB/weights/crb_enc_mse_cyclic_full_spare_generator.pth",
    "decoder": "DecoderCRB/weights/crb_dec_mse_cyclic_full_spare_generator.pth",
    "both": "BothCRB/weights/crb_both_mse_cyclic_full_spare_generator.pth",
}

DEFAULT_PAIRS = [
    "01_to_02", "01_to_06", "01_to_10",
    "06_to_01", "06_to_06", "03_to_07", "05_to_08",
]


def norm_ct(x):
    x = x.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi <= lo else (x - lo) / (hi - lo)


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
    if m.any() and gt.max() > 0:
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
    for ax, im, ttl, cmap, vmin, vmax in panels:
        kw = dict(cmap=cmap, origin="upper", aspect="equal")
        if vmin is not None:
            kw["vmin"] = vmin
        if vmax is not None:
            kw["vmax"] = vmax
        elif im is img_err[z] or im is err[z]:
            pos = im[im > 0]
            kw["vmax"] = float(np.percentile(pos, 99)) if pos.size else 1.0
        h = ax.imshow(im, **kw)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="decoder", choices=list(ARCH_CHOICES))
    ap.add_argument("--scan", default="CV_P1_V_01")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--pairs", default=",".join(DEFAULT_PAIRS))
    ap.add_argument("--all_directed", action="store_true", help="All 90 directed pairs (metrics + panels)")
    ap.add_argument("--all", action="store_true", help="All 100 pairs incl. identity (metrics + panels)")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    data_dir = Path(args.data_root) if args.data_root else next(
        (p for p in (EXP / "data" / args.scan / "all", E1 / "data" / args.scan / "all") if p.is_dir()),
        EXP / "data" / args.scan / "all",
    )
    # Parse patient from CV_P1_V_01 → P1
    m = re.match(r"CV_(P\d+)_", args.scan)
    patient = m.group(1) if m else "unknown"
    default_out = (
        EXP
        / {"encoder": "EncoderCRB", "decoder": "DecoderCRB", "both": "BothCRB"}[args.arch]
        / "plots"
        / "qc_varian"
        / patient
        / args.scan
    )
    out_dir = Path(args.out_dir or default_out)
    ckpt = EXP / (args.ckpt or DEFAULT_CKPTS[args.arch])
    if not ckpt.is_file():
        raise SystemExit(f"missing ckpt: {ckpt}")

    ARCH, state, cond_dim, phase_tag = load_arch_classes(ckpt)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    g = ARCH[args.arch](im_size=128, n_phases=10).to(device)
    g.load_state_dict(state)
    g.eval()
    dvf_l1 = DVFLoss()
    print(f"[ckpt] cond_dim={cond_dim} phase={phase_tag}", flush=True)

    mask = (np.load(data_dir / "Mask_Lung.npy") > 0).astype(np.float32)
    z = lung_mid_z(mask)

    if args.all:
        pair_stems = [f"{i:02d}_to_{j:02d}" for i in range(1, 11) for j in range(1, 11)]
        panel_pairs = set(pair_stems)
    elif args.all_directed:
        pair_stems = [f"{i:02d}_to_{j:02d}" for i in range(1, 11) for j in range(1, 11) if i != j]
        panel_pairs = set(pair_stems)
    else:
        pair_stems = [p.strip() for p in args.pairs.split(",") if p.strip()]
        panel_pairs = set(pair_stems)

    print(
        f"[{args.arch}] scan={args.scan} pairs={len(pair_stems)} panels={len(panel_pairs)} "
        f"ckpt={ckpt.name} out={out_dir}",
        flush=True,
    )

    rows = []
    with torch.no_grad():
        for i, stem in enumerate(pair_stems, 1):
            ref, tgt = (int(stem[:2]), int(stem[-2:]))
            ct_r = norm_ct(np.load(data_dir / f"CT_{ref:02d}.npy"))
            ct_t = norm_ct(np.load(data_dir / f"CT_{tgt:02d}.npy"))
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
            print(
                f"[{i:03d}/{len(pair_stems)}] {stem}  L1={l1:.3f}  L1/zero={ratio:.2f}  "
                f"cos={cos:.2f}  k={k:.2f}",
                flush=True,
            )

            if stem in panel_pairs:
                save_panel(
                    out_dir / f"{stem}.png",
                    ct_r, ct_t, warped, gt, pred, mask, z,
                    f"{args.scan} {stem} (zero-shot {args.arch} {phase_tag})",
                    l1, cos, k,
                )

    directed = [r for r in rows if r["ref"] != r["tgt"]]
    summary = {
        "scan": args.scan,
        "arch": args.arch,
        "phase": phase_tag,
        "cond_dim": cond_dim,
        "ckpt": str(ckpt),
        "n_pairs": len(rows),
        "n_directed": len(directed),
        "mean_L1": float(np.mean([r["l1"] for r in directed])) if directed else float("nan"),
        "mean_L1_over_zero": float(np.mean([r["ratio"] for r in directed])) if directed else float("nan"),
        "mean_cos": float(np.nanmean([r["cos"] for r in directed])) if directed else float("nan"),
        "mean_k": float(np.nanmean([r["k"] for r in directed])) if directed else float("nan"),
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
        f"\n[{args.arch}] Directed mean: L1={summary['mean_L1']:.3f}  "
        f"L1/zero={summary['mean_L1_over_zero']:.2f}  "
        f"cos={summary['mean_cos']:.2f}  k={summary['mean_k']:.2f}  "
        f"beat_zero={summary['beat_zero_pct']:.0f}%",
        flush=True,
    )
    print(f"wrote {out_dir} ({len(list(out_dir.glob('*.png')))} pngs)", flush=True)


if __name__ == "__main__":
    main()
