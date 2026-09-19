#!/usr/bin/env python3
"""DIR-Lab QC for Iso-E2 full models on packed_iso (2 mm, 160³).

Requires pack_dirlab_iso.py + prepare_dir_dvf_library_iso.py first.
Metrics include L1 in mm (iso-voxels × 2 mm).

  cd PopulationStudy/IsoExperiments/Experiment2
  PYTHONPATH=. python scripts/qc_dirlab.py --arches encoder,decoder,both --norm minmax --no-panels --gpu 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

E2 = Path(__file__).resolve().parents[1]
DIR_E2 = E2.parents[1] / "DIR-Experiments" / "Experiment2"
PACKED_ISO = DIR_E2 / "packed_iso"
sys.path.insert(0, str(E2))

from losses.losses import DVFLoss  # noqa: E402
from networks.generator_crb import UNetCRB  # noqa: E402
from networks.generator_crb_both import UNetCRBBoth  # noqa: E402
from networks.generator_crb_dec import UNetCRBDecoder  # noqa: E402
from utilities.warp import warp  # noqa: E402

ARCH = {"encoder": UNetCRB, "decoder": UNetCRBDecoder, "both": UNetCRBBoth}
ARCH_DIR = {"encoder": "EncoderCRB", "decoder": "DecoderCRB", "both": "BothCRB"}
DEFAULT_CKPTS = {
    "encoder": E2 / "EncoderCRB/weights/crb_enc_mse_iso_e2_full_generator.pth",
    "decoder": E2 / "DecoderCRB/weights/crb_dec_mse_iso_e2_full_generator.pth",
    "both": E2 / "BothCRB/weights/crb_both_mse_iso_e2_full_generator.pth",
}
DEFAULT_PATIENTS = [f"P{i}_DIR" for i in range(1, 11)]
DEFAULT_PANEL_PAIRS = {"01_to_02", "01_to_06", "01_to_10", "06_to_01", "06_to_06", "03_to_07", "05_to_08"}
PLANE_SLICE_AXIS = {"coronal": 1, "axial": 0}
SPACING_MM = 2.0
IM_SIZE = 64


def norm_minmax(x):
    x = x.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi <= lo else (x - lo) / (hi - lo)


def to_cdhw(dvf):
    if dvf.ndim == 4 and dvf.shape[-1] == 3:
        return np.moveaxis(dvf, -1, 0)
    return dvf


def lung_mid_index(mask: np.ndarray, slice_axis: int) -> int:
    other = tuple(i for i in range(3) if i != slice_axis)
    idx = np.where(mask.any(axis=other))[0]
    return int(idx[len(idx) // 2]) if idx.size else mask.shape[slice_axis] // 2


def slice_2d(vol: np.ndarray, slice_axis: int, index: int) -> np.ndarray:
    if slice_axis == 0:
        return vol[index]
    if slice_axis == 1:
        return vol[:, index, :]
    return vol[:, :, index]


def display_coronal(im):
    """Match clinical mid-AP coronal orientation (SI×LR, diaphragm view)."""
    return np.rot90(im, 2)


def display_axial(im):
    return np.rot90(im, 2)


def metrics(pred, gt, mask, dvf_l1):
    m = mask > 0.5
    gt_t = torch.from_numpy(gt)[None].float()
    pr_t = torch.from_numpy(pred)[None].float()
    mk_t = torch.from_numpy(mask)[None, None].float()
    l1_vx = float(dvf_l1.loss(gt_t, pr_t, mk_t).item())
    l1z_vx = float(dvf_l1.loss(gt_t, torch.zeros_like(gt_t), mk_t).item())
    l1_mm = l1_vx * SPACING_MM
    l1z_mm = l1z_vx * SPACING_MM
    if m.any() and float(np.abs(gt).max()) > 0:
        a, b = pred[:, m], gt[:, m]
        cos = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-8))
        k = float(np.linalg.norm(pred[:, m]) / (np.linalg.norm(gt[:, m]) + 1e-8))
    else:
        cos, k = float("nan"), float("nan")
    return l1_vx, l1z_vx, l1_mm, l1z_mm, l1_mm / (l1z_mm + 1e-8), cos, k


def save_panel(out, ct_r, ct_t, warped, gt, pred, mask, slice_axis, slice_idx, title, l1_mm, cos, k, plane):
    gt_mm = gt * SPACING_MM
    pred_mm = pred * SPACING_MM
    gt_mag = np.linalg.norm(gt_mm, axis=0)
    pr_mag = np.linalg.norm(pred_mm, axis=0)
    err = np.linalg.norm(pred_mm - gt_mm, axis=0) * mask
    img_err = np.abs(ct_t - warped) * mask
    vmax = max(float(np.percentile(np.concatenate([gt_mag[mask > 0], pr_mag[mask > 0]]), 99)), 1e-3)
    show = display_coronal if plane == "coronal" else display_axial
    fig, axs = plt.subplots(2, 4, figsize=(14, 7))
    panels = [
        (axs[0, 0], slice_2d(ct_r, slice_axis, slice_idx), "ref CT", "gray", 0, 1),
        (axs[0, 1], slice_2d(ct_t, slice_axis, slice_idx), "target CT", "gray", 0, 1),
        (axs[0, 2], slice_2d(warped, slice_axis, slice_idx), "warp(ref,pred)", "gray", 0, 1),
        (axs[0, 3], slice_2d(img_err, slice_axis, slice_idx), "|tgt-warp|·lung", "hot", 0, None),
        (axs[1, 0], slice_2d(gt_mag, slice_axis, slice_idx), "|Elastix| mm", "magma", 0, vmax),
        (axs[1, 1], slice_2d(pr_mag, slice_axis, slice_idx), "|pred| mm", "magma", 0, vmax),
        (axs[1, 2], slice_2d(err, slice_axis, slice_idx), "|pred-GT|·lung mm", "hot", 0, None),
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
        h = ax.imshow(show(im), **kw)
        ax.set_title(ttl, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(h, ax=ax, fraction=0.046, pad=0.04)
    axs[1, 3].axis("off")
    ax_names = {0: "z", 1: "y", 2: "x"}
    fig.suptitle(
        f"{title}  L1={l1_mm:.3f} mm  cos={cos:.2f}  k={k:.2f}  "
        f"{plane} {ax_names[slice_axis]}={slice_idx}",
        fontsize=10,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def qc_panel_root(arch: str, norm_name: str, plane: str) -> Path:
    sub = f"qc_dir_{norm_name}" + (f"_{plane}" if plane == "coronal" else "")
    return E2 / ARCH_DIR[arch] / "plots" / sub


def run_one(arch, pid, g, device, dvf_l1, norm_fn, norm_name, panel_pairs, no_panels, force, packed_root, plane):
    data_dir = packed_root / pid / "all"
    out_dir = qc_panel_root(arch, norm_name, plane) / pid
    metrics_dir = E2 / ARCH_DIR[arch] / "plots" / f"qc_dir_{norm_name}" / pid
    if (metrics_dir / "summary.json").is_file() and not force:
        print(f"skip {arch} {pid} {norm_name} (exists)", flush=True)
        return json.loads((metrics_dir / "summary.json").read_text())

    pair0 = data_dir / "01_to_02_pair.npy"
    if not pair0.is_file():
        raise FileNotFoundError(f"missing {pair0} — run prepare_dir_dvf_library_iso.py for {pid}")

    mask = (np.load(data_dir / "Mask_Lung.npy") > 0).astype(np.float32)
    slice_axis = PLANE_SLICE_AXIS[plane]
    slice_idx = lung_mid_index(mask, slice_axis)
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
            l1_vx, l1z_vx, l1_mm, l1z_mm, ratio, cos, k = metrics(pred, gt, mask, dvf_l1)
            rows.append(dict(
                pair=stem, ref=ref, tgt=tgt,
                l1_vx=l1_vx, l1_zero_vx=l1z_vx, l1_mm=l1_mm, l1_zero_mm=l1z_mm,
                ratio=ratio, cos=cos, k=k,
            ))
            if not no_panels and stem in panel_pairs:
                save_panel(
                    out_dir / f"{stem}.png",
                    ct_r, ct_t, warped, gt, pred, mask, slice_axis, slice_idx,
                    f"{pid} {stem} (Iso-E2 full {arch} · {norm_name})",
                    l1_mm, cos, k, plane,
                )
            if i % 25 == 0:
                print(f"  [{arch}|{pid}|{norm_name}] {i}/100", flush=True)

    directed = [r for r in rows if r["ref"] != r["tgt"]]
    summary = {
        "patient": pid,
        "arch": arch,
        "norm": norm_name,
        "experiment": "Iso-E2_full_DIR_iso",
        "grid": f"iso {SPACING_MM}mm 160³",
        "ckpt": str(DEFAULT_CKPTS[arch]),
        "n_pairs": len(rows),
        "n_directed": len(directed),
        "mean_L1_mm": float(np.mean([r["l1_mm"] for r in directed])),
        "mean_L1_over_zero": float(np.mean([r["ratio"] for r in directed])),
        "mean_cos": float(np.nanmean([r["cos"] for r in directed])),
        "mean_k": float(np.nanmean([r["k"] for r in directed])),
        "beat_zero_pct": float(100.0 * sum(r["ratio"] < 1.0 for r in directed) / max(len(directed), 1)),
        "pairs": rows,
    }
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with open(metrics_dir / "metrics.tsv", "w") as f:
        f.write("pair\tnote\tL1_mm\tL1_zero_mm\tL1_over_zero\tcos\tk\n")
        for r in rows:
            note = "identity" if r["ref"] == r["tgt"] else "pair"
            f.write(
                f"{r['pair']}\t{note}\t{r['l1_mm']:.6f}\t{r['l1_zero_mm']:.6f}\t"
                f"{r['ratio']:.6f}\t{r['cos']}\t{r['k']}\n"
            )
    print(
        f"[{arch}|{pid}|{norm_name}] L1={summary['mean_L1_mm']:.3f}mm cos={summary['mean_cos']:.3f} "
        f"beat={summary['beat_zero_pct']:.0f}% → {metrics_dir}",
        flush=True,
    )
    if not no_panels:
        print(f"  panels ({plane}) → {out_dir}", flush=True)
    return summary


def run_panels_only(arch, pid, g, device, dvf_l1, norm_fn, norm_name, panel_pairs, force, packed_root, plane):
    data_dir = packed_root / pid / "all"
    out_dir = qc_panel_root(arch, norm_name, plane) / pid
    summary_path = E2 / ARCH_DIR[arch] / "plots" / f"qc_dir_{norm_name}" / pid / "summary.json"
    if not summary_path.is_file():
        print(f"WARN no summary for {arch} {pid} — run full QC first", flush=True)
        return None
    summary = json.loads(summary_path.read_text())
    pair_metrics = {r["pair"]: r for r in summary.get("pairs", [])}

    mask = (np.load(data_dir / "Mask_Lung.npy") > 0).astype(np.float32)
    slice_axis = PLANE_SLICE_AXIS[plane]
    slice_idx = lung_mid_index(mask, slice_axis)
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
                l1_mm, cos, k = pm["l1_mm"], pm["cos"], pm["k"]
            else:
                _, _, l1_mm, _, _, cos, k = metrics(pred, gt, mask, dvf_l1)
            save_panel(
                png,
                ct_r, ct_t, warped, gt, pred, mask, slice_axis, slice_idx,
                f"{pid} {stem} (Iso-E2 full {arch} · {norm_name})",
                l1_mm, cos, k, plane,
            )
            n_saved += 1
    print(f"[{arch}|{pid}|{norm_name}] panels saved={n_saved} → {out_dir}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--arches", default="encoder,decoder,both")
    ap.add_argument("--norm", default="minmax", choices=("minmax",))
    ap.add_argument("--patients", default=",".join(DEFAULT_PATIENTS))
    ap.add_argument("--packed-root", type=Path, default=PACKED_ISO)
    ap.add_argument("--no-panels", action="store_true")
    ap.add_argument("--panels-only", action="store_true", help="write PNGs from existing metrics")
    ap.add_argument("--plane", default="coronal", choices=("coronal", "axial"), help="panel slice plane")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    arches = [a.strip() for a in args.arches.split(",") if a.strip()]
    patients = [p.strip() for p in args.patients.split(",") if p.strip()]
    packed_root = args.packed_root

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dvf_l1 = DVFLoss()
    print(
        f"Iso-E2 DIR QC  device={device}  packed={packed_root}  arches={arches}"
        f"  plane={args.plane}  panels_only={args.panels_only}",
        flush=True,
    )

    all_summ = {}
    lines = ["experiment\tarch\tnorm\tpatient\tL1_mm\tcos\tL1_over_zero\tbeat"]
    for arch in arches:
        ckpt = DEFAULT_CKPTS[arch]
        if not ckpt.is_file():
            raise SystemExit(f"missing ckpt: {ckpt}")
        g = ARCH[arch](im_size=IM_SIZE, n_phases=10).to(device)
        g.load_state_dict(torch.load(ckpt, map_location=device))
        g.eval()
        print(f"loaded {arch} {ckpt.name}", flush=True)
        all_summ[arch] = {}
        for pid in patients:
            if not (packed_root / pid / "pack_meta.json").is_file():
                print(f"WARN skip {pid}: not packed at {packed_root}", flush=True)
                continue
            if args.panels_only:
                s = run_panels_only(
                    arch, pid, g, device, dvf_l1, norm_minmax, "minmax",
                    DEFAULT_PANEL_PAIRS, args.force, packed_root, args.plane,
                )
                if s is None:
                    continue
            else:
                s = run_one(
                    arch, pid, g, device, dvf_l1, norm_minmax, "minmax",
                    DEFAULT_PANEL_PAIRS, args.no_panels, args.force, packed_root, args.plane,
                )
            all_summ[arch][pid] = {
                k: s[k]
                for k in ("mean_L1_mm", "mean_L1_over_zero", "mean_cos", "mean_k", "beat_zero_pct")
            }
            lines.append(
                f"Iso-E2_DIR\t{arch}\tminmax\t{pid}\t{s['mean_L1_mm']:.4f}\t{s['mean_cos']:.4f}\t"
                f"{s['mean_L1_over_zero']:.4f}\t{s['beat_zero_pct']:.1f}"
            )

    if args.panels_only:
        print("DONE (panels only)", flush=True)
        return

    print("\n======== ISO-E2 DIR SUMMARY ========", flush=True)
    for arch in all_summ:
        subset = list(all_summ[arch].values())
        if not subset:
            continue
        l1 = float(np.mean([x["mean_L1_mm"] for x in subset]))
        cos = float(np.mean([x["mean_cos"] for x in subset]))
        beat = float(np.mean([x["beat_zero_pct"] for x in subset]))
        ratio = float(np.mean([x["mean_L1_over_zero"] for x in subset]))
        print(
            f"{arch:8s}  L1={l1:.3f} mm  cos={cos:.3f}  L1/z={ratio:.3f}  beat={beat:.0f}%  n={len(subset)}",
            flush=True,
        )

    out_root = E2 / "plots" / "qc_summary"
    out_root.mkdir(parents=True, exist_ok=True)
    tag = "_".join(arches) + "_dir_iso"
    (out_root / f"summary_{tag}.json").write_text(json.dumps(all_summ, indent=2) + "\n")
    (out_root / f"metrics_{tag}.tsv").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_root / f'summary_{tag}.json'}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
