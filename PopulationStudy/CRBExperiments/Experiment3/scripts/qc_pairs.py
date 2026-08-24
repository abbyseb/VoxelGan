"""Hold-out QC for CRBExperiments Experiment 1 (oracle amplitude on data_iso).

Network predicts shape û; QC scales pred → û · A_p (iso-voxels) then reports mm.

  PYTHONPATH=. python scripts/qc_pairs.py \\
    --ckpt DecoderCRB/weights/crb_dec_amp_oracle_e1_generator.pth \\
    --out_dir DecoderCRB/plots/qc_holdout \\
    --gpu 0
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

E1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E1))

from losses import losses as loss_mod
from networks.generator_crb_dec_amp import UNetCRBDecoderAmp
from utilities.amplitude import load_amplitude, patient_amp
from utilities.view_config import load_view_config, show_ct_slice, show_mag_slice
from utilities.warp import warp

SPACING_MM = 2.0


def _norm_ct(x):
    x = x.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _to_cdhw(dvf):
    if dvf.ndim == 4 and dvf.shape[-1] == 3:
        return np.moveaxis(dvf, -1, 0)
    return dvf


def lung_mid_z(mask: np.ndarray) -> int:
    zs = np.where(mask.any(axis=(1, 2)))[0]
    if zs.size == 0:
        return mask.shape[0] // 2
    return int(zs[len(zs) // 2])


def parse_pair(name: str):
    stem = name.replace("_pair.npy", "")
    patient, rest = stem.split("_", 1)
    a, b = rest.split("_to_")
    return patient, int(a), int(b)


def mag_lung(u_cdhw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.linalg.norm(u_cdhw, axis=0)[mask > 0.5]


def save_panel(out_path, ct_r, ct_t, warped, abs_img_err, gt, pred, mask, cfg, title, l1_mm, cos):
    gt_mm = gt * SPACING_MM
    pred_mm = pred * SPACING_MM
    r2 = show_ct_slice(ct_r, cfg)
    t2 = show_ct_slice(ct_t, cfg)
    w2 = show_ct_slice(warped, cfg)
    e2 = show_ct_slice(abs_img_err, cfg)
    gt_mag = show_mag_slice(gt_mm, cfg)
    pr_mag = show_mag_slice(pred_mm, cfg)
    err_mag = show_ct_slice(np.linalg.norm(pred_mm - gt_mm, axis=0) * mask, cfg)

    vmax_mag = float(np.percentile(np.concatenate([gt_mag.ravel(), pr_mag.ravel()]), 99))
    vmax_mag = max(vmax_mag, 1e-3)
    pos_img = e2[e2 > 0]
    vmax_img = float(np.percentile(pos_img, 99)) if pos_img.size else 1.0
    vmax_img = max(vmax_img, 1e-3)
    pos = err_mag[err_mag > 0]
    vmax_err = float(np.percentile(pos, 99)) if pos.size else 1.0
    vmax_err = max(vmax_err, 1e-3)

    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    for ax, im, ttl, cmap, vmin, vmax, cbl in [
        (axs[0, 0], r2, "ref CT (original)", "gray", None, None, "norm. HU"),
        (axs[0, 1], t2, "target CT (original)", "gray", None, None, "norm. HU"),
        (axs[0, 2], w2, "warp(ref, pred·A_p)", "gray", None, None, "norm. HU"),
        (axs[0, 3], e2, "|target − warp| · lung", "hot", 0, vmax_img, "|ΔI|"),
        (axs[1, 0], gt_mag, "|Elastix DVF|", "magma", 0, vmax_mag, "|u| (mm)"),
        (axs[1, 1], pr_mag, "|pred·A_p|", "magma", 0, vmax_mag, "|u| (mm)"),
        (axs[1, 2], err_mag, "|pred − Elastix| · lung", "hot", 0, vmax_err, "|Δu| (mm)"),
    ]:
        kw = dict(cmap=cmap, origin="upper", aspect="equal")
        if vmin is not None:
            kw["vmin"] = vmin
            kw["vmax"] = vmax
        h = ax.imshow(im, **kw)
        ax.set_title(ttl)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(h, ax=ax, fraction=0.046, pad=0.04).set_label(cbl)
    axs[1, 3].axis("off")
    fig.suptitle(
        f"{title}  |  lung mid-Z {cfg.slice_index}  |  "
        f"L1={l1_mm:.3f} mm  cos={cos:.3f}  |  oracle amp · data_iso 2 mm"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--view_config", default=str(E1 / "configs" / "dvf_view_config.json"))
    ap.add_argument("--max_pairs", type=int, default=0)
    ap.add_argument("--panels", default="extreme", choices=["all", "extreme", "none"])
    args = ap.parse_args()

    man = json.loads((E1 / "data" / "pooled" / "manifest.json").read_text())
    amp_table = load_amplitude()
    if args.split == "test":
        data_dir = Path(man["pooled_test_dir"])
        pair_names = man["test_pairs"]
    else:
        data_dir = Path(man["pooled_train_dir"])
        pair_names = man["val_pairs"] if args.split == "val" else man["train_pairs"]
    if args.max_pairs > 0:
        pair_names = pair_names[: args.max_pairs]

    base_cfg = load_view_config(args.view_config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    g = UNetCRBDecoderAmp(im_size=64, n_phases=10, cond_dim=3).to(device)
    g.load_state_dict(torch.load(args.ckpt, map_location=device))
    g.eval()
    dvf_l1 = loss_mod.DVFLoss()

    print(
        f"CRB-E1 QC oracle-amp split={args.split} pairs={len(pair_names)} "
        f"spacing={SPACING_MM}mm out={out_dir}",
        flush=True,
    )

    rows = []
    mask_cache = {}
    amp_cache = {}
    best_mag = {}

    with torch.no_grad():
        for i, name in enumerate(pair_names, 1):
            patient, ref, tgt = parse_pair(name)
            if patient not in mask_cache:
                mask_cache[patient] = (
                    np.load(data_dir / f"{patient}_Mask_Lung.npy") > 0
                ).astype(np.float32)
                A_p, log_r = patient_amp(amp_table, patient)
                amp_cache[patient] = (A_p, log_r)
            mask = mask_cache[patient]
            A_p, log_r = amp_cache[patient]
            cfg = deepcopy(base_cfg)
            cfg.slice_index = lung_mid_z(mask)

            ct_r = _norm_ct(np.load(data_dir / f"{patient}_CT_{ref:02d}.npy"))
            ct_t = _norm_ct(np.load(data_dir / f"{patient}_CT_{tgt:02d}.npy"))
            if ref == tgt:
                gt = np.zeros((3,) + ct_r.shape, dtype=np.float32)
            else:
                gt = _to_cdhw(np.load(data_dir / name).astype(np.float32))

            ref_t = torch.from_numpy(ct_r)[None, None].to(device)
            mask_t = torch.from_numpy(mask)[None, None].to(device)
            gt_t = torch.from_numpy(gt)[None].to(device)
            shape_t = g(
                ref_t,
                torch.tensor([ref - 1], device=device),
                torch.tensor([tgt - 1], device=device),
                torch.tensor([log_r], dtype=torch.float32, device=device),
            )
            pred_t = shape_t * A_p  # iso-voxels
            pred = pred_t[0].cpu().numpy()
            warped = warp(ref_t, pred_t)[0, 0].cpu().numpy()
            abs_img_err = np.abs(ct_t - warped) * mask

            l1_vx = float(dvf_l1.loss(gt_t, pred_t, mask_t).item())
            zero_vx = float(dvf_l1.loss(gt_t, torch.zeros_like(gt_t), mask_t).item())
            l1_mm = l1_vx * SPACING_MM
            zero_mm = zero_vx * SPACING_MM
            ratio = l1_mm / max(zero_mm, 1e-8)

            if ref != tgt:
                m = mask > 0.5
                a, b = pred * m[None], gt * m[None]
                cos = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-8))
                mg = mag_lung(gt, mask)
                mp = mag_lung(pred, mask)
                k = float(mp.mean() / (mg.mean() + 1e-8))
                mean_gt_mm = float(mg.mean() * SPACING_MM)
                if mean_gt_mm > best_mag.get(patient, (0.0, ""))[0]:
                    best_mag[patient] = (mean_gt_mm, name)
            else:
                cos = float("nan")
                k = float("nan")

            tag = f"{patient}_{ref:02d}_to_{tgt:02d}"
            note = "identity" if ref == tgt else "pair"
            rows.append((tag, note, patient, l1_mm, zero_mm, ratio, cos, k, A_p, log_r))
            print(
                f"[{i:03d}/{len(pair_names)}] {tag} L1={l1_mm:.3f}mm "
                f"L1/zero={ratio:.3f} cos={cos:.3f} k={k:.3f} A_p={A_p:.2f}",
                flush=True,
            )

            if args.panels == "all":
                save_panel(
                    out_dir / f"{tag}.png",
                    ct_r, ct_t, warped, abs_img_err, gt, pred, mask, cfg,
                    f"{tag} ({note})", l1_mm, 0.0 if cos != cos else cos,
                )

    if args.panels == "extreme":
        for patient, (_, name) in sorted(best_mag.items()):
            patient, ref, tgt = parse_pair(name)
            mask = mask_cache[patient]
            A_p, log_r = amp_cache[patient]
            cfg = deepcopy(base_cfg)
            cfg.slice_index = lung_mid_z(mask)
            ct_r = _norm_ct(np.load(data_dir / f"{patient}_CT_{ref:02d}.npy"))
            ct_t = _norm_ct(np.load(data_dir / f"{patient}_CT_{tgt:02d}.npy"))
            gt = _to_cdhw(np.load(data_dir / name).astype(np.float32))
            ref_t = torch.from_numpy(ct_r)[None, None].to(device)
            with torch.no_grad():
                shape_t = g(
                    ref_t,
                    torch.tensor([ref - 1], device=device),
                    torch.tensor([tgt - 1], device=device),
                    torch.tensor([log_r], dtype=torch.float32, device=device),
                )
                pred_t = shape_t * A_p
            pred = pred_t[0].cpu().numpy()
            warped = warp(ref_t, pred_t)[0, 0].cpu().numpy()
            abs_img_err = np.abs(ct_t - warped) * mask
            row = next(r for r in rows if r[0] == f"{patient}_{ref:02d}_to_{tgt:02d}")
            save_panel(
                out_dir / f"{row[0]}.png",
                ct_r, ct_t, warped, abs_img_err, gt, pred, mask, cfg,
                f"{row[0]} (extreme)", row[3], 0.0 if row[6] != row[6] else row[6],
            )
            print(f"panel extreme {row[0]}", flush=True)

    with open(out_dir / "metrics.tsv", "w") as f:
        f.write("pair\tnote\tpatient\tL1_mm\tL1_zero_mm\tL1_over_zero\tcos\tk\tA_p\tlog_r_p\n")
        for tag, note, patient, l1, zero, ratio, cos, k, A_p, log_r in rows:
            f.write(
                f"{tag}\t{note}\t{patient}\t{l1:.6f}\t{zero:.6f}\t{ratio:.6f}\t"
                f"{cos:.6f}\t{k:.6f}\t{A_p:.6f}\t{log_r:.6f}\n"
            )

    directed = [r for r in rows if r[1] != "identity"]
    summary = {
        "experiment": "CRB-E1-oracle-amp",
        "grid": "data_iso 2mm 160³",
        "spacing_mm": SPACING_MM,
        "n_directed": len(directed),
        "mean_L1_mm": float(np.mean([r[3] for r in directed])),
        "mean_L1_over_zero": float(np.mean([r[5] for r in directed])),
        "mean_cos": float(np.nanmean([r[6] for r in directed])),
        "mean_k": float(np.nanmean([r[7] for r in directed])),
        "per_patient": {},
    }
    print(
        f"directed mean L1={summary['mean_L1_mm']:.4f} mm  "
        f"L1/zero={summary['mean_L1_over_zero']:.4f}  "
        f"cos={summary['mean_cos']:.4f}  k={summary['mean_k']:.4f}  n={len(directed)}"
    )
    for patient in sorted({r[2] for r in directed}):
        sub = [r for r in directed if r[2] == patient]
        entry = {
            "L1_mm": float(np.mean([r[3] for r in sub])),
            "L1_over_zero": float(np.mean([r[5] for r in sub])),
            "cos": float(np.nanmean([r[6] for r in sub])),
            "k": float(np.nanmean([r[7] for r in sub])),
            "n": len(sub),
        }
        summary["per_patient"][patient] = entry
        print(
            f"  {patient} L1={entry['L1_mm']:.3f}mm L1/zero={entry['L1_over_zero']:.3f} "
            f"cos={entry['cos']:.3f} k={entry['k']:.3f} n={entry['n']}"
        )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
