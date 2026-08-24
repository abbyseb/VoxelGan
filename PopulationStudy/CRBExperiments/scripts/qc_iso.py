#!/usr/bin/env python3
"""Hold-out QC for CRBExperiments E2/E3/E4 on data_iso (2 mm).

  cd PopulationStudy/CRBExperiments/Experiment2
  PYTHONPATH=. python ../scripts/qc_iso.py --exp 2 --gpu 0 --panels all

  Or from ExperimentN with local scripts/qc_pairs.py wrappers.
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

CRB = Path(__file__).resolve().parents[1]
SPACING_MM = 2.0

SPECS = {
    2: {
        "name": "CRB-E2-A1-normonly",
        "ckpt": "crb_dec_a1_normonly_e2_generator.pth",
        "cond_dim": 3,
        "mode": "a1",  # log_r=0, scale=A_train
    },
    3: {
        "name": "CRB-E3-cyclic-oracle",
        "ckpt": "crb_dec_cyclic_oracle_e3_generator.pth",
        "cond_dim": 5,
        "mode": "oracle",
    },
    4: {
        "name": "CRB-E4-delta-cyclic-oracle",
        "ckpt": "crb_dec_delta_cyclic_oracle_e4_generator.pth",
        "cond_dim": 6,
        "mode": "delta",
    },
}


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
    from utilities.view_config import show_ct_slice, show_mag_slice

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
        (axs[0, 0], r2, "ref CT", "gray", None, None, "norm"),
        (axs[0, 1], t2, "target CT", "gray", None, None, "norm"),
        (axs[0, 2], w2, "warp(ref, pred)", "gray", None, None, "norm"),
        (axs[0, 3], e2, "|target − warp| · lung", "hot", 0, vmax_img, "|ΔI|"),
        (axs[1, 0], gt_mag, "|Elastix|", "magma", 0, vmax_mag, "|u| mm"),
        (axs[1, 1], pr_mag, "|pred|", "magma", 0, vmax_mag, "|u| mm"),
        (axs[1, 2], err_mag, "|pred − Elastix| · lung", "hot", 0, vmax_err, "|Δu| mm"),
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
    fig.suptitle(f"{title} | L1={l1_mm:.3f} mm cos={cos:.3f} | data_iso 2 mm")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=int, required=True, choices=[2, 3, 4])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--panels", default="all", choices=["all", "extreme", "none"])
    ap.add_argument("--max_pairs", type=int, default=0)
    args = ap.parse_args()

    spec = SPECS[args.exp]
    E = CRB / f"Experiment{args.exp}"
    sys.path.insert(0, str(E))

    from losses import losses as loss_mod
    from networks.generator_crb_dec_amp import UNetCRBDecoderAmp
    from utilities.amplitude import load_amplitude, patient_amp
    from utilities.view_config import load_view_config
    from utilities.warp import warp

    man = json.loads((E / "data" / "pooled" / "manifest.json").read_text())
    # use local pooled dirs
    data_dir = E / "data" / "pooled" / "test"
    pair_names = man["test_pairs"]
    if args.max_pairs > 0:
        pair_names = pair_names[: args.max_pairs]

    amp_table = load_amplitude(E / "amplitudes.json")
    A_train = float(amp_table["A_train_mean"])
    limbs = None
    if spec["mode"] == "delta":
        limbs = json.loads((E / "phase_limbs.json").read_text())["patients"]

    base_cfg = load_view_config(E / "configs" / "dvf_view_config.json")
    out_dir = E / "DecoderCRB" / "plots" / "qc_holdout"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    g = UNetCRBDecoderAmp(im_size=64, n_phases=10, cond_dim=spec["cond_dim"]).to(device)
    ckpt = E / "DecoderCRB" / "weights" / spec["ckpt"]
    g.load_state_dict(torch.load(ckpt, map_location=device))
    g.eval()
    dvf_l1 = loss_mod.DVFLoss()

    print(
        f"{spec['name']} QC pairs={len(pair_names)} scale="
        f"{'A_train' if spec['mode']=='a1' else 'A_p'} out={out_dir}",
        flush=True,
    )

    rows = []
    mask_cache = {}
    amp_cache = {}

    def forward(ref_t, ref, tgt, patient, A_p, log_r):
        rp = torch.tensor([ref - 1], device=device)
        tp = torch.tensor([tgt - 1], device=device)
        if spec["mode"] == "a1":
            lr = torch.zeros(1, device=device)
            shape = g(ref_t, rp, tp, lr)
            return shape * A_train
        if spec["mode"] == "oracle":
            lr = torch.tensor([log_r], dtype=torch.float32, device=device)
            shape = g(ref_t, rp, tp, lr)
            return shape * A_p
        # delta
        pid = f"P{int(patient[1:])}"
        dlt = float(limbs[pid]["delta_tgt"][tgt - 1])
        lr = torch.tensor([log_r], dtype=torch.float32, device=device)
        dd = torch.tensor([dlt], dtype=torch.float32, device=device)
        shape = g(ref_t, rp, tp, lr, dd)
        return shape * A_p

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
            pred_t = forward(ref_t, ref, tgt, patient, A_p, log_r)
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
                k = float(mag_lung(pred, mask).mean() / (mag_lung(gt, mask).mean() + 1e-8))
            else:
                cos = float("nan")
                k = float("nan")

            tag = f"{patient}_{ref:02d}_to_{tgt:02d}"
            note = "identity" if ref == tgt else "pair"
            scale_used = A_train if spec["mode"] == "a1" else A_p
            rows.append((tag, note, patient, l1_mm, zero_mm, ratio, cos, k, scale_used, log_r))
            print(
                f"[{i:03d}/{len(pair_names)}] {tag} L1={l1_mm:.3f}mm "
                f"L1/zero={ratio:.3f} cos={cos:.3f} k={k:.3f}",
                flush=True,
            )
            if args.panels == "all":
                save_panel(
                    out_dir / f"{tag}.png",
                    ct_r, ct_t, warped, abs_img_err, gt, pred, mask, cfg,
                    f"{tag} ({note})", l1_mm, 0.0 if cos != cos else cos,
                )

    with open(out_dir / "metrics.tsv", "w") as f:
        f.write("pair\tnote\tpatient\tL1_mm\tL1_zero_mm\tL1_over_zero\tcos\tk\tscale\tlog_r_p\n")
        for tag, note, patient, l1, zero, ratio, cos, k, sc, log_r in rows:
            f.write(
                f"{tag}\t{note}\t{patient}\t{l1:.6f}\t{zero:.6f}\t{ratio:.6f}\t"
                f"{cos:.6f}\t{k:.6f}\t{sc:.6f}\t{log_r:.6f}\n"
            )

    directed = [r for r in rows if r[1] != "identity"]
    summary = {
        "experiment": spec["name"],
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
        f"cos={summary['mean_cos']:.4f}  k={summary['mean_k']:.4f}"
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
            f"cos={entry['cos']:.3f} k={entry['k']:.3f}"
        )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
