#!/usr/bin/env python3
"""Multi-plane / multi-slice CT survey for one clinical scan (Elekta or Varian).

Packed volumes are (AP, SI, LR) = axes (0, 1, 2). Coronal = fixed AP (axis 0).

  python scripts/survey_clinical_scan_slices.py --scan CE_P3_V_01 --ref 1 --tgt 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if ROOT.name == "Grid128":
    ROOT = ROOT.parent
CE1 = ROOT / "Experiment1"
OUT_ROOT = ROOT / "plots" / "clinical_slice_survey"


def window(x, lo=None, hi=None):
    if lo is None:
        lo = float(np.percentile(x, 1))
    if hi is None:
        hi = float(np.percentile(x, 99))
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((x - lo) / (hi - lo), 0, 1)


def lung_bounds(mask: np.ndarray) -> dict[str, tuple[int, int, int]]:
    ap = np.where(mask.any(axis=(1, 2)))[0]
    si = np.where(mask.any(axis=(0, 2)))[0]
    lr = np.where(mask.any(axis=(0, 1)))[0]
    mid = lambda idx: (int(idx[0]), int(idx[-1]), int(idx[len(idx) // 2]))
    return {"ap": mid(ap), "si": mid(si), "lr": mid(lr)}


def slice_indices(lo: int, hi: int, mid: int, n: int = 5) -> list[int]:
    if hi <= lo:
        return [mid]
    qs = np.linspace(lo, hi, n)
    return sorted({int(round(v)) for v in qs} | {mid})


def show_coronal(im: np.ndarray) -> np.ndarray:
    return np.rot90(im, 2)


def show_axial(im: np.ndarray) -> np.ndarray:
    return np.rot90(im, 2)


def show_sagittal(im: np.ndarray) -> np.ndarray:
    return np.rot90(im, 1)


def plot_row(axs, ct_a, ct_b, mask, indices, slicer, shower, title_prefix):
    lo = float(np.percentile(np.concatenate([ct_a.ravel(), ct_b.ravel()]), 1))
    hi = float(np.percentile(np.concatenate([ct_a.ravel(), ct_b.ravel()]), 99))
    for ax, idx in zip(axs, indices):
        sl_a = window(slicer(ct_a, idx), lo, hi)
        sl_b = window(slicer(ct_b, idx), lo, hi)
        sl_m = shower(slicer(mask.astype(float), idx))
        rgb = np.stack([sl_a, sl_b, sl_a], axis=-1)
        rgb[..., 1] = np.maximum(rgb[..., 1], sl_m * 0.35)
        ax.imshow(rgb, origin="upper", aspect="equal")
        ax.contour(sl_m, levels=[0.5], colors="lime", linewidths=0.5)
        ax.set_title(f"{title_prefix}={idx}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="CE_P3_V_01")
    ap.add_argument("--ref", type=int, default=1)
    ap.add_argument("--tgt", type=int, default=6)
    ap.add_argument("--n-slices", type=int, default=5)
    args = ap.parse_args()

    data_dir = CE1 / "data" / args.scan / "all"
    ct_r = np.load(data_dir / f"CT_{args.ref:02d}.npy").astype(np.float32)
    ct_t = np.load(data_dir / f"CT_{args.tgt:02d}.npy").astype(np.float32)
    mask = (np.load(data_dir / "Mask_Lung.npy") > 0).astype(np.float32)
    b = lung_bounds(mask > 0)

    stem = f"{args.ref:02d}_to_{args.tgt:02d}"
    gt_path = data_dir / f"{stem}_pair.npy"
    gt_mag = None
    if gt_path.is_file() and args.ref != args.tgt:
        gt = np.load(gt_path).astype(np.float32)
        if gt.ndim == 4 and gt.shape[-1] == 3:
            gt = np.moveaxis(gt, -1, 0)
        gt_mag = np.linalg.norm(gt, axis=0)

    vendor = "elekta" if args.scan.startswith("CE_") else "varian"
    out_dir = OUT_ROOT / vendor / args.scan
    out_dir.mkdir(parents=True, exist_ok=True)

    ap_idxs = slice_indices(*b["ap"], args.n_slices)
    si_idxs = slice_indices(*b["si"], args.n_slices)
    lr_idxs = slice_indices(*b["lr"], args.n_slices)

    ncols = max(len(ap_idxs), len(si_idxs), len(lr_idxs))
    fig, axs = plt.subplots(4, ncols, figsize=(2.6 * ncols, 11))
    if ncols == 1:
        axs = axs[:, None]

    plot_row(
        axs[0, : len(ap_idxs)],
        ct_r, ct_t, mask, ap_idxs,
        lambda v, i: v[i], show_coronal, "AP",
    )
    plot_row(
        axs[1, : len(si_idxs)],
        ct_r, ct_t, mask, si_idxs,
        lambda v, i: v[:, i, :], show_axial, "SI",
    )
    plot_row(
        axs[2, : len(lr_idxs)],
        ct_r, ct_t, mask, lr_idxs,
        lambda v, i: v[:, :, i], show_sagittal, "LR",
    )

    if gt_mag is not None:
        vmax = float(np.percentile(gt_mag[mask > 0], 99))
        for ax, idx in zip(axs[3, : len(ap_idxs)], ap_idxs):
            sl = show_coronal(gt_mag[idx])
            m = show_coronal(mask[idx])
            h = ax.imshow(sl, cmap="magma", origin="upper", aspect="equal", vmin=0, vmax=max(vmax, 1e-3))
            ax.contour(m, levels=[0.5], colors="lime", linewidths=0.5)
            ax.set_title(f"|Elastix| AP={idx}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(h, ax=ax, fraction=0.046, pad=0.04)
    else:
        for ax in axs[3]:
            ax.axis("off")

    for c in range(ncols):
        if c >= len(ap_idxs):
            axs[0, c].axis("off")
        if c >= len(si_idxs):
            axs[1, c].axis("off")
        if c >= len(lr_idxs):
            axs[2, c].axis("off")
        if gt_mag is None or c >= len(ap_idxs):
            axs[3, c].axis("off")

    fig.text(0.01, 0.98, "RGB: ref=red, target=green, overlap=yellow · green contour=lung", fontsize=9, va="top")
    axs[0, 0].set_ylabel("coronal\n(AP)", fontsize=10, fontweight="bold")
    axs[1, 0].set_ylabel("axial\n(SI)", fontsize=10, fontweight="bold")
    axs[2, 0].set_ylabel("sagittal\n(LR)", fontsize=10, fontweight="bold")
    if gt_mag is not None:
        axs[3, 0].set_ylabel("|Elastix|\ncoronal", fontsize=10, fontweight="bold")

    fig.suptitle(
        f"{args.scan}  phase {args.ref:02d}→{args.tgt:02d}  ·  multi-slice survey\n"
        f"lung AP{b['ap']} SI{b['si']} LR{b['lr']}",
        fontsize=12,
    )
    fig.tight_layout()
    out = out_dir / f"{stem}_multislice.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
