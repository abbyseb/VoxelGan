#!/usr/bin/env python3
"""Per-case panel: 10-phase mid-coronal real vs synth vs diff, plus Elastix vs synth DVF.

R3/SPARE numpy ZYX = (AP, SI, LR). Coronal = mid-AP → SI×LR.

sub_CT / DVF_sub are full-FOV resamples to 128³ (spacing forced to 1). All panel
cells share one GridSpec size + the same physical mm extent so |u| matches
Real/Synth (no thin strips from CT voxel-aspect on 128×128).

  cd "DIR EXPERIMENTS"
  /home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python \\
    scripts/plot_a3_synth_panels.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from matplotlib.gridspec import GridSpec

DIR_EXP = Path(__file__).resolve().parents[1]
A1 = DIR_EXP / "arms" / "A1_oracle_dirlab" / "runs"
A3 = DIR_EXP / "arms" / "A3_synth_conditioned" / "runs"
OUT = DIR_EXP / "arms" / "A3_synth_conditioned" / "plots" / "synth_phase_panels" / "coronal"


def load_img(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = sitk.ReadImage(str(path))
    # spacing itk (LR, SI, AP); array (AP, SI, LR)
    return sitk.GetArrayFromImage(img).astype(np.float32), tuple(float(s) for s in img.GetSpacing())


def load_dvf_mag(path: Path, *, negate: bool = False) -> np.ndarray:
    a = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
    if negate:
        a = -a
    return np.linalg.norm(a, axis=-1)


def mid_coronal(vol: np.ndarray) -> np.ndarray:
    """Mid-AP slice → SI × LR (true coronal), rotated 180° for upright display."""
    return np.rot90(vol[vol.shape[0] // 2, :, :], 2)


def coronal_extent_mm(
    sl_si_lr: np.ndarray, spacing_lr_si_ap: tuple[float, float, float]
) -> tuple[float, float, float, float]:
    """Physical (left, right, bottom, top) for mid-coronal SI×LR using native CT spacing."""
    sp_lr, sp_si, _sp_ap = spacing_lr_si_ap
    n_si, n_lr = sl_si_lr.shape
    return (0.0, n_lr * sp_lr, 0.0, n_si * sp_si)


def hu_window(sl: np.ndarray, lo: float = -1000.0, hi: float = 500.0) -> np.ndarray:
    return np.clip(sl, lo, hi)


def _imshow(ax, sl: np.ndarray, *, extent, cmap, vmin, vmax):
    # aspect='auto' fills the fixed GridSpec cell; cells are sized to SI/LR mm ratio.
    return ax.imshow(
        sl,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        extent=extent,
        aspect="auto",
        interpolation="nearest",
    )


def plot_case(case: int, out_dir: Path) -> tuple[Path, Path]:
    sid = f"DIR_C{case:02d}"
    a1_train = A1 / sid / sid / "train"
    a3_train = A3 / sid / sid / "train"

    real01, spacing = load_img(a1_train / "CT_01.mha")
    real01 = hu_window(mid_coronal(real01))
    extent = coronal_extent_mm(real01, spacing)
    fov_lr = extent[1] - extent[0]
    fov_si = extent[3] - extent[2]
    box_asp = fov_si / fov_lr  # ~0.95 for DIR R3

    synth01 = hu_window(mid_coronal(load_img(a3_train / "CT_01.mha")[0]))
    vmax_diff = max(float(np.percentile(np.abs(synth01 - real01), 99)), 50.0)

    el_mag = mid_coronal(load_dvf_mag(a1_train / "DVF_sub_01.mha"))
    syn_raw = mid_coronal(load_dvf_mag(a3_train / "DVF_sub_01.mha", negate=False))
    syn_neg = mid_coronal(load_dvf_mag(a3_train / "DVF_sub_01.mha", negate=True))
    vmax_u = max(float(np.percentile(np.concatenate([el_mag.ravel(), syn_neg.ravel()]), 98)), 1.0)
    blank_u = np.zeros_like(el_mag)

    # Fixed cell size from physical FOV; extra column for colorbars (doesn't shrink rows).
    n_cols = 10
    cell_w = 1.85
    cell_h = cell_w * box_asp
    cbar_w = 0.22
    fig_w = n_cols * cell_w + cbar_w + 1.1
    fig_h = 4 * cell_h + 1.35
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        4,
        n_cols + 1,
        figure=fig,
        width_ratios=[1.0] * n_cols + [0.12],
        height_ratios=[1, 1, 1, 1],
        left=0.06,
        right=0.98,
        top=0.90,
        bottom=0.04,
        wspace=0.06,
        hspace=0.18,
    )
    axes = np.empty((4, n_cols), dtype=object)
    for i in range(4):
        for j in range(n_cols):
            axes[i, j] = fig.add_subplot(gs[i, j])
    cax_d = fig.add_subplot(gs[2, n_cols])
    cax_u = fig.add_subplot(gs[3, n_cols])

    fig.suptitle(
        f"{sid}  ·  mid-coronal mid-AP (SI×LR, shared mm FOV)  ·  real | synth | synth−real | −synth |u|",
        fontsize=12,
        fontweight="bold",
    )

    im_d = im_u = None
    for p in range(1, 11):
        j = p - 1
        real = hu_window(mid_coronal(load_img(a1_train / f"CT_{p:02d}.mha")[0]))
        synth = hu_window(mid_coronal(load_img(a3_train / f"CT_{p:02d}.mha")[0]))
        diff = synth - real

        _imshow(axes[0, j], real, extent=extent, cmap="gray", vmin=-1000, vmax=500)
        _imshow(axes[1, j], synth, extent=extent, cmap="gray", vmin=-1000, vmax=500)
        im_d = _imshow(axes[2, j], diff, extent=extent, cmap="coolwarm", vmin=-vmax_diff, vmax=vmax_diff)

        if p == 6:
            im_u = _imshow(axes[3, j], blank_u, extent=extent, cmap="magma", vmin=0, vmax=vmax_u)
        else:
            mag = mid_coronal(load_dvf_mag(a3_train / f"DVF_sub_{p:02d}.mha", negate=True))
            im_u = _imshow(axes[3, j], mag, extent=extent, cmap="magma", vmin=0, vmax=vmax_u)

        axes[0, j].set_title(f"φ{p:02d}", fontsize=9)
        for i in range(4):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    for i, lab in enumerate(["Real CT", "Synth CT", "Synth − Real", "−synth |u|"]):
        axes[i, 0].set_ylabel(lab, fontsize=9)

    fig.colorbar(im_d, cax=cax_d, label="HU")
    fig.colorbar(im_u, cax=cax_u, label="|u| vox")

    out_dir.mkdir(parents=True, exist_ok=True)
    out1 = out_dir / f"{sid}_synth_phase_panel_coronal.png"
    fig.savefig(out1, dpi=140)
    plt.close(fig)

    fig2, axs = plt.subplots(1, 4, figsize=(14, 3.5))
    fig2.suptitle(f"{sid}  ·  DVF |u| mid-coronal (mid-AP)  ·  phase 06→01", fontsize=12, fontweight="bold")
    fig2.subplots_adjust(left=0.04, right=0.98, top=0.82, bottom=0.08, wspace=0.25)
    for ax, img, title, vmax in [
        (axs[0], el_mag, "Elastix |u|", vmax_u),
        (axs[1], syn_raw, "Synth |u| (raw)", vmax_u),
        (axs[2], syn_neg, "−Synth |u| (sign fix)", vmax_u),
        (
            axs[3],
            np.abs(el_mag - syn_neg),
            "|Elastix − (−synth)|",
            max(float(np.percentile(np.abs(el_mag - syn_neg), 98)), 0.5),
        ),
    ]:
        im = _imshow(ax, img, extent=extent, cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(box_asp)
        fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    out2 = out_dir / f"{sid}_dvf_elastix_vs_synth_coronal.png"
    fig2.savefig(out2, dpi=140)
    plt.close(fig2)
    return out1, out2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=int, nargs="*", default=list(range(1, 11)))
    args = ap.parse_args()
    for c in args.cases:
        p1, p2 = plot_case(c, OUT)
        print("wrote", p1)
        print("wrote", p2)
    print("OUT", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
