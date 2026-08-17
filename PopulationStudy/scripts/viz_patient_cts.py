#!/usr/bin/env python3
"""Axial slice montage of SPARE GTVol phases for PopulationStudy screening."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
OUT = ROOT / "plots" / "ct_survey"


def load_mha(path: Path) -> np.ndarray:
    try:
        import SimpleITK as sitk
    except ImportError:
        import itk

        return np.asarray(itk.imread(str(path)), dtype=np.float32)
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)


def window_ct(vol_or_sl: np.ndarray, lo=None, hi=None) -> np.ndarray:
    """Percentile window — SPARE GTVol is attenuation float, not HU."""
    x = vol_or_sl.astype(np.float32)
    if lo is None or hi is None:
        lo = float(np.percentile(x, 1))
        hi = float(np.percentile(x, 99))
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((x - lo) / (hi - lo), 0, 1)


def axial_slice(vol: np.ndarray, idx: int) -> np.ndarray:
    z = int(np.clip(idx, 0, vol.shape[0] - 1))
    # SPARE GT axial display: rotate 180 so anatomy is upright vs raw array orientation
    return np.rot90(vol[z], 2)


def maybe_rot_mask(m: np.ndarray) -> np.ndarray:
    return np.rot90(m, 2)


def lung_z_bounds(mask: np.ndarray):
    zz = np.where(mask.any(axis=(1, 2)))[0]
    if zz.size == 0:
        return None
    return int(zz[0]), int(zz[-1]), int(zz[len(zz) // 2])


def montage_patient(patient_dir: Path, slice_idx: int, out_path: Path, overlay_lung: bool, title_extra: str = ""):
    vols = [load_mha(patient_dir / f"GTVol_{p:02d}.mha") for p in range(1, 11)]
    mask = None
    mp = patient_dir / "Mask_Lung.mha"
    if mp.exists():
        mask = load_mha(mp) > 0

    # global window from phase 5 full volume
    lo = float(np.percentile(vols[4], 1))
    hi = float(np.percentile(vols[4], 99))

    fig, axs = plt.subplots(2, 5, figsize=(14, 6))
    shape = vols[0].shape
    z_used = int(np.clip(slice_idx, 0, shape[0] - 1))
    for i, vol in enumerate(vols):
        ax = axs[i // 5, i % 5]
        sl = window_ct(axial_slice(vol, z_used), lo, hi)
        ax.imshow(sl, cmap="gray", origin="lower", aspect="equal", vmin=0, vmax=1)
        if overlay_lung and mask is not None:
            m = maybe_rot_mask(mask[z_used].astype(np.float32))
            if m.any():
                ax.contour(m, levels=[0.5], colors="lime", linewidths=0.6)
        ax.set_title(f"ph {i + 1:02d}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"{patient_dir.name}  GTVol  axial z={z_used}/{shape[0] - 1}  "
        f"shape={shape}{title_extra}",
        fontsize=12,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return shape, z_used


def comparison_grid(patients: list[str], slice_indices: dict[str, int], out_path: Path, phase: int = 5, title: str = ""):
    n = len(patients)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    axs = np.atleast_2d(axs)
    for i, pid in enumerate(patients):
        ax = axs[i // cols, i % cols]
        vol = load_mha(RAW / pid / f"GTVol_{phase:02d}.mha")
        z = int(np.clip(slice_indices[pid], 0, vol.shape[0] - 1))
        lo, hi = float(np.percentile(vol, 1)), float(np.percentile(vol, 99))
        ax.imshow(window_ct(axial_slice(vol, z), lo, hi), cmap="gray", origin="lower", aspect="equal", vmin=0, vmax=1)
        mask_p = RAW / pid / "Mask_Lung.mha"
        if mask_p.exists():
            m = maybe_rot_mask(((load_mha(mask_p) > 0).astype(np.float32))[z])
            if m.any():
                ax.contour(m, levels=[0.5], colors="lime", linewidths=0.5)
        ax.set_title(f"{pid} z={z}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    for j in range(n, rows * cols):
        axs[j // cols, j % cols].axis("off")
    fig.suptitle(title or f"All patients  phase {phase:02d}", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", type=int, default=54, help="Fixed axial index (may be outside lung on 450-Z GT)")
    ap.add_argument("--patients", default="P1,P2,P3,P4,P5,P6,P7,P8,P9")
    ap.add_argument("--no_overlay_lung", action="store_true")
    ap.add_argument("--also_lung_mid", action="store_true", default=True)
    args = ap.parse_args()
    overlay = not args.no_overlay_lung
    patients = [p.strip() for p in args.patients.split(",") if p.strip()]

    fixed_idx = {}
    lung_mid = {}
    for pid in patients:
        pdir = RAW / pid
        mask = load_mha(pdir / "Mask_Lung.mha") > 0
        bounds = lung_z_bounds(mask)
        mid = bounds[2] if bounds else args.slice
        lung_mid[pid] = mid
        fixed_idx[pid] = args.slice

        out = OUT / f"{pid}_slice{args.slice:02d}.png"
        shape, z = montage_patient(pdir, args.slice, out, overlay, title_extra="  [fixed index]")
        print(f"{pid}: {out.name}  vol={shape}  z={z}  lung_mid_z={mid}", flush=True)

        if args.also_lung_mid:
            out_m = OUT / f"{pid}_lung_mid.png"
            montage_patient(pdir, mid, out_m, overlay, title_extra="  [lung mid-Z]")
            print(f"{pid}: {out_m.name}  z={mid}", flush=True)

    comparison_grid(
        patients,
        fixed_idx,
        OUT / f"all_patients_slice{args.slice:02d}.png",
        title=f"All patients  phase 05  fixed axial z={args.slice}",
    )
    comparison_grid(
        patients,
        lung_mid,
        OUT / "all_patients_lung_mid.png",
        title="All patients  phase 05  lung mid-Z (per patient)",
    )
    print(f"wrote comparison grids under {OUT}", flush=True)


if __name__ == "__main__":
    main()
