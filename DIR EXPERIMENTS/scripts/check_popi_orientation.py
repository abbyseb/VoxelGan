#!/usr/bin/env python3
"""POPI orientation and intensity check against the TCIA training frame.

Read-only. For every POPI patient it reports:
  - .mhd header: size, spacing, origin, direction matrix, AnatomicalOrientation /
    TransformMatrix lines, and value percentiles of 00.mhd (HU or HU+1024?).
  - the packed cube (PACKED/<pid>/CT_01.npy): min value vs padding, lung base end
    (head-foot) and larger-lung side (left-right), compared with TCIA S1.
  - landmark motion (phase 00 -> reference) mean shift per axis in the packed frame.
It writes one PNG per patient: coronal + axial of TCIA S1 and packed POPI side by side.

  cd "DIR EXPERIMENTS"
  python scripts/check_popi_orientation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

DIR_EXP = Path(__file__).resolve().parents[1]
POPI = Path("/media/abhishek/3CCA3CADCA3C6574/POPI/MedPhys11")
PACKED = Path("/media/abhishek/3CCA3CADCA3C6574/POPI/packed_r3_tcia3")
TCIA_CT = Path("/media/abhishek/3CCA3CADCA3C6574/TCIA_4D-Lung/packed_r3_hu/S1/all/CT_01.npy")
TCIA_LUNG = Path("/media/abhishek/3CCA3CADCA3C6574/TCIA_4D-Lung/packed_r3_hu/S1/all/Mask_Lung.npy")
OUT = DIR_EXP / "popi_tcia3" / "orientation_check"
PATIENTS = [("bl", 6), ("ng", 5), ("dx", 5), ("gt", 5), ("mm2", 5), ("bh", 5)]


def header_lines(mhd: Path) -> dict:
    keep = {}
    for line in mhd.read_text(errors="ignore").splitlines():
        if "=" in line:
            k, v = [s.strip() for s in line.split("=", 1)]
            if k in ("AnatomicalOrientation", "TransformMatrix", "Offset", "ElementSpacing",
                     "DimSize", "ElementType", "CenterOfRotation"):
                keep[k] = v
    return keep


def lung_from_hu(vol_hu: np.ndarray) -> np.ndarray:
    raw = (vol_hu > -950) & (vol_hu < -300)
    lab, n = ndimage.label(raw)
    if n == 0:
        return raw
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    # drop components touching the cube border (outside air / padding)
    border = set(np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(),
                                           lab[:, -1].ravel(), lab[:, :, 0].ravel(), lab[:, :, -1].ravel()])))
    keep = np.zeros(n + 1, bool)
    for i in np.argsort(counts)[::-1][:3]:
        if i != 0 and i not in border and counts[i] > 5000:
            keep[i] = True
    if not keep.any():
        # The lung itself touches the cube face, so the border rule deletes it.
        for i in np.argsort(counts)[::-1][:3]:
            if i != 0 and counts[i] > 5000:
                keep[i] = True
    return keep[lab]


def base_row_fraction(lung: np.ndarray) -> float:
    """Along numpy axis 1 (Y). >0.5: lung base (widest part) at high Y."""
    area = lung.sum(axis=(0, 2)).astype(float)
    rows = np.nonzero(area)[0]
    if rows.size == 0:
        return float("nan")
    lo, hi = rows.min(), rows.max()
    return float((np.argmax(area) - lo) / max(hi - lo, 1))


def bigger_lung_side(lung: np.ndarray) -> str:
    """Along numpy axis 2 (X): which half holds more lung. The right lung is larger."""
    xs = np.nonzero(lung.any(axis=(0, 1)))[0]
    if xs.size == 0:
        return "?"
    mid = 0.5 * (xs.min() + xs.max())
    low = lung[:, :, : int(mid)].sum()
    high = lung[:, :, int(mid):].sum()
    return "low-X" if low > high else "high-X"


def slices(vol: np.ndarray, lung: np.ndarray):
    zc = int(np.round(np.nonzero(lung.any(axis=(1, 2)))[0].mean()))   # AP (axis 0)
    yc = int(np.round(np.nonzero(lung.any(axis=(0, 2)))[0].mean()))   # SI (axis 1)
    return vol[zc], vol[:, yc, :]  # coronal (Y,X), axial (Z,X)


def save_png(pid, tcia, tcia_lung, popi, popi_lung):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tc, ta = slices(tcia, tcia_lung)
    pc, pa = slices(popi, popi_lung)
    fig, ax = plt.subplots(2, 2, figsize=(9, 9))
    for a, im, t in zip(ax.ravel(), (tc, pc, ta, pa),
                        ("TCIA S1 coronal", f"POPI {pid} coronal", "TCIA S1 axial", f"POPI {pid} axial")):
        a.imshow(np.clip(im, -1000, 400), cmap="gray", origin="upper")
        a.set_title(t)
        a.set_xlabel("X index")
    ax[0, 0].set_ylabel("Y index (row 0 top)")
    ax[1, 0].set_ylabel("Z index (row 0 top)")
    fig.suptitle("Apex at same end (coronal)? Heart / bigger lung on same side (axial)?")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"popi_{pid}_vs_tcia.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def main() -> None:
    import argparse
    global PACKED, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", type=Path, default=PACKED)
    args = ap.parse_args()
    PACKED = args.packed
    OUT = DIR_EXP / "popi_tcia3" / f"orientation_check_{PACKED.name}"
    tcia = np.load(TCIA_CT).astype(np.float32)
    tcia_lung = np.load(TCIA_LUNG) > 0
    t_base = base_row_fraction(tcia_lung)
    t_side = bigger_lung_side(tcia_lung)
    print(f"TCIA S1: min {tcia.min():.0f} max {tcia.max():.0f} | base_row_frac {t_base:.2f} | bigger lung {t_side}")
    report = {"tcia_s1": {"base_row_frac": t_base, "bigger_lung_side": t_side,
                          "min": float(tcia.min()), "max": float(tcia.max())}, "popi": {}}

    for pid, ref in PATIENTS:
        mhd = POPI / pid / "mhd" / "00.mhd"
        img = sitk.ReadImage(str(mhd))
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        p = np.percentile(arr, [0.1, 1, 5, 50, 95, 99.9])
        hdr = header_lines(mhd)
        offset_guess = "HU+1024 (unsigned)" if p[1] > -200 else "plain HU"

        packed = np.load(PACKED / pid / "CT_01.npy").astype(np.float32)
        vol_hu = packed - 1024.0 if offset_guess.startswith("HU+") else packed
        lung = lung_from_hu(vol_hu)
        base = base_row_fraction(lung)
        side = bigger_lung_side(lung)
        pad_frac = float((packed == -1000.0).mean())

        lm0 = np.load(PACKED / pid / "lm_00_zyx.npy")
        lmr = np.load(PACKED / pid / f"lm_{ref:02d}_zyx.npy")
        n = min(len(lm0), len(lmr))
        shift = ((lmr[:n] - lm0[:n]).mean(axis=0) * 2.0).tolist()  # zyx mm

        same_si = (base > 0.5) == (t_base > 0.5)
        same_lr = side == t_side
        png = save_png(pid, tcia, tcia_lung, vol_hu, lung)
        row = {
            "sitk_direction": list(img.GetDirection()),
            "sitk_origin": list(img.GetOrigin()),
            "sitk_spacing": list(img.GetSpacing()),
            "header": hdr,
            "native_percentiles_0.1_1_5_50_95_99.9": p.tolist(),
            "intensity_guess": offset_guess,
            "packed_min": float(packed.min()),
            "packed_frac_equal_minus1000": pad_frac,
            "base_row_frac": base,
            "bigger_lung_side": side,
            "same_head_foot_as_tcia": bool(same_si),
            "same_left_right_as_tcia": bool(same_lr),
            "landmark_shift_00_to_ref_zyx_mm": shift,
            "png": str(png),
        }
        report["popi"][pid] = row
        print(
            f"{pid}: dir={np.round(img.GetDirection(), 2).tolist()} | {hdr.get('AnatomicalOrientation', '-')} | "
            f"p1={p[1]:.0f} p50={p[3]:.0f} -> {offset_guess} | packed min {packed.min():.0f} "
            f"(pad {pad_frac:.2f}) | base {base:.2f} SI {'OK' if same_si else 'FLIPPED'} | "
            f"bigger lung {side} LR {'OK' if same_lr else 'MIRRORED?'} | "
            f"lm shift zyx {np.round(shift, 1).tolist()} mm",
            flush=True,
        )

    (OUT / "popi_orientation_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {OUT / 'popi_orientation_check.json'} and one PNG per patient")


if __name__ == "__main__":
    main()
