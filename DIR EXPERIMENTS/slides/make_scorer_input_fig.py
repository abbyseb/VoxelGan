#!/usr/bin/env python3
"""Slide figure: what the network was fed from DIR-Lab, old scorer vs fixed scorer, next to a TCIA training scan.

Old path  (prepare_a3_dir_case.synthesize): HU -> mu -> resize_volume_zyx to 160^3 -> norm_mu
New path  (eval_dir_tcia3_iso2mm_v2):       HU -> resample to 2 mm iso 160^3 centred on lung -> mu -> minmax
Mid-coronal slices, all on the same 0..1 grey scale, so size and brightness are directly comparable.

  cd "DIR EXPERIMENTS"; LEARN-GUI/.venv/bin/python slides/make_scorer_input_fig.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

DIR_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIR_EXP / "scripts"))
import eval_dir_tcia3_iso2mm_v2 as V2  # noqa: E402
from prepare_a3_dir_case import hu_to_mu, norm_mu, resize_volume_zyx  # noqa: E402

OUT = DIR_EXP / "slides" / "figs" / "20_scorer_input_old_vs_new.png"
CASES = (1, 8)


def mid_coronal(v, m):
    z = int(np.round(np.nonzero(m.any(axis=(1, 2)))[0].mean()))
    return v[z]


def lung_width_mm(m, spacing_x):
    cols = np.nonzero(m.any(axis=(0, 1)))[0]
    return (cols.max() - cols.min() + 1) * spacing_x


def old_input(case):
    hu, lung, _, spacing, _ = V2.load_case_image(case)
    x = norm_mu(resize_volume_zyx(hu_to_mu(hu, 0.02), 160))
    m = resize_volume_zyx(lung.astype(np.float32), 160) > 0.5
    vox = spacing * np.array(hu.shape[::-1]) / 160.0  # mm per voxel (x, y, z) after squeezing
    return x, m, vox


def new_input(case):
    p = V2.pack_case(case)
    return V2.minmax(p["mu"]), p["lung_iso"]


def main():
    tcia = V2.minmax(np.load(V2.TCIA_MU).squeeze())
    tlung = np.load(V2.TCIA_LUNG).squeeze() > 0
    tw = lung_width_mm(tlung, 2.0)

    fig, ax = plt.subplots(len(CASES), 3, figsize=(12, 4.9 * len(CASES)))
    for r, case in enumerate(CASES):
        xo, mo, vox = old_input(case)
        xn, mn = new_input(case)
        wn = lung_width_mm(mn, 2.0)
        wo_model = lung_width_mm(mo, 2.0)  # width the network "thinks" it is, since it assumes 2 mm voxels
        panels = [
            (tcia, tlung, f"Training scan (TCIA)\n2 mm voxels · lung {tw:.0f} mm wide"),
            (xo, mo, f"DIR-Lab C{case:02d}, OLD scorer\nvoxels {vox.min():.1f}-{vox.max():.1f} mm (uneven) · "
                     f"model sees {wo_model:.0f} mm (real {wn:.0f})"),
            (xn, mn, f"DIR-Lab C{case:02d}, FIXED scorer\n2 mm voxels · lung {wn:.0f} mm wide"),
        ]
        for a, (v, m, title) in zip(ax[r], panels):
            a.imshow(mid_coronal(v, m), cmap="gray", vmin=0, vmax=1, origin="upper")
            a.set_title(title, fontsize=10)
            a.set_xticks([]), a.set_yticks([])
        print(f"C{case:02d}: old vox {np.round(vox, 2)}, width as seen {wo_model:.0f} vs real {wn:.0f} mm "
              f"({100 * (1 - wo_model / wn):.0f}% narrower); lung bright old {np.median(xo[mo]):.2f} "
              f"new {np.median(xn[mn]):.2f} tcia {np.median(tcia[tlung]):.2f}")
    fig.suptitle("What the network was given: same 160³ box, same grey scale", fontsize=13)
    fig.tight_layout(h_pad=2.5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
