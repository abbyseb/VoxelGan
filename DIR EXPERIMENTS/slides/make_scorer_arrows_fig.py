#!/usr/bin/env python3
"""Slide figure: motion arrows on DIR-Lab C08, Elastix vs TCIA3 ep100 fed by the OLD vs the FIXED scorer input.

Old: HU -> mu -> resize_volume_zyx to 160^3 -> norm_mu -> net -> upsample_dvf_voxel to native (as synthesize()).
New: eval_dir_tcia3_iso2mm_v2.pack_case + predict (no flip).
All three fields are put on the fixed scorer's 2 mm grid in mm, then drawn on the mid-coronal slice (in-plane
left-right and head-foot components). Lung-mean 3D cosine vs Elastix is printed in each title.

  cd "DIR EXPERIMENTS"; CUDA_VISIBLE_DEVICES=1 <LEARN-GUI venv python> slides/make_scorer_arrows_fig.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import map_coordinates

DIR_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIR_EXP / "scripts"))
import eval_dir_tcia3_iso2mm_v2 as V2  # noqa: E402
from prepare_a3_dir_case import hu_to_mu, norm_mu, resize_volume_zyx, upsample_dvf_voxel  # noqa: E402

CASE = 8
OUT = DIR_EXP / "slides" / "figs" / f"21_arrows_c{CASE:02d}_old_vs_new.png"
STEP = 6  # voxels between arrows


def old_field_mm(g, device, pack):
    hu, _, origin, spacing, _ = V2.load_case_image(CASE)
    x = torch.from_numpy(norm_mu(resize_volume_zyx(hu_to_mu(hu, 0.02), 160))[None, None]).to(device)
    with torch.no_grad():
        d = g(x, torch.tensor([5], device=device), torch.tensor([0], device=device))[0].cpu().numpy()
    nat = upsample_dvf_voxel(d, hu.shape, 160)  # native voxels, exactly as synthesize() warps CT_06
    xx, yy, zz = V2.iso_physical_grid(pack["origin_iso_xyz"])
    pts = np.stack([((zz - origin[2]) / spacing[2]).ravel(), ((yy - origin[1]) / spacing[1]).ravel(),
                    ((xx - origin[0]) / spacing[0]).ravel()])
    u = np.stack([map_coordinates(nat[c], pts, order=1, mode="nearest").reshape(160, 160, 160) * spacing[c]
                  for c in range(3)], -1)
    return u.astype(np.float32)


def lung_cos(a, b, m):
    a, b = a[m], b[m]
    return float(np.mean((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-6)))


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    g = V2.load_generator(device)
    pack = V2.pack_case(CASE)
    lung = pack["lung_iso"]
    el = V2.elastix_on_iso(pack) * V2.SPACING
    new = V2.predict(g, device, pack["mu"], flip=False) * V2.SPACING
    old = old_field_mm(g, device, pack)
    c_old, c_new = lung_cos(old, el, lung), lung_cos(new, el, lung)
    print(f"C{CASE:02d} lung cosine vs Elastix: old {c_old:.2f}  new {c_new:.2f}")

    z = int(np.round(np.nonzero(lung.any(axis=(1, 2)))[0].mean()))
    img = V2.minmax(pack["mu"])[z]
    m2 = lung[z]
    yy, xx = np.mgrid[0:160:STEP, 0:160:STEP]
    keep = m2[yy, xx]
    panels = [(el, "Elastix (reference)", "tab:green"),
              (old, f"Model, OLD scorer input\nagreement {c_old:.2f}", "tab:red"),
              (new, f"Model, FIXED scorer input\nagreement {c_new:.2f}", "tab:blue")]
    fig, ax = plt.subplots(1, 3, figsize=(15, 6.0))
    for a, (u, title, col) in zip(ax, panels):
        a.imshow(img, cmap="gray", vmin=0, vmax=1, origin="upper")
        ux, uy = u[z, yy, xx, 0], u[z, yy, xx, 1]
        # u is a pull field (points up, toward the apex, for exhale->inhale); -u is drawn so arrows show tissue motion
        # (diaphragm moving down); both in-plane components negated, same for all panels.
        a.quiver(xx[keep], yy[keep], -ux[keep], -uy[keep], color=col, angles="xy", scale_units="xy",
                 scale=0.5, width=0.004)
        a.set_title(title, fontsize=12)
        a.set_xticks([]), a.set_yticks([])
    fig.suptitle(f"DIR-Lab C{CASE:02d}: breathing motion arrows (same model TCIA3, same slice, same arrow scale)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT, dpi=120)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
