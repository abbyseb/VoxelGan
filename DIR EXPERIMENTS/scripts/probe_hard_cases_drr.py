#!/usr/bin/env python3
"""Front-view phase 01 vs 06 at 128×128 for the hard DIR cases.

Same projector as the C08 check. Writes one comparison image.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom

DIR_EXP = Path("/home/abhishek/Voxel_GAN/DIR EXPERIMENTS")
sys.path.insert(0, str(DIR_EXP / "scripts"))
from probe_c08_full_drr import load_geometry, project  # noqa: E402

A1 = DIR_EXP / "arms" / "A1_oracle_dirlab" / "runs"
OUT = DIR_EXP / "arms" / "A3_synth_conditioned" / "runs" / "probe_drr_128"
CASES = (4, 6, 7, 9, 10)


def two_angles(src: Path, dest: Path) -> Path:
    text = src.read_text()
    blocks = re.findall(r"<Projection>.*?</Projection>", text, flags=re.S)
    angles = [float(re.search(r"<GantryAngle>([^<]+)</GantryAngle>", b).group(1)) for b in blocks]

    def closest(target: float) -> int:
        return int(np.argmin([min(abs(a - target), 360 - abs(a - target)) for a in angles]))

    i0, i180 = closest(0.0), closest(180.0)
    start = text.index("<Projection>")
    end = text.rindex("</Projection>") + len("</Projection>")
    dest.write_text(text[:start] + "\n".join([blocks[i0], blocks[i180]]) + text[end:])
    return dest


def ct_diaphragm_mm(train: Path) -> float:
    """How far the bottom of the lung moves between phase 01 and 06, in mm."""
    im1 = sitk.ReadImage(str(train / "CT_01.mha"))
    im6 = sitk.ReadImage(str(train / "CT_06.mha"))
    a1 = sitk.GetArrayFromImage(im1)
    a6 = sitk.GetArrayFromImage(im6)
    sp_y = float(im1.GetSpacing()[1])

    def edge(vol: np.ndarray) -> float:
        lung = vol < -400
        zc = lung.shape[0] // 2
        slab = lung[zc - 20 : zc + 20].any(axis=0)
        rows = []
        x0, x1 = slab.shape[1] // 4, 3 * slab.shape[1] // 4
        for xi in range(x0, x1):
            ys = np.where(slab[:, xi])[0]
            if len(ys) > 10:
                rows.append(ys.max())
        return float(np.median(rows))

    return abs(edge(a1) - edge(a6)) * sp_y


def to128(img: np.ndarray) -> np.ndarray:
    z = zoom(img.astype(np.float32), (128 / img.shape[0], 128 / img.shape[1]), order=1)
    return z[:128, :128]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    panels = {}
    print(f"{'case':6s} {'CT breath mm':>12s}", flush=True)
    for case in CASES:
        train = A1 / f"DIR_C{case:02d}" / f"DIR_C{case:02d}" / "train"
        mm = ct_diaphragm_mm(train)
        geom = two_angles(train / "Proj" / "Geometry.xml", OUT / f"geom_C{case:02d}.xml")
        geometry = load_geometry(geom)
        print(f"C{case:02d}   {mm:8.1f}   projecting", flush=True)
        p1 = project(train / "CT_01.mha", geometry, 2)
        p6 = project(train / "CT_06.mha", geometry, 2)
        # view 0 is the front (0°)
        panels[case] = (to128(p6[0]), to128(p1[0]), np.abs(to128(p1[0]) - to128(p6[0])), mm)

    # C08 already projected
    c08 = (
        DIR_EXP
        / "arms"
        / "A3_synth_conditioned"
        / "runs"
        / "DIR_C08_tcia3_amp4"
        / "probe_proj"
        / "full_drr"
    )
    if (c08 / "phase01.npy").is_file():
        p1 = np.load(c08 / "phase01.npy")
        p6 = np.load(c08 / "phase06.npy")
        mm = ct_diaphragm_mm(A1 / "DIR_C08" / "DIR_C08" / "train")
        panels[8] = (to128(p6[0]), to128(p1[0]), np.abs(to128(p1[0]) - to128(p6[0])), mm)
        print(f"C08   {mm:8.1f}   from saved full X-ray", flush=True)

    order = [c for c in (4, 6, 7, 8, 9, 10) if c in panels]
    fig, axes = plt.subplots(len(order), 3, figsize=(7.2, 2.3 * len(order)))
    for row, case in enumerate(order):
        b, a, d, mm = panels[case]
        for col, im, title in (
            (0, b, f"C{case:02d}  phase 06"),
            (1, a, f"C{case:02d}  phase 01   breath {mm:.0f} mm"),
            (2, d, f"C{case:02d}  difference"),
        ):
            axes[row, col].imshow(im, cmap="gray", interpolation="nearest")
            axes[row, col].set_title(title, fontsize=9)
            axes[row, col].axis("off")
    fig.tight_layout()
    png = OUT / "hard_cases_128_front.png"
    fig.savefig(png, dpi=120)
    plt.close(fig)
    print(f"wrote {png}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
