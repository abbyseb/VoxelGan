#!/usr/bin/env python3
"""Project C08 phase 01 and 06 at full detector size, then measure the diaphragm.

Two gantry angles only (0° and 180°). The 128×128 shrink is applied afterwards
on the same pixels, so we can see whether the breath is lost in the shrink.

Does not write into the training folders.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

LEARN = Path("/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python")
sys.path.insert(0, str(LEARN))

import itk  # noqa: E402
from itk import RTK as rtk  # noqa: E402

DIR_EXP = Path("/home/abhishek/Voxel_GAN/DIR EXPERIMENTS")
TRAIN = DIR_EXP / "arms" / "A1_oracle_dirlab" / "runs" / "DIR_C08" / "DIR_C08" / "train"
GEOM = TRAIN / "Proj" / "Geometry.xml"
OUT = (
    DIR_EXP
    / "arms"
    / "A3_synth_conditioned"
    / "runs"
    / "DIR_C08_tcia3_amp4"
    / "probe_proj"
    / "full_drr"
)

# Same detector the A1 DRR uses.
ORIGIN = (-200.0, -150.0, 0.0)
SPACING = (0.388, 0.388, 1.0)  # mm on the detector
SIZE = (1024, 768)  # (u, v)
SID, SDD = 1000.0, 1500.0
# mm at isocenter for one detector pixel
MM_PER_PX = SPACING[0] * (SID / SDD)


def two_angle_geometry(src: Path) -> Path:
    text = src.read_text()
    blocks = re.findall(r"<Projection>.*?</Projection>", text, flags=re.S)
    angles = [float(re.search(r"<GantryAngle>([^<]+)</GantryAngle>", b).group(1)) for b in blocks]

    def closest(target: float) -> int:
        return int(np.argmin([min(abs(a - target), 360 - abs(a - target)) for a in angles]))

    i0, i180 = closest(0.0), closest(180.0)
    start = text.index("<Projection>")
    end = text.rindex("</Projection>") + len("</Projection>")
    kept = "\n".join([blocks[i0], blocks[i180]])
    out = OUT / "geometry_ap_pa.xml"
    out.write_text(text[:start] + kept + text[end:])
    print(
        f"[full] angles {angles[i0]:.2f}° and {angles[i180]:.2f}° "
        f"(source indices {i0 + 1}, {i180 + 1})",
        flush=True,
    )
    return out


def load_geometry(path: Path):
    reader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
    reader.SetFilename(str(path))
    reader.GenerateOutputInformation()
    try:
        return reader.GetOutputObject()
    except AttributeError:
        return reader.GetOutput()


def project(volume: Path, geometry, n_proj: int) -> np.ndarray:
    """Return line integrals, shape (n_proj, v, u)."""
    ImageF3 = itk.Image[itk.F, 3]
    vol = itk.imread(str(volume), itk.F)
    use_cuda = hasattr(rtk, "CudaForwardProjectionImageFilter")
    stack = None
    if use_cuda:
        try:
            CudaImageF3 = rtk.CudaImage[itk.F, 3]
            vol.Update()
            gpu_vol = itk.cast_image_filter(vol, ttype=CudaImageF3).GetOutput()
            source = rtk.ConstantImageSource[CudaImageF3].New()
            source.SetOrigin([float(x) for x in ORIGIN])
            source.SetSpacing([float(x) for x in SPACING])
            source.SetSize([SIZE[0], SIZE[1], n_proj])
            source.SetConstant(0.0)
            projector = rtk.CudaForwardProjectionImageFilter[CudaImageF3, CudaImageF3].New()
            projector.SetInput(0, source.GetOutput())
            projector.SetInput(1, gpu_vol)
            projector.SetGeometry(geometry)
            projector.Update()
            stack = np.asarray(itk.GetArrayFromImage(projector.GetOutput()))
            print("[full] CUDA projector", flush=True)
        except Exception as exc:
            print(f"[full] CUDA failed ({exc}); using CPU", flush=True)
            stack = None
    if stack is None:
        source = rtk.ConstantImageSource[ImageF3].New()
        source.SetOrigin([float(x) for x in ORIGIN])
        source.SetSpacing([float(x) for x in SPACING])
        source.SetSize([SIZE[0], SIZE[1], n_proj])
        source.SetConstant(0.0)
        projector = rtk.JosephForwardProjectionImageFilter[ImageF3, ImageF3].New()
        projector.SetInput(0, source.GetOutput())
        projector.SetInput(1, vol)
        projector.SetGeometry(geometry)
        projector.Update()
        stack = np.asarray(itk.GetArrayFromImage(projector.GetOutput()))
        print("[full] CPU projector", flush=True)
    if stack.shape[0] != n_proj:
        if stack.shape[-1] == n_proj:
            stack = np.moveaxis(stack, -1, 0)
        else:
            raise SystemExit(f"unexpected stack {stack.shape}")
    return stack.astype(np.float32)


def edge_row(img: np.ndarray, axis: int) -> float:
    """Position of the steepest falling edge along one axis, in pixels."""
    prof = img.mean(axis=1 - axis)
    prof = np.convolve(prof, np.ones(9) / 9.0, mode="same")
    gy = -np.gradient(prof)
    lo, hi = int(0.15 * len(gy)), int(0.9 * len(gy))
    seg = gy[lo:hi]
    i = int(np.argmax(seg))
    if 0 < i < len(seg) - 1:
        y0, y1, y2 = seg[i - 1], seg[i], seg[i + 1]
        den = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / den if abs(den) > 1e-8 else 0.0
    else:
        delta = 0.0
    return lo + i + float(np.clip(delta, -1.0, 1.0))


def shrink(img: np.ndarray) -> np.ndarray:
    from scipy.ndimage import zoom

    h, w = img.shape
    small = zoom(img, (128 / h, 128 / w), order=1)
    return small[:128, :128]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    geom_path = two_angle_geometry(GEOM)
    geometry = load_geometry(geom_path)
    views = {}
    for phase in (1, 6):
        vol = TRAIN / f"CT_{phase:02d}.mha"
        print(f"[full] projecting {vol.name}", flush=True)
        views[phase] = project(vol, geometry, 2)
        print(f"[full] {vol.name} stack {views[phase].shape}", flush=True)

    print(
        f"[full] one detector pixel = {MM_PER_PX:.3f} mm at the isocenter",
        flush=True,
    )
    labels = ["0°", "180°"]
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for v, lab in enumerate(labels):
        a = views[1][v]
        b = views[6][v]
        for axis, name in ((0, "rows"), (1, "cols")):
            dpx = edge_row(a, axis) - edge_row(b, axis)
            print(
                f"[full] {lab} {name:4s}  {dpx:+7.1f} px   {dpx * MM_PER_PX:+6.1f} mm",
                flush=True,
            )
        sa, sb = shrink(a), shrink(b)
        # 128 pixel covers the same detector span as the full axis 0
        mm128 = SPACING[1] * (a.shape[0] / 128.0) * (SID / SDD)
        d128 = edge_row(sa, 0) - edge_row(sb, 0)
        print(
            f"[full] {lab} after 128 shrink, rows {d128:+7.1f} px   {d128 * mm128:+6.1f} mm",
            flush=True,
        )
        for ax, im, title in (
            (axes[v, 0], b, f"{lab} phase 06"),
            (axes[v, 1], a, f"{lab} phase 01"),
            (axes[v, 2], np.abs(a - b), f"{lab} |01−06|"),
        ):
            ax.imshow(im, cmap="gray")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    png = OUT / "c08_full_drr_01_vs_06.png"
    fig.savefig(png, dpi=80)
    plt.close(fig)
    np.save(OUT / "phase01.npy", views[1])
    np.save(OUT / "phase06.npy", views[6])
    print(f"[full] wrote {png}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
