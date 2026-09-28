#!/usr/bin/env python3
"""Re-DRR an existing TCIA3 case so the lung fills the 128×128 picture.

The DVF is the one TCIA3 already wrote. This script does not run the decoder
and does not read DIR Elastix. It only replaces the projections:

  full detector → crop to the lung (same window at every phase) → 128×128

Real phase 01 and 06 are projected with that same window for the TRE test.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

LEARN = Path("/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python")
sys.path.insert(0, str(LEARN))

import itk  # noqa: E402
from itk import RTK as rtk  # noqa: E402
from scipy.ndimage import zoom  # noqa: E402

DIR_EXP = Path("/home/abhishek/Voxel_GAN/DIR EXPERIMENTS")
A1 = DIR_EXP / "arms" / "A1_oracle_dirlab" / "runs"
A3 = DIR_EXP / "arms" / "A3_synth_conditioned" / "runs"

ORIGIN = (-200.0, -150.0, 0.0)
SPACING = (0.388, 0.388, 1.0)
SIZE = (1024, 768)  # (u, v)
OUT_HW = 128


def log(msg: str) -> None:
    print(msg, flush=True)


def load_geometry(path: Path):
    reader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
    reader.SetFilename(str(path))
    reader.GenerateOutputInformation()
    try:
        return reader.GetOutputObject()
    except AttributeError:
        return reader.GetOutput()


def projection_matrices(xml: Path) -> list[np.ndarray]:
    text = xml.read_text()
    blocks = re.findall(r"<Projection>.*?</Projection>", text, flags=re.S)
    mats = []
    for block in blocks:
        nums = re.search(r"<Matrix>(.*?)</Matrix>", block, flags=re.S).group(1)
        vals = np.fromstring(nums, sep=" ")
        mats.append(vals.reshape(3, 4))
    return mats


def lung_points(mask_path: Path, step: int = 3) -> np.ndarray:
    import SimpleITK as sitk

    im = sitk.ReadImage(str(mask_path))
    a = sitk.GetArrayFromImage(im) > 0
    sp = np.array(im.GetSpacing(), dtype=np.float64)
    org = np.array(im.GetOrigin(), dtype=np.float64)
    zz, yy, xx = np.where(a[::step, ::step, ::step])
    pts = np.stack(
        [
            org[0] + (xx * step) * sp[0],
            org[1] + (yy * step) * sp[1],
            org[2] + (zz * step) * sp[2],
            np.ones(len(xx), dtype=np.float64),
        ],
        axis=1,
    )
    return pts


def lung_boxes(pts: np.ndarray, mats: list[np.ndarray]) -> np.ndarray:
    """Per-view crop in full-detector pixels: r0, r1, c0, c1 (v, then u)."""
    origin = np.array(ORIGIN[:2])
    sp = np.array(SPACING[:2])
    boxes = np.zeros((len(mats), 4), dtype=np.int32)
    for i, mat in enumerate(mats):
        uvw = pts @ mat.T
        pix = (uvw[:, :2] / uvw[:, 2:3] - origin) / sp
        u0, u1 = np.percentile(pix[:, 0], [0.5, 99.5])
        v0, v1 = np.percentile(pix[:, 1], [0.5, 99.5])
        du = max(24.0, 0.08 * (u1 - u0))
        dv = max(24.0, 0.08 * (v1 - v0))
        c0 = int(np.floor(u0 - du))
        c1 = int(np.ceil(u1 + du))
        r0 = int(np.floor(v0 - dv))
        r1 = int(np.ceil(v1 + dv))
        c0 = int(np.clip(c0, 0, SIZE[0] - 2))
        c1 = int(np.clip(c1, c0 + 2, SIZE[0]))
        r0 = int(np.clip(r0, 0, SIZE[1] - 2))
        r1 = int(np.clip(r1, r0 + 2, SIZE[1]))
        boxes[i] = (r0, r1, c0, c1)
    return boxes


def project_stack(volume: Path, geometry, n_proj: int) -> np.ndarray:
    """Line integrals, shape (n_proj, v, u)."""
    ImageF3 = itk.Image[itk.F, 3]
    vol = itk.imread(str(volume), itk.F)
    stack = None
    if hasattr(rtk, "CudaForwardProjectionImageFilter"):
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
            log(f"[drr] CUDA {volume.name} {tuple(stack.shape)}")
        except Exception as exc:
            log(f"[drr] CUDA failed ({exc}); CPU Joseph")
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
        log(f"[drr] CPU {volume.name} {tuple(stack.shape)}")
    if stack.shape[0] != n_proj and stack.shape[-1] == n_proj:
        stack = np.moveaxis(stack, -1, 0)
    return stack


def crop_to_128(frame: np.ndarray, box: np.ndarray) -> np.ndarray:
    r0, r1, c0, c1 = (int(v) for v in box)
    crop = frame[r0:r1, c0:c1]
    if crop.size == 0:
        crop = frame
    out = zoom(crop, (OUT_HW / crop.shape[0], OUT_HW / crop.shape[1]), order=1)
    out = out[:OUT_HW, :OUT_HW].astype(np.float32)
    lo, hi = float(out.min()), float(out.max())
    if hi > lo:
        out = (out - lo) / (hi - lo) * 255.0
    return out.astype(np.float32)


def write_bins(stack: np.ndarray, boxes: np.ndarray, out_dir: Path, phase: int) -> None:
    for i in range(stack.shape[0]):
        frame = crop_to_128(stack[i], boxes[i])
        path = out_dir / f"{phase:02d}_Proj_{i + 1:03d}.bin"
        frame.ravel(order="F").tofile(path)


def link_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.suffix == ".bin":
            continue
        target = dst / item.name
        if target.exists() or target.is_symlink():
            continue
        target.symlink_to(item)


def save_preview(old_bin: Path, new_bin: Path, out_png: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    old = np.fromfile(old_bin, dtype=np.float32).reshape(OUT_HW, OUT_HW, order="F")
    new = np.fromfile(new_bin, dtype=np.float32).reshape(OUT_HW, OUT_HW, order="F")

    def show(a):
        lo, hi = np.percentile(a, [1, 99])
        return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(show(old), cmap="gray")
    axes[0].set_title("current 128")
    axes[1].imshow(show(new), cmap="gray")
    axes[1].set_title("lung crop 128")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    log(f"[preview] {out_png}")


def build_real_eval(case: int, boxes: np.ndarray, geometry, run: Path) -> Path:
    scan = f"DIR_C{case:02d}"
    a1_train = A1 / scan / scan / "train"
    a1_mt = A1 / scan / "ModelTraining" / "train" / scan
    dst = run / "real_eval" / scan
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    (dst / "Angles.csv").symlink_to(a1_mt / "Angles.csv")
    (dst / "SourceVolumes").symlink_to(a1_mt / "SourceVolumes")
    (dst / "SourceProjections").mkdir()
    (dst / "TargetProjections").mkdir()
    for phase, folder in ((6, "SourceProjections"), (1, "TargetProjections")):
        vol = a1_train / f"CT_{phase:02d}.mha"
        stack = project_stack(vol, geometry, boxes.shape[0])
        for i in range(stack.shape[0]):
            frame = crop_to_128(stack[i], boxes[i])
            name = f"{phase:02d}_Proj_{i + 1:03d}_bin.npy"
            np.save(dst / folder / name, frame)
        del stack
    log(f"[real] {dst}")
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", type=int, required=True, choices=range(1, 11))
    ap.add_argument("--src-suffix", default="tcia3")
    ap.add_argument("--dst-suffix", default="tcia3_lung128")
    args = ap.parse_args()

    scan = f"DIR_C{args.case:02d}"
    src_run = A3 / f"{scan}_{args.src_suffix}"
    src_train = src_run / scan / "train"
    run = A3 / f"{scan}_{args.dst_suffix}"
    train = run / scan / "train"
    if (train / "01_Proj_001.bin").is_file() and (run / "ModelTraining" / "train" / scan).is_dir():
        log(f"[skip] already prepared {run}")
        return 0
    if not (src_train / "CT_06.mha").is_file():
        raise SystemExit(f"Missing TCIA3 CTs: {src_train}")

    geom_xml = src_train / "Proj" / "Geometry.xml"
    mats = projection_matrices(geom_xml)
    mask = A1 / scan / scan / "train" / "Mask_Lung.mha"
    pts = lung_points(mask)
    boxes = lung_boxes(pts, mats)
    train.mkdir(parents=True, exist_ok=True)
    np.save(run / "lung_boxes.npy", boxes)
    h = boxes[:, 1] - boxes[:, 0]
    w = boxes[:, 3] - boxes[:, 2]
    log(
        f"[box] {scan} views={len(boxes)} "
        f"SI zoom median {np.median(SIZE[1] / h):.2f}x "
        f"width zoom median {np.median(SIZE[0] / w):.2f}x "
        f"box px median {int(np.median(h))} x {int(np.median(w))}"
    )
    link_tree(src_train, train)
    geometry = load_geometry(geom_xml)

    phases = sorted(int(p.name[3:5]) for p in src_train.glob("CT_*.mha"))
    for phase in phases:
        have = list(train.glob(f"{phase:02d}_Proj_*.bin"))
        if len(have) >= len(boxes):
            log(f"[drr] phase {phase:02d} exists")
            continue
        for stale in have:
            stale.unlink()
        stack = project_stack(src_train / f"CT_{phase:02d}.mha", geometry, len(boxes))
        write_bins(stack, boxes, train, phase)
        del stack
        log(f"[drr] wrote phase {phase:02d}")

    old = src_train / "06_Proj_001.bin"
    new = train / "06_Proj_001.bin"
    if old.is_file() and new.is_file():
        save_preview(
            old,
            new,
            A3 / "probe_drr_128" / f"{scan.lower()}_lung128_front.png",
            f"{scan} phase 06 front",
        )

    meta = {
        "case": args.case,
        "dvf": "reused TCIA3 synth DVF_sub (decoder, not Elastix)",
        "crop": "lung voxels projected onto the full detector, 8% margin, then 128",
        "si_zoom_median": float(np.median(SIZE[1] / h)),
        "width_zoom_median": float(np.median(SIZE[0] / w)),
        "source_run": str(src_run),
    }
    (run / "lung128_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    log("[prep] ModelTraining")
    sys.path.insert(0, str(DIR_EXP / "scripts"))
    from prepare_a3_dir_case import run_prep

    run_prep(run, scan, geom_xml, with_test=False)
    build_real_eval(args.case, boxes, geometry, run)
    log(f"[done] {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
