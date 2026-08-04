#!/usr/bin/env python3
"""Prepare SPARE MC P1 phase CTs + masked B-spline Elastix DVFs (90 pairs).

Uses lung-mask-restricted B-spline registration (not MultiBSpline sliding).
Writes data/spare/train and data/spare/val with leave-phase-out split.

Requires the LEARN-GUI venv (itk-elastix), e.g.:
  /home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python \\
      scripts/prepare_elastix_pairs.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _bbox_from_mask(mask_zyx: np.ndarray, pad: int = 8):
    coords = np.argwhere(mask_zyx > 0)
    if coords.size == 0:
        raise RuntimeError('Lung mask is empty')
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1
    z0 = max(0, z0 - pad)
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    z1 = min(mask_zyx.shape[0], z1 + pad)
    y1 = min(mask_zyx.shape[1], y1 + pad)
    x1 = min(mask_zyx.shape[2], x1 + pad)
    return (z0, z1, y0, y1, x0, x1)


def _resample_zyx(vol_zyx: np.ndarray, out_size: int, is_mask: bool = False) -> np.ndarray:
    import itk

    img = itk.GetImageFromArray(vol_zyx.astype(np.float32 if not is_mask else np.uint8))
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, 0.0))

    in_size = np.array(img.GetLargestPossibleRegion().GetSize(), dtype=np.float64)
    # ITK size is (x, y, z); array shape is (z, y, x)
    in_size_xyz = np.array([vol_zyx.shape[2], vol_zyx.shape[1], vol_zyx.shape[0]], dtype=np.float64)
    out_size_xyz = np.array([out_size, out_size, out_size], dtype=np.float64)
    out_spacing = (in_size_xyz / out_size_xyz).tolist()

    Dimension = 3
    ImageType = itk.Image[itk.F, Dimension] if not is_mask else itk.Image[itk.UC, Dimension]
    if is_mask:
        src = itk.GetImageFromArray(vol_zyx.astype(np.uint8))
    else:
        src = itk.GetImageFromArray(vol_zyx.astype(np.float32))
    src.SetSpacing((1.0, 1.0, 1.0))
    src.SetOrigin((0.0, 0.0, 0.0))

    interpolator = (
        itk.NearestNeighborInterpolateImageFunction.New(src)
        if is_mask
        else itk.LinearInterpolateImageFunction.New(src)
    )
    resample = itk.ResampleImageFilter.New(src)
    resample.SetInterpolator(interpolator)
    resample.SetSize([int(out_size)] * 3)
    resample.SetOutputSpacing(out_spacing)
    resample.SetOutputOrigin(src.GetOrigin())
    resample.SetOutputDirection(src.GetDirection())
    if is_mask:
        resample.SetDefaultPixelValue(0)
    else:
        resample.SetDefaultPixelValue(float(vol_zyx.min()))
    resample.Update()
    out = itk.array_from_image(resample.GetOutput())
    if is_mask:
        out = (out > 0).astype(np.uint8)
    else:
        out = out.astype(np.float32)
    # Force unit spacing convention for Elastix / training (1 voxel = 1 mm)
    return out


def _to_itk_float(vol_zyx: np.ndarray):
    import itk

    img = itk.GetImageFromArray(vol_zyx.astype(np.float32))
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection(itk.matrix_from_array(np.eye(3)))
    return img


def _to_itk_mask(mask_zyx: np.ndarray):
    import itk

    img = itk.GetImageFromArray(mask_zyx.astype(np.uint8))
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection(itk.matrix_from_array(np.eye(3)))
    return img


def register_pair(fixed_zyx, moving_zyx, mask_zyx, param_file: Path):
    """Elastix: fixed=target, moving=reference. DVF warps moving → fixed."""
    import itk

    fixed = _to_itk_float(fixed_zyx)
    moving = _to_itk_float(moving_zyx)
    mask = _to_itk_mask(mask_zyx)

    param_obj = itk.ParameterObject.New()
    param_obj.AddParameterFile(str(param_file))

    ImageType = type(fixed)
    elastix = itk.ElastixRegistrationMethod[ImageType, ImageType].New()
    elastix.SetFixedImage(fixed)
    elastix.SetMovingImage(moving)
    elastix.SetFixedMask(mask)
    elastix.SetMovingMask(mask)
    elastix.SetParameterObject(param_obj)
    elastix.LogToConsoleOff()
    elastix.Update()

    transform_params = elastix.GetTransformParameterObject()
    transformix = itk.TransformixFilter[ImageType].New()
    transformix.SetMovingImage(moving)
    transformix.SetTransformParameterObject(transform_params)
    transformix.SetComputeDeformationField(True)
    transformix.Update()

    dvf_img = transformix.GetOutputDeformationField()
    # ITK vector image → array (Z, Y, X, 3) in physical units (= voxels at spacing 1)
    dvf = itk.array_from_image(dvf_img).astype(np.float32)
    return dvf


def preprocess_case(case_dir: Path, out_size: int, pad: int):
    import itk

    mask = itk.array_from_image(
        itk.imread(str(case_dir / 'Mask_Lung.mha'))
    )
    bbox = _bbox_from_mask(mask, pad=pad)
    z0, z1, y0, y1, x0, x1 = bbox
    print(f'Lung bbox (zyx): [{z0}:{z1}, {y0}:{y1}, {x0}:{x1}]')

    cts = []
    for p in range(1, 11):
        vol = itk.array_from_image(
            itk.imread(str(case_dir / f'GTVol_{p:02d}.mha'), itk.F)
        ).astype(np.float32)
        cropped = vol[z0:z1, y0:y1, x0:x1]
        cts.append(_resample_zyx(cropped, out_size, is_mask=False))
        print(f'  CT_{p:02d}: {vol.shape} → crop {cropped.shape} → {cts[-1].shape}')

    mask_c = mask[z0:z1, y0:y1, x0:x1]
    mask_r = _resample_zyx(mask_c, out_size, is_mask=True)
    print(f'  Mask_Lung → {mask_r.shape}, voxels={int(mask_r.sum())}')
    return cts, mask_r


def save_split(out_dir: Path, cts, pairs):
    """pairs: list of (ref_1idx, tgt_1idx, dvf_zyx3) with 1-indexed phases."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, ct in enumerate(cts, start=1):
        np.save(out_dir / f'CT_{i:02d}.npy', ct.astype(np.float32))
    for ref, tgt, dvf in pairs:
        np.save(out_dir / f'{ref:02d}_to_{tgt:02d}_pair.npy', dvf.astype(np.float32))
    print(f'Wrote {len(cts)} CTs + {len(pairs)} pairs → {out_dir}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--case_dir',
        type=Path,
        default=ROOT / 'Data/MonteCarloDatasets/Training/P1/MC_T_P1_NS',
    )
    ap.add_argument('--out_size', type=int, default=128)
    ap.add_argument('--pad', type=int, default=8)
    ap.add_argument(
        '--param_file',
        type=Path,
        default=ROOT / 'configs/elastix_bspline_masked.txt',
    )
    ap.add_argument('--held_out', type=int, nargs='+', default=[5, 9],
                    help='1-indexed phases held out for validation')
    ap.add_argument('--train_dir', type=Path, default=ROOT / 'data/spare/train')
    ap.add_argument('--val_dir', type=Path, default=ROOT / 'data/spare/val')
    ap.add_argument('--max_pairs', type=int, default=None,
                    help='Optional cap for smoke tests')
    ap.add_argument('--skip_existing', action='store_true')
    args = ap.parse_args()

    try:
        import itk  # noqa: F401
    except ImportError:
        print(
            'itk/itk-elastix not found. Run with:\n'
            '  /home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python '
            'scripts/prepare_elastix_pairs.py',
            file=sys.stderr,
        )
        sys.exit(1)

    print('Preprocessing phases…')
    cts, mask = preprocess_case(args.case_dir, args.out_size, args.pad)

    held = set(args.held_out)
    print(f'Held-out phases (1-indexed): {sorted(held)}')

    # All directed pairs + identity
    jobs = [(i, j) for i in range(1, 11) for j in range(1, 11)]
    if args.max_pairs is not None:
        jobs = jobs[: args.max_pairs]

    train_pairs, val_pairs = [], []
    for n, (ref, tgt) in enumerate(jobs, start=1):
        split = 'val' if (ref in held or tgt in held) else 'train'
        out_dir = args.val_dir if split == 'val' else args.train_dir
        out_path = out_dir / f'{ref:02d}_to_{tgt:02d}_pair.npy'

        if args.skip_existing and out_path.exists():
            dvf = np.load(out_path)
            print(f'[{n}/{len(jobs)}] skip existing {out_path.name}')
        elif ref == tgt:
            dvf = np.zeros(cts[0].shape + (3,), dtype=np.float32)
            print(f'[{n}/{len(jobs)}] identity {ref:02d}→{tgt:02d}')
        else:
            print(f'[{n}/{len(jobs)}] Elastix {ref:02d}→{tgt:02d} ({split}) …', flush=True)
            # fixed=target, moving=reference so DVF warps ref → target
            dvf = register_pair(cts[tgt - 1], cts[ref - 1], mask, args.param_file)
            print(f'    DVF range per-axis: '
                  f'{dvf[...,0].min():.2f}:{dvf[...,0].max():.2f}, '
                  f'{dvf[...,1].min():.2f}:{dvf[...,1].max():.2f}, '
                  f'{dvf[...,2].min():.2f}:{dvf[...,2].max():.2f}',
                  flush=True)

        if split == 'val':
            val_pairs.append((ref, tgt, dvf))
        else:
            train_pairs.append((ref, tgt, dvf))

        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_path, dvf.astype(np.float32))

    # Always write CTs + lung mask into both splits (same anatomy volumes)
    for out_dir in (args.train_dir, args.val_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, ct in enumerate(cts, start=1):
            np.save(out_dir / f'CT_{i:02d}.npy', ct.astype(np.float32))
        np.save(out_dir / 'Mask_Lung.npy', mask.astype(np.uint8))

    print(f'Done. train pairs={len(train_pairs)}, val pairs={len(val_pairs)}')
    print(f'Train dir: {args.train_dir}')
    print(f'Val dir:   {args.val_dir}')


if __name__ == '__main__':
    main()
