"""Shared packing recipe = the TCIA training data (synth_g160_r3/pooled): HU volume (any direction, resampled in physical LPS
space) -> 2 mm iso 160^3 grid, identity direction, origin = lung centroid - 160 mm (z,y,x), linear, outside -1000 HU ->
R3 layout arr.transpose(1, 0, 2)[::-1, ::-1, :] -> mu = (clip(HU, -1000, 3000) + 1000) * 2e-5 (raw file units, min 0).
Masks: nearest neighbour, same grid and layout."""
import numpy as np
import SimpleITK as sitk

SIZE, SP = 160, 2.0


def lung_centroid_zyx(mask_img: sitk.Image) -> np.ndarray:
    m = sitk.GetArrayFromImage(mask_img) > 0
    idx = np.argwhere(m)[:, ::-1].mean(0)                              # (x,y,z) continuous index
    return np.array(mask_img.TransformContinuousIndexToPhysicalPoint(tuple(idx)))[::-1]


def grid(origin_zyx):
    ref = sitk.Image(SIZE, SIZE, SIZE, sitk.sitkFloat32)
    ref.SetSpacing((SP, SP, SP)); ref.SetOrigin(tuple(float(v) for v in origin_zyx[::-1])); ref.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    return ref


def resample(img, ref, interp, default):
    return sitk.GetArrayFromImage(sitk.Resample(img, ref, sitk.Transform(), interp, default, sitk.sitkFloat32))


def to_r3(arr_zyx):
    return np.ascontiguousarray(arr_zyx.transpose(1, 0, 2)[::-1, ::-1, :])


def pack_ct(hu_img, ref):
    hu = resample(hu_img, ref, sitk.sitkLinear, -1000.0)
    return to_r3(((np.clip(hu, -1000, 3000) + 1000.0) * 2e-5).astype(np.float32))


def pack_mask(mask_img, ref):
    return to_r3((resample(mask_img, ref, sitk.sitkNearestNeighbor, 0.0) > 0.5).astype(np.uint8))
