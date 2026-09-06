"""FOV / CBCT source-side augmentation for ClinicalExperiments Experiment 3.

Modes (equal 1/4 when enabled):
  0 normal
  1 half-FOV (random L/R)
  2 CBCT noise (full FOV)
  3 half-FOV + CBCT noise

Applied after spatial crop to ref CT, target CT, and lung mask together.
"""

from __future__ import annotations

import numpy as np

MODE_NORMAL = 0
MODE_HALF_FOV = 1
MODE_CBCT = 2
MODE_HALF_FOV_CBCT = 3
MODE_NAMES = {
    MODE_NORMAL: "normal",
    MODE_HALF_FOV: "half_fov",
    MODE_CBCT: "cbct_noise",
    MODE_HALF_FOV_CBCT: "half_fov_cbct",
}


def half_fov_mask(shape, rng: np.random.Generator, axis: int = -1):
    """Boolean keep-mask: True = visible FOV."""
    keep = np.ones(shape, dtype=bool)
    w = shape[axis]
    jitter = int(rng.uniform(-0.1, 0.1) * w)
    cut = int(np.clip(w // 2 + jitter, w // 5, 4 * w // 5))
    # slicer for arbitrary axis
    sl = [slice(None)] * len(shape)
    if rng.random() < 0.5:
        sl[axis] = slice(0, cut)  # keep left, zero right
        keep_sl = slice(cut, None)
    else:
        sl[axis] = slice(cut, None)  # keep right
        keep_sl = slice(0, cut)
    # build keep: start True, zero the discarded side
    discard = [slice(None)] * len(shape)
    discard[axis] = keep_sl
    keep[tuple(discard)] = False
    return keep


def apply_cbct_noise(vol: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Crude CBCT-like corruption on [0,1] volume."""
    out = vol.astype(np.float32, copy=True)
    sigma = float(rng.uniform(0.03, 0.10))
    out = out + rng.normal(0.0, sigma, size=out.shape).astype(np.float32)
    n_streaks = int(rng.integers(5, 21))
    d, h, w = out.shape
    for _ in range(n_streaks):
        zi = int(rng.integers(0, d))
        yi = int(rng.integers(0, h))
        band = int(rng.integers(1, 3))
        amp = float(rng.uniform(-2.0 * sigma, 2.0 * sigma))
        out[zi : zi + band, yi : yi + band, :] += amp
    # cupping
    zz, yy, xx = np.ogrid[:d, :h, :w]
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt(((yy - cy) / max(cy, 1e-6)) ** 2 + ((xx - cx) / max(cx, 1e-6)) ** 2)
    cup_w = float(rng.uniform(0.0, 0.20))
    out = out + (cup_w * np.clip(r - 0.3, 0, None)).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def apply_fov_aug(
    reference_ct: np.ndarray,
    target_ct: np.ndarray,
    lung_mask: np.ndarray,
    target_dvf: np.ndarray,
    rng: np.random.Generator,
    mode: int | None = None,
):
    """
    reference_ct/target_ct/lung_mask: (S,S,S)
    target_dvf: (3,S,S,S)
    Returns possibly modified arrays + mode int.
    """
    if mode is None:
        mode = int(rng.integers(0, 4))

    ref = reference_ct
    tgt = target_ct
    mask = lung_mask.astype(np.float32, copy=True)
    dvf = target_dvf

    do_half = mode in (MODE_HALF_FOV, MODE_HALF_FOV_CBCT)
    do_noise = mode in (MODE_CBCT, MODE_HALF_FOV_CBCT)

    if do_half:
        keep = half_fov_mask(ref.shape, rng, axis=-1)
        fill = 0.0
        ref = ref.copy()
        tgt = tgt.copy()
        ref[~keep] = fill
        tgt[~keep] = fill
        mask = mask * keep.astype(np.float32)
        # zero DVF outside FOV (clean target; loss already masked)
        dvf = dvf.copy()
        dvf[:, ~keep] = 0.0

    if do_noise:
        ref = apply_cbct_noise(ref, rng)
        # independent noise on target (different acquisition realization)
        tgt = apply_cbct_noise(tgt, rng)

    return ref, tgt, mask, dvf, mode
