"""Load VoxelMap-style view config for QC CT / warp panels."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_VMC = Path('/home/abhishek/Documents/VoxelMap_Clinical')
if str(_VMC) not in sys.path:
    sys.path.insert(0, str(_VMC))

from ml.volume_view import VolumeViewConfig, extract_slice  # noqa: E402

DEFAULT_VIEW_CONFIG = Path(__file__).resolve().parents[1] / 'configs' / 'dvf_view_config.json'


def load_view_config(path=None) -> VolumeViewConfig:
    path = Path(path) if path else DEFAULT_VIEW_CONFIG
    return VolumeViewConfig.load_json(path)


def show_ct_slice(vol: np.ndarray, cfg: VolumeViewConfig) -> np.ndarray:
    return extract_slice(np.asarray(vol, dtype=np.float32), cfg)


def show_mag_slice(vec_chw: np.ndarray, cfg: VolumeViewConfig) -> np.ndarray:
    mag = np.linalg.norm(vec_chw, axis=0).astype(np.float32)
    return extract_slice(mag, cfg)
