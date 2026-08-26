"""Dataset for ClinicalExperiments Experiment 3 — cyclic + FOV/CBCT aug.

Train: random FOV mode @ 1/4 each. Val: normal only (fov_aug=False).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch

E3 = Path(__file__).resolve().parents[1]
IE1 = E3.parents[1] / "InitialExperiments" / "Experiment1"

# Load IE1 dataset by path to avoid clashing with this package's utilities.dataset
_spec = importlib.util.spec_from_file_location(
    "ie1_phase_dataset", IE1 / "utilities" / "dataset.py"
)
_ie1 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_ie1)
MultiPatientPhasePairDataset = _ie1.MultiPatientPhasePairDataset
parse_prefixed_pair_name = _ie1.parse_prefixed_pair_name

if str(E3) not in sys.path:
    sys.path.insert(0, str(E3))

from utilities.fov_aug import MODE_NAMES, apply_fov_aug  # noqa: E402


class FOVAugPhasePairDataset(MultiPatientPhasePairDataset):
    def __init__(self, *args, fov_aug: bool = False, aug_seed: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.fov_aug = bool(fov_aug)
        self.aug_seed = int(aug_seed)

    def __getitem__(self, idx):
        pair_idx = idx // self.patches_per_pair
        patch_idx = idx % self.patches_per_pair
        pair_name = self.pair_files[pair_idx]
        patient, ref_phase, target_phase = parse_prefixed_pair_name(pair_name)
        meta = self._patient_meta[patient]

        reference_ct = self._load_ct(patient, ref_phase + 1)
        target_ct = self._load_ct(patient, target_phase + 1)

        dvf_raw = np.load(os.path.join(self.im_dir, pair_name)).astype(np.float32)
        if dvf_raw.ndim == 4 and dvf_raw.shape[0] == 3:
            dvf_zyx3 = np.moveaxis(dvf_raw, 0, -1)
        else:
            dvf_zyx3 = dvf_raw

        z0, y0, x0 = self._crop_params(meta, patch_idx)
        s = self.im_size
        reference_ct = reference_ct[z0 : z0 + s, y0 : y0 + s, x0 : x0 + s].copy()
        target_ct = target_ct[z0 : z0 + s, y0 : y0 + s, x0 : x0 + s].copy()
        dvf_zyx3 = dvf_zyx3[z0 : z0 + s, y0 : y0 + s, x0 : x0 + s]
        lung_mask = meta["lung_full"][z0 : z0 + s, y0 : y0 + s, x0 : x0 + s].astype(
            np.float32
        )

        target_dvf = np.zeros((3, s, s, s), dtype=np.float32)
        target_dvf[0] = dvf_zyx3[:, :, :, 0]
        target_dvf[1] = dvf_zyx3[:, :, :, 1]
        target_dvf[2] = dvf_zyx3[:, :, :, 2]

        aug_mode = 0
        if self.fov_aug:
            rng = np.random.default_rng(
                self.aug_seed + int(idx) * 10007 + patch_idx * 17
            )
            reference_ct, target_ct, lung_mask, target_dvf, aug_mode = apply_fov_aug(
                reference_ct, target_ct, lung_mask, target_dvf, rng, mode=None
            )

        return {
            "reference_ct": torch.from_numpy(
                np.asarray(reference_ct, dtype=np.float32)[None, ...]
            ),
            "target_ct": torch.from_numpy(
                np.asarray(target_ct, dtype=np.float32)[None, ...]
            ),
            "lung_mask": torch.from_numpy(
                np.asarray(lung_mask, dtype=np.float32)[None, ...]
            ),
            "ref_phase": torch.tensor(ref_phase, dtype=torch.long),
            "target_phase": torch.tensor(target_phase, dtype=torch.long),
            "target_dvf": torch.from_numpy(target_dvf),
            "patient": patient,
            "aug_mode": torch.tensor(aug_mode, dtype=torch.long),
            "aug_name": MODE_NAMES[aug_mode],
        }
