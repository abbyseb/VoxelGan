"""TCIA4 phase-pair dataset: mmap the full 160³ DVF.

Same crop API as TCIA3. With im_size equal to the packed volume the crop
origin is (0, 0, 0), so each sample is the whole thorax.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from utilities.dataset import FOVAugPhasePairDataset, parse_prefixed_pair_name
from utilities.fov_aug import MODE_NAMES, apply_fov_aug


class FastMmapPhasePairDataset(FOVAugPhasePairDataset):
    """Same API as FOVAugPhasePairDataset; DVF via numpy memmap."""

    def __getitem__(self, idx):
        pair_idx = idx // self.patches_per_pair
        patch_idx = idx % self.patches_per_pair
        pair_name = self.pair_files[pair_idx]
        patient, ref_phase, target_phase = parse_prefixed_pair_name(pair_name)
        meta = self._patient_meta[patient]

        z0, y0, x0 = self._crop_params(meta, patch_idx)
        s = self.im_size

        reference_ct = self._load_ct(patient, ref_phase + 1)
        target_ct = self._load_ct(patient, target_phase + 1)
        reference_ct = np.asarray(
            reference_ct[z0 : z0 + s, y0 : y0 + s, x0 : x0 + s], dtype=np.float32
        )
        target_ct = np.asarray(
            target_ct[z0 : z0 + s, y0 : y0 + s, x0 : x0 + s], dtype=np.float32
        )
        lung_mask = np.asarray(
            meta["lung_full"][z0 : z0 + s, y0 : y0 + s, x0 : x0 + s], dtype=np.float32
        )

        path = os.path.join(self.im_dir, pair_name)
        dvf_mm = np.load(path, mmap_mode="r")
        if dvf_mm.ndim == 4 and dvf_mm.shape[0] == 3:
            dvf_crop = np.asarray(
                dvf_mm[:, z0 : z0 + s, y0 : y0 + s, x0 : x0 + s], dtype=np.float32
            )
            target_dvf = dvf_crop
        else:
            dvf_crop = np.asarray(
                dvf_mm[z0 : z0 + s, y0 : y0 + s, x0 : x0 + s, :], dtype=np.float32
            )
            target_dvf = np.empty((3, s, s, s), dtype=np.float32)
            target_dvf[0] = dvf_crop[:, :, :, 0]
            target_dvf[1] = dvf_crop[:, :, :, 1]
            target_dvf[2] = dvf_crop[:, :, :, 2]

        aug_mode = 0
        if self.fov_aug:
            rng = np.random.default_rng(
                self.aug_seed + int(idx) * 10007 + patch_idx * 17
            )
            reference_ct, target_ct, lung_mask, target_dvf, aug_mode = apply_fov_aug(
                reference_ct, target_ct, lung_mask, target_dvf, rng, mode=None
            )

        return {
            "reference_ct": torch.from_numpy(reference_ct[None, ...]),
            "target_ct": torch.from_numpy(target_ct[None, ...]),
            "lung_mask": torch.from_numpy(lung_mask[None, ...]),
            "ref_phase": torch.tensor(ref_phase, dtype=torch.long),
            "target_phase": torch.tensor(target_phase, dtype=torch.long),
            "target_dvf": torch.from_numpy(target_dvf),
            "patient": patient,
            "aug_mode": torch.tensor(aug_mode, dtype=torch.long),
            "aug_name": MODE_NAMES[aug_mode],
        }
