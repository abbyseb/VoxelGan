"""Multi-patient phase-pair dataset for PopulationStudy Experiment 3.

Pooled layout (patient prefix on every file so CTs/DVFs/masks cannot mix):

  P01_CT_01.npy … P01_CT_10.npy
  P01_Mask_Lung.npy
  P01_01_to_02_pair.npy   → loads only P01_CT_01, P01_CT_02, P01_Mask_Lung

Phase filenames are 1-indexed (SPARE). Returned phase tensors are 0-indexed
longs for the CRB condition vector. No patient-ID embedding.

Dense patch sampling: each listed pair is repeated `patches_per_pair` times
per epoch. Train crops are lung-biased random 64³ windows; val uses a fixed
per-patient grid for stable metrics.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset


def parse_prefixed_pair_name(name: str):
    """'P01_01_to_02_pair.npy' → ('P01', 0, 1)  (phases 0-indexed)."""
    stem = name.replace('_pair.npy', '')
    patient, rest = stem.split('_', 1)
    ref_s, tgt_s = rest.split('_to_')
    return patient, int(ref_s) - 1, int(tgt_s) - 1


class MultiPatientPhasePairDataset(Dataset):
    def __init__(
        self,
        im_dir,
        pair_files=None,
        im_size=64,
        random_crop=True,
        patches_per_pair=1,
        min_lung_fraction=0.1,
    ):
        self.im_dir = im_dir
        self.im_size = im_size
        self.random_crop = random_crop
        self.patches_per_pair = max(1, int(patches_per_pair))
        self.min_lung_fraction = min_lung_fraction

        if pair_files is None:
            pair_files = sorted(
                n for n in os.listdir(self.im_dir) if n.endswith('_pair.npy')
            )
        self.pair_files = list(pair_files)
        if not self.pair_files:
            raise FileNotFoundError(f'no pair files in {im_dir}')

        self._ct_cache = {}
        self._patient_meta = {}
        patients = sorted({parse_prefixed_pair_name(n)[0] for n in self.pair_files})
        for patient in patients:
            lung_path = os.path.join(self.im_dir, f'{patient}_Mask_Lung.npy')
            lung_full = (np.load(lung_path) > 0).astype(np.float32)
            self._patient_meta[patient] = {
                'lung_full': lung_full,
                'lung_coords': np.argwhere(lung_full > 0),
                'vol_shape': lung_full.shape,
                'grid_origins': self._build_grid_origins(
                    lung_full, self.patches_per_pair
                ),
            }

    def __len__(self):
        return len(self.pair_files) * self.patches_per_pair

    def _build_grid_origins(self, lung_full, n_patches):
        d, h, w = lung_full.shape
        s = self.im_size
        if d == s and h == s and w == s:
            return [(0, 0, 0)] * n_patches

        max_z, max_y, max_x = d - s, h - s, w - s
        if n_patches == 1:
            return [(max_z // 2, max_y // 2, max_x // 2)]

        n_axis = int(np.ceil(n_patches ** (1.0 / 3.0)))
        zs = np.linspace(0, max_z, n_axis, dtype=int)
        ys = np.linspace(0, max_y, n_axis, dtype=int)
        xs = np.linspace(0, max_x, n_axis, dtype=int)
        origins = [(int(z), int(y), int(x)) for z in zs for y in ys for x in xs]
        scored = []
        for z0, y0, x0 in origins:
            frac = float(lung_full[z0:z0 + s, y0:y0 + s, x0:x0 + s].mean())
            scored.append((frac, (z0, y0, x0)))
        scored.sort(reverse=True)
        origins = [o for _, o in scored[:n_patches]]
        while len(origins) < n_patches:
            origins.append(origins[len(origins) % max(len(scored), 1)] if scored else (0, 0, 0))
        return origins

    def _lung_biased_crop(self, meta):
        d, h, w = meta['vol_shape']
        s = self.im_size
        if d == s and h == s and w == s:
            return 0, 0, 0
        coords = meta['lung_coords']
        lung_full = meta['lung_full']
        for _ in range(32):
            cz, cy, cx = coords[np.random.randint(len(coords))]
            z0 = int(np.clip(cz - s // 2, 0, d - s))
            y0 = int(np.clip(cy - s // 2, 0, h - s))
            x0 = int(np.clip(cx - s // 2, 0, w - s))
            frac = float(lung_full[z0:z0 + s, y0:y0 + s, x0:x0 + s].mean())
            if frac >= self.min_lung_fraction:
                return z0, y0, x0
        return (d - s) // 2, (h - s) // 2, (w - s) // 2

    def _crop_params(self, meta, patch_idx):
        if self.random_crop:
            return self._lung_biased_crop(meta)
        return meta['grid_origins'][patch_idx % len(meta['grid_origins'])]

    def _load_ct(self, patient, phase_1idx):
        key = (patient, phase_1idx)
        cached = self._ct_cache.get(key)
        if cached is not None:
            return cached
        path = os.path.join(self.im_dir, f'{patient}_CT_{phase_1idx:02d}.npy')
        x = np.load(path).astype(np.float32)
        x = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)
        self._ct_cache[key] = x
        return x

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
        reference_ct = reference_ct[z0:z0 + s, y0:y0 + s, x0:x0 + s]
        target_ct = target_ct[z0:z0 + s, y0:y0 + s, x0:x0 + s]
        dvf_zyx3 = dvf_zyx3[z0:z0 + s, y0:y0 + s, x0:x0 + s]
        lung_mask = meta['lung_full'][z0:z0 + s, y0:y0 + s, x0:x0 + s]

        target_dvf = np.zeros((3, s, s, s), dtype=np.float32)
        target_dvf[0] = dvf_zyx3[:, :, :, 0]
        target_dvf[1] = dvf_zyx3[:, :, :, 1]
        target_dvf[2] = dvf_zyx3[:, :, :, 2]

        return {
            'reference_ct': torch.from_numpy(np.asarray(reference_ct, dtype=np.float32)[None, ...]),
            'target_ct': torch.from_numpy(np.asarray(target_ct, dtype=np.float32)[None, ...]),
            'lung_mask': torch.from_numpy(np.asarray(lung_mask, dtype=np.float32)[None, ...]),
            'ref_phase': torch.tensor(ref_phase, dtype=torch.long),
            'target_phase': torch.tensor(target_phase, dtype=torch.long),
            'target_dvf': torch.from_numpy(target_dvf),
            'patient': patient,
        }


def group_pairs_by_patient(pair_files):
    """Split prefixed pair names into { 'P01': [...], ... }."""
    by = {}
    for name in pair_files:
        patient, _, _ = parse_prefixed_pair_name(name)
        by.setdefault(patient, []).append(name)
    return by
