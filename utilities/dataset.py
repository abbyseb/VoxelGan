"""Phase-pair dataset for Elastix-supervised DVF training.

Expected layout under im_dir:
  CT_01.npy … CT_10.npy            # per-phase volumes, shape (D, H, W)
  Mask_Lung.npy                    # lung mask, same grid as CTs
  {ref:02d}_to_{tgt:02d}_pair.npy  # Elastix DVF, (D, H, W, 3) or (3, D, H, W)

Phase filenames are 1-indexed (SPARE convention). Returned phase tensors are
0-indexed longs for the FiLM embedding table.

Dense patch sampling: each phase-pair is repeated `patches_per_pair` times
per epoch. Train crops are lung-biased random 64³ windows; val uses a fixed
grid (or center if patches_per_pair==1) for stable metrics.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset


class PhasePairDataset(Dataset):
    def __init__(
        self,
        im_dir=None,
        im_size=None,
        held_out_phases=None,
        random_crop=True,
        patches_per_pair=1,
        min_lung_fraction=0.1,
    ):
        self.im_dir = im_dir
        self.im_size = im_size
        self.random_crop = random_crop
        self.patches_per_pair = max(1, int(patches_per_pair))
        self.min_lung_fraction = min_lung_fraction
        # held_out_phases: 0-indexed phase ids to exclude (leave-phase-out)
        self.held_out_phases = set(held_out_phases or [])

        pair_names = sorted(
            n for n in os.listdir(self.im_dir) if n.endswith('_pair.npy')
        )
        self.pair_files = []
        for name in pair_names:
            ref_phase, target_phase = self._parse_pair_name(name)
            if ref_phase in self.held_out_phases or target_phase in self.held_out_phases:
                continue
            self.pair_files.append(name)

        lung_path = os.path.join(self.im_dir, 'Mask_Lung.npy')
        self.lung_full = (np.load(lung_path) > 0).astype(np.float32)
        self.lung_coords = np.argwhere(self.lung_full > 0)
        self.vol_shape = self.lung_full.shape
        self.grid_origins = self._build_grid_origins(self.patches_per_pair)

    @staticmethod
    def _parse_pair_name(name):
        # e.g. "03_to_07_pair.npy" → phases 3, 7 (1-indexed file) → 2, 6 (0-indexed)
        stem = name.replace('_pair.npy', '')
        ref_s, tgt_s = stem.split('_to_')
        return int(ref_s) - 1, int(tgt_s) - 1

    def __len__(self):
        return len(self.pair_files) * self.patches_per_pair

    def _build_grid_origins(self, n_patches):
        """Fixed crop origins for val / deterministic sampling."""
        d, h, w = self.vol_shape
        s = self.im_size
        if d == s and h == s and w == s:
            return [(0, 0, 0)] * n_patches

        max_z, max_y, max_x = d - s, h - s, w - s
        if n_patches == 1:
            return [(max_z // 2, max_y // 2, max_x // 2)]

        # 3D lattice covering the volume, truncated/padded to n_patches
        n_axis = int(np.ceil(n_patches ** (1.0 / 3.0)))
        zs = np.linspace(0, max_z, n_axis, dtype=int)
        ys = np.linspace(0, max_y, n_axis, dtype=int)
        xs = np.linspace(0, max_x, n_axis, dtype=int)
        origins = [(int(z), int(y), int(x)) for z in zs for y in ys for x in xs]
        # prefer high-lung windows first
        scored = []
        for z0, y0, x0 in origins:
            frac = float(self.lung_full[z0:z0 + s, y0:y0 + s, x0:x0 + s].mean())
            scored.append((frac, (z0, y0, x0)))
        scored.sort(reverse=True)
        origins = [o for _, o in scored[:n_patches]]
        while len(origins) < n_patches:
            origins.append(origins[len(origins) % max(len(scored), 1)] if scored else (0, 0, 0))
        return origins

    def _lung_biased_crop(self):
        """Random crop centered near a lung voxel; reject low-lung windows."""
        d, h, w = self.vol_shape
        s = self.im_size
        if d == s and h == s and w == s:
            return 0, 0, 0

        for _ in range(32):
            cz, cy, cx = self.lung_coords[np.random.randint(len(self.lung_coords))]
            z0 = int(np.clip(cz - s // 2, 0, d - s))
            y0 = int(np.clip(cy - s // 2, 0, h - s))
            x0 = int(np.clip(cx - s // 2, 0, w - s))
            frac = float(self.lung_full[z0:z0 + s, y0:y0 + s, x0:x0 + s].mean())
            if frac >= self.min_lung_fraction:
                return z0, y0, x0

        # fallback: center
        return (d - s) // 2, (h - s) // 2, (w - s) // 2

    def _crop_params(self, patch_idx):
        if self.random_crop:
            return self._lung_biased_crop()
        return self.grid_origins[patch_idx % len(self.grid_origins)]

    def __getitem__(self, idx):
        pair_idx = idx // self.patches_per_pair
        patch_idx = idx % self.patches_per_pair
        pair_name = self.pair_files[pair_idx]

        # Find reference / target phase indices (0-indexed for FiLM)
        ref_phase, target_phase = self._parse_pair_name(pair_name)

        # Find reference CT
        ref_ct_path = os.path.join(self.im_dir, f'CT_{ref_phase + 1:02d}.npy')
        reference_ct = np.load(ref_ct_path).astype(np.float32)
        reference_ct = (reference_ct - np.min(reference_ct)) / (
            np.max(reference_ct) - np.min(reference_ct) + 1e-8
        )

        # Find target CT (needed for image-similarity loss)
        tgt_ct_path = os.path.join(self.im_dir, f'CT_{target_phase + 1:02d}.npy')
        target_ct = np.load(tgt_ct_path).astype(np.float32)
        target_ct = (target_ct - np.min(target_ct)) / (
            np.max(target_ct) - np.min(target_ct) + 1e-8
        )

        # Find target DVF (Elastix ground truth for this phase pair)
        dvf_raw = np.load(os.path.join(self.im_dir, pair_name)).astype(np.float32)
        if dvf_raw.ndim == 4 and dvf_raw.shape[0] == 3:
            dvf_zyx3 = np.moveaxis(dvf_raw, 0, -1)
        else:
            dvf_zyx3 = dvf_raw

        lung_mask = self.lung_full

        z0, y0, x0 = self._crop_params(patch_idx)
        s = self.im_size
        reference_ct = reference_ct[z0:z0 + s, y0:y0 + s, x0:x0 + s]
        target_ct = target_ct[z0:z0 + s, y0:y0 + s, x0:x0 + s]
        dvf_zyx3 = dvf_zyx3[z0:z0 + s, y0:y0 + s, x0:x0 + s]
        lung_mask = lung_mask[z0:z0 + s, y0:y0 + s, x0:x0 + s]

        target_dvf = np.zeros((3, s, s, s), dtype=np.float32)
        target_dvf[0] = dvf_zyx3[:, :, :, 0]
        target_dvf[1] = dvf_zyx3[:, :, :, 1]
        target_dvf[2] = dvf_zyx3[:, :, :, 2]

        reference_ct = np.asarray(reference_ct, dtype=np.float32)[None, ...]
        target_ct = np.asarray(target_ct, dtype=np.float32)[None, ...]
        lung_mask = np.asarray(lung_mask, dtype=np.float32)[None, ...]

        data = {
            'reference_ct': torch.from_numpy(reference_ct),
            'target_ct': torch.from_numpy(target_ct),
            'lung_mask': torch.from_numpy(lung_mask),
            'ref_phase': torch.tensor(ref_phase, dtype=torch.long),
            'target_phase': torch.tensor(target_phase, dtype=torch.long),
            'target_dvf': torch.from_numpy(target_dvf),
        }
        return data
