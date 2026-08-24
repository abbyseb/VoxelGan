"""Multi-patient dataset with oracle amp shape targets + inhale/exhale δ_tgt.

Returns target_dvf / A_p, log_r_p, and δ_tgt ∈ {+1,−1} for the target phase.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from utilities.amplitude import load_amplitude, patient_amp

E4 = Path(__file__).resolve().parents[1]


def parse_prefixed_pair_name(name: str):
    stem = name.replace("_pair.npy", "")
    patient, rest = stem.split("_", 1)
    ref_s, tgt_s = rest.split("_to_")
    return patient, int(ref_s) - 1, int(tgt_s) - 1


def load_phase_limbs(path: Path | None = None) -> dict:
    path = path or (E4 / "phase_limbs.json")
    with open(path) as f:
        return json.load(f)["patients"]


class MultiPatientPhasePairDatasetAmp(Dataset):
    def __init__(
        self,
        im_dir,
        pair_files=None,
        im_size=64,
        random_crop=True,
        patches_per_pair=1,
        min_lung_fraction=0.1,
        amplitude_table=None,
        phase_limbs=None,
    ):
        self.im_dir = im_dir
        self.im_size = im_size
        self.random_crop = random_crop
        self.patches_per_pair = max(1, int(patches_per_pair))
        self.min_lung_fraction = min_lung_fraction
        self.amp_table = amplitude_table or load_amplitude()
        self.phase_limbs = phase_limbs or load_phase_limbs()

        if pair_files is None:
            pair_files = sorted(
                n for n in os.listdir(self.im_dir) if n.endswith("_pair.npy")
            )
        self.pair_files = list(pair_files)
        if not self.pair_files:
            raise FileNotFoundError(f"no pair files in {im_dir}")

        self._ct_cache = {}
        self._patient_meta = {}
        patients = sorted({parse_prefixed_pair_name(n)[0] for n in self.pair_files})
        for patient in patients:
            lung_path = os.path.join(self.im_dir, f"{patient}_Mask_Lung.npy")
            lung_full = (np.load(lung_path) > 0).astype(np.float32)
            A_p, log_r = patient_amp(self.amp_table, patient)
            pid = f"P{int(patient[1:])}"
            delta = list(self.phase_limbs[pid]["delta_tgt"])
            self._patient_meta[patient] = {
                "lung_full": lung_full,
                "lung_coords": np.argwhere(lung_full > 0),
                "vol_shape": lung_full.shape,
                "grid_origins": self._build_grid_origins(lung_full, self.patches_per_pair),
                "A_p": A_p,
                "log_r_p": log_r,
                "delta_tgt": delta,  # length 10, index by 0-based phase
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
            frac = float(lung_full[z0 : z0 + s, y0 : y0 + s, x0 : x0 + s].mean())
            scored.append((frac, (z0, y0, x0)))
        scored.sort(reverse=True)
        origins = [o for _, o in scored[:n_patches]]
        while len(origins) < n_patches:
            origins.append(origins[len(origins) % max(len(scored), 1)] if scored else (0, 0, 0))
        return origins

    def _lung_biased_crop(self, meta):
        d, h, w = meta["vol_shape"]
        s = self.im_size
        if d == s and h == s and w == s:
            return 0, 0, 0
        coords = meta["lung_coords"]
        lung_full = meta["lung_full"]
        for _ in range(32):
            cz, cy, cx = coords[np.random.randint(len(coords))]
            z0 = int(np.clip(cz - s // 2, 0, d - s))
            y0 = int(np.clip(cy - s // 2, 0, h - s))
            x0 = int(np.clip(cx - s // 2, 0, w - s))
            frac = float(lung_full[z0 : z0 + s, y0 : y0 + s, x0 : x0 + s].mean())
            if frac >= self.min_lung_fraction:
                return z0, y0, x0
        return (d - s) // 2, (h - s) // 2, (w - s) // 2

    def _crop_params(self, meta, patch_idx):
        if self.random_crop:
            return self._lung_biased_crop(meta)
        return meta["grid_origins"][patch_idx % len(meta["grid_origins"])]

    def _load_ct(self, patient, phase_1idx):
        key = (patient, phase_1idx)
        cached = self._ct_cache.get(key)
        if cached is not None:
            return cached
        path = os.path.join(self.im_dir, f"{patient}_CT_{phase_1idx:02d}.npy")
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
        A_p = max(float(meta["A_p"]), 1e-6)

        reference_ct = self._load_ct(patient, ref_phase + 1)
        target_ct = self._load_ct(patient, target_phase + 1)
        dvf_raw = np.load(os.path.join(self.im_dir, pair_name)).astype(np.float32)
        if dvf_raw.ndim == 4 and dvf_raw.shape[0] == 3:
            dvf_zyx3 = np.moveaxis(dvf_raw, 0, -1)
        else:
            dvf_zyx3 = dvf_raw

        z0, y0, x0 = self._crop_params(meta, patch_idx)
        s = self.im_size
        reference_ct = reference_ct[z0 : z0 + s, y0 : y0 + s, x0 : x0 + s]
        target_ct = target_ct[z0 : z0 + s, y0 : y0 + s, x0 : x0 + s]
        dvf_zyx3 = dvf_zyx3[z0 : z0 + s, y0 : y0 + s, x0 : x0 + s]
        lung_mask = meta["lung_full"][z0 : z0 + s, y0 : y0 + s, x0 : x0 + s]

        # shape-normalized target
        target_dvf = np.zeros((3, s, s, s), dtype=np.float32)
        target_dvf[0] = dvf_zyx3[:, :, :, 0] / A_p
        target_dvf[1] = dvf_zyx3[:, :, :, 1] / A_p
        target_dvf[2] = dvf_zyx3[:, :, :, 2] / A_p

        delta = float(meta["delta_tgt"][target_phase])
        return {
            "reference_ct": torch.from_numpy(np.asarray(reference_ct, dtype=np.float32)[None, ...]),
            "target_ct": torch.from_numpy(np.asarray(target_ct, dtype=np.float32)[None, ...]),
            "lung_mask": torch.from_numpy(np.asarray(lung_mask, dtype=np.float32)[None, ...]),
            "ref_phase": torch.tensor(ref_phase, dtype=torch.long),
            "target_phase": torch.tensor(target_phase, dtype=torch.long),
            "target_dvf": torch.from_numpy(target_dvf),
            "log_r_p": torch.tensor(meta["log_r_p"], dtype=torch.float32),
            "delta_tgt": torch.tensor(delta, dtype=torch.float32),
            "A_p": torch.tensor(A_p, dtype=torch.float32),
            "patient": patient,
        }
