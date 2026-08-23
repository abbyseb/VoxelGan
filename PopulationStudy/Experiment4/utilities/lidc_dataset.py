"""LIDC 128³ volumes for Experiment 4 anatomy autoencoder.

Per-volume min-max, same as SPARE DVF loader. Lung-biased 64³ crops by default.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

E4 = Path(__file__).resolve().parents[1]
LIDC = E4 / "data" / "lidc128"


class LidcAnatomyDataset(Dataset):
    def __init__(
        self,
        patients,
        root=None,
        im_size=64,
        random_crop=True,
        patches_per_vol=16,
        min_lung_fraction=0.1,
    ):
        self.root = Path(root) if root else LIDC
        self.patients = list(patients)
        self.im_size = int(im_size)
        self.random_crop = random_crop
        self.patches_per_vol = max(1, int(patches_per_vol))
        self.min_lung_fraction = min_lung_fraction
        self._cache = {}
        for pid in self.patients:
            ct = np.load(self.root / pid / "CT.npy").astype(np.float32)
            ct = (ct - ct.min()) / (ct.max() - ct.min() + 1e-8)
            mask = (np.load(self.root / pid / "Mask_Lung.npy") > 0).astype(np.float32)
            self._cache[pid] = {
                "ct": ct,
                "mask": mask,
                "coords": np.argwhere(mask > 0),
            }

    def __len__(self):
        return len(self.patients) * self.patches_per_vol

    def _origin(self, pid, patch_idx):
        meta = self._cache[pid]
        d, h, w = meta["ct"].shape
        s = self.im_size
        if d == s and h == s and w == s:
            return 0, 0, 0
        if not self.random_crop:
            return (d - s) // 2, (h - s) // 2, (w - s) // 2
        coords = meta["coords"]
        lung = meta["mask"]
        for _ in range(32):
            cz, cy, cx = coords[np.random.randint(len(coords))]
            z0 = int(np.clip(cz - s // 2, 0, d - s))
            y0 = int(np.clip(cy - s // 2, 0, h - s))
            x0 = int(np.clip(cx - s // 2, 0, w - s))
            if float(lung[z0 : z0 + s, y0 : y0 + s, x0 : x0 + s].mean()) >= self.min_lung_fraction:
                return z0, y0, x0
        return (d - s) // 2, (h - s) // 2, (w - s) // 2

    def __getitem__(self, idx):
        pid = self.patients[idx // self.patches_per_vol]
        patch_idx = idx % self.patches_per_vol
        z0, y0, x0 = self._origin(pid, patch_idx)
        s = self.im_size
        meta = self._cache[pid]
        ct = meta["ct"][z0 : z0 + s, y0 : y0 + s, x0 : x0 + s]
        mask = meta["mask"][z0 : z0 + s, y0 : y0 + s, x0 : x0 + s]
        return {
            "ct": torch.from_numpy(ct[None]),
            "lung_mask": torch.from_numpy(mask[None]),
            "patient": pid,
        }


def lidc_split(manifest_path=None):
    path = Path(manifest_path) if manifest_path else LIDC / "manifest.json"
    man = json.loads(path.read_text())
    return man["train_patients"], man["val_patients"]
