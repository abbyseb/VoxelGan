"""Oracle per-patient amplitude A_p and train-pool normalization.

A_p_mm = q90 lung ‖u‖ in millimetres on the extreme pair (P0-A iso grid).
Training uses A_p_vox = A_p_mm / spacing_mm so targets match DVF iso-voxel units.
r_p = A_p_mm / mean(A_train_mm). Cond uses log(r_p).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

POP = Path(__file__).resolve().parents[3]  # PopulationStudy
DATA = POP / "data_iso"
E1 = Path(__file__).resolve().parents[1]


def mag(u_zyx3: np.ndarray) -> np.ndarray:
    return np.linalg.norm(u_zyx3.astype(np.float64), axis=-1)


def extreme_pair_A(pid: str, q: float = 90.0) -> tuple[float, float, str, float]:
    """Returns A_mm, A_vox, extreme_pair_name, spacing_mm."""
    d = DATA / pid / "all"
    mask_path = d / "Mask_Lung_unpadded.npy"
    if not mask_path.exists():
        mask_path = d / "Mask_Lung.npy"
    mask = np.load(mask_path) > 0
    spacing = 2.0
    meta_path = d / "meta.json"
    if meta_path.exists():
        spacing = float(json.loads(meta_path.read_text()).get("dvf_mm_per_voxel", 2.0))
    best_mean = -1.0
    best_name = "01_to_06"
    best_A_mm = 0.0
    for ref in range(1, 11):
        for tgt in range(1, 11):
            if ref == tgt:
                continue
            name = f"{ref:02d}_to_{tgt:02d}"
            u = np.load(d / f"{name}_pair.npy")
            if u.ndim == 4 and u.shape[0] == 3:
                u = np.moveaxis(u, 0, -1)
            m_vx = mag(u)[mask]
            mean_m = float((m_vx * spacing).mean())
            if mean_m > best_mean:
                best_mean = mean_m
                best_name = name
                best_A_mm = float(np.percentile(m_vx * spacing, q))
    return best_A_mm, best_A_mm / spacing, best_name, spacing


def compute_table(train_patients: list[str], all_patients: list[str] | None = None) -> dict:
    if all_patients is None:
        all_patients = [f"P{i}" for i in range(1, 10)]
    rows = {}
    spacing = 2.0
    for pid in all_patients:
        A_mm, A_vox, pair, spacing = extreme_pair_A(pid)
        rows[pid] = {
            "A_p_mm": A_mm,
            "A_p": A_vox,
            "extreme_pair": pair,
        }
    A_train = float(np.mean([rows[p]["A_p_mm"] for p in train_patients]))
    for pid, row in rows.items():
        row["r_p"] = row["A_p_mm"] / max(A_train, 1e-8)
        row["log_r_p"] = float(np.log(max(row["r_p"], 1e-8)))
    return {
        "q": 90.0,
        "units": "mm_and_iso_voxels",
        "spacing_mm": spacing,
        "grid": "data_iso 2mm 160³",
        "train_patients": train_patients,
        "A_train_mean_mm": A_train,
        "A_train_mean": A_train / spacing,
        "patients": rows,
    }


def load_amplitude(path: Path | None = None) -> dict:
    path = path or (E1 / "amplitudes.json")
    with open(path) as f:
        return json.load(f)


def patient_amp(table: dict, patient_tag: str) -> tuple[float, float]:
    """patient_tag like P03 → lookup P3. Returns (A_p_vox, log_r_p)."""
    pid = f"P{int(patient_tag[1:])}"
    row = table["patients"][pid]
    return float(row["A_p"]), float(row["log_r_p"])
