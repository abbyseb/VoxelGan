#!/usr/bin/env python3
"""DIR-Lab oracle TRE for TCIA3 on the training grid.

The old test path stretched each R3 CT to 160³ and scaled brightness with
norm_mu. This script resamples to 2 mm, keeps a 160³ cube centred on the
lung, and min-maxes µ the same way the trainer does. The weights are not
changed. T00 landmarks are moved by one step of the checked pull convention:
the network sees CT_06 with phases (5, 0), and pred = lm00 + u(lm00).

  CUDA_VISIBLE_DEVICES=1 python scripts/eval_dir_tcia3_iso2mm.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from scipy.ndimage import map_coordinates

DIR_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIR_EXP / "scripts"))
from dirlab_tre import CASE_INFO, landmarks_300, landmarks_75, stats, tre_mm  # noqa: E402

VOXEL = DIR_EXP.parent
A1 = DIR_EXP / "arms" / "A1_oracle_dirlab" / "runs"
CKPT = (
    VOXEL
    / "PopulationStudy/ClinicalExperiments/Grid160/TCIA3/DecoderCRB/checkpoints/epoch_100.pt"
)
NET = VOXEL / "PopulationStudy/ClinicalExperiments/Grid160/Experiment1"
TCIA_MU = Path(
    "/media/abhishek/3CCA3CADCA3C6574/TCIA_4D-Lung/packed_r3_hu/S1/all/CT_01_mu.npy"
)
TCIA_LUNG = Path(
    "/media/abhishek/3CCA3CADCA3C6574/TCIA_4D-Lung/packed_r3_hu/S1/all/Mask_Lung.npy"
)
OUT = (
    VOXEL
    / "PopulationStudy/ClinicalExperiments/Grid160/TCIA3/DecoderCRB/plots/qc_dir_oracle/tre75_ep100_iso2mm_minmax.json"
)

SPACING = 2.0
SIZE = 160
HALF = 0.5 * SIZE * SPACING
MU_WATER = 0.02


def minmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def hu_to_mu(hu: np.ndarray) -> np.ndarray:
    return ((hu.astype(np.float32) + 1000.0) * (MU_WATER / 1000.0)).astype(np.float32)


def brightness(mu: np.ndarray, lung: np.ndarray) -> dict:
    """Where water (µ=0.02) and the lung sit after the training min-max."""
    norm = minmax(mu)
    lung = lung > 0
    return {
        "mu_min": float(mu.min()),
        "mu_max": float(mu.max()),
        "water_after_minmax": float((MU_WATER - mu.min()) / (mu.max() - mu.min() + 1e-8)),
        "lung_median_after_minmax": float(np.median(norm[lung])) if lung.any() else None,
    }


def official_to_r3_index(xyz: np.ndarray, ny: int, nz: int) -> np.ndarray:
    """Official (x, y, z) = (Left, Posterior, Superior) → A1 itk index (Left, Superior, Anterior)."""
    out = np.empty_like(xyz, dtype=np.float64)
    out[:, 0] = xyz[:, 0]
    out[:, 1] = (nz - 1) - xyz[:, 2]
    out[:, 2] = (ny - 1) - xyz[:, 1]
    return out


def index_to_physical(idx_xyz: np.ndarray, origin_xyz: np.ndarray, spacing_xyz: np.ndarray) -> np.ndarray:
    return origin_xyz + idx_xyz * spacing_xyz


def physical_to_iso(phys_xyz: np.ndarray, origin_iso_xyz: np.ndarray) -> np.ndarray:
    return (phys_xyz - origin_iso_xyz) / SPACING


def lung_centroid_xyz(mask_zyx: np.ndarray, origin_xyz: np.ndarray, spacing_xyz: np.ndarray) -> np.ndarray:
    iz, iy, ix = np.where(mask_zyx > 0)
    if ix.size == 0:
        raise RuntimeError("empty lung mask")
    pts = np.stack([ix, iy, iz], axis=1).astype(np.float64)
    return index_to_physical(pts, origin_xyz, spacing_xyz).mean(axis=0)


def resample_hu(hu_zyx: np.ndarray, origin_xyz, spacing_xyz, origin_iso_xyz) -> np.ndarray:
    ii = np.arange(SIZE, dtype=np.float64)
    zz, yy, xx = np.meshgrid(
        origin_iso_xyz[2] + ii * SPACING,
        origin_iso_xyz[1] + ii * SPACING,
        origin_iso_xyz[0] + ii * SPACING,
        indexing="ij",
    )
    iz = (zz - origin_xyz[2]) / spacing_xyz[2]
    iy = (yy - origin_xyz[1]) / spacing_xyz[1]
    ix = (xx - origin_xyz[0]) / spacing_xyz[0]
    samp = map_coordinates(
        hu_zyx,
        np.stack([iz.ravel(), iy.ravel(), ix.ravel()], axis=0),
        order=1,
        mode="constant",
        cval=-1000.0,
    )
    return samp.reshape(SIZE, SIZE, SIZE).astype(np.float32)


def sample_u(dvf_zyx3: np.ndarray, idx_xyz: np.ndarray) -> np.ndarray:
    """dvf (Z,Y,X,3) channels (dx,dy,dz). Returns (N,3) in xyz."""
    pts = np.stack([idx_xyz[:, 2], idx_xyz[:, 1], idx_xyz[:, 0]], axis=0)
    out = np.stack(
        [map_coordinates(dvf_zyx3[..., c], pts, order=1, mode="nearest") for c in range(3)],
        axis=1,
    )
    return out


def load_case_image(case: int):
    sid = f"DIR_C{case:02d}"
    train = A1 / sid / sid / "train"
    ct = sitk.ReadImage(str(train / "CT_06.mha"))
    mask = sitk.ReadImage(str(train / "Mask_Lung.mha"))
    if ct.GetSize() != mask.GetSize() or ct.GetSpacing() != mask.GetSpacing():
        raise RuntimeError(f"{sid} CT and lung mask grids differ")
    hu = sitk.GetArrayFromImage(ct).astype(np.float32)
    lung = sitk.GetArrayFromImage(mask) > 0
    origin = np.array(ct.GetOrigin(), dtype=np.float64)
    spacing = np.array(ct.GetSpacing(), dtype=np.float64)
    return hu, lung, origin, spacing


def pack_case(case: int) -> dict:
    (nx, ny, nz), _ = CASE_INFO[case]
    hu, lung, origin, spacing = load_case_image(case)
    # A1 image is already R3: itk size (Left, Superior, Anterior) = (nx, nz, ny)
    if tuple(int(v) for v in (hu.shape[2], hu.shape[1], hu.shape[0])) != (nx, nz, ny):
        raise RuntimeError(
            f"case {case}: CT shape zyx {hu.shape} is not R3 (nx,nz,ny)=({nx},{nz},{ny})"
        )
    centroid = lung_centroid_xyz(lung, origin, spacing)
    origin_iso = centroid - HALF
    hu_iso = resample_hu(hu, origin, spacing, origin_iso)
    mu = hu_to_mu(hu_iso)
    return {
        "hu_iso": hu_iso,
        "mu": mu,
        "norm": minmax(mu),
        "origin_iso_xyz": origin_iso,
        "centroid_xyz": centroid,
        "native_spacing_xyz": spacing,
        "nx": nx,
        "ny": ny,
        "nz": nz,
    }


def landmarks_iso(case: int, phase: str, which: str, pack: dict) -> np.ndarray:
    official = landmarks_75(case, phase) if which == "75" else landmarks_300(case, phase)
    r3 = official_to_r3_index(official, pack["ny"], pack["nz"])
    # physical position on the A1 image. The A1 origin/spacing are the R3 grid.
    _, _, origin, spacing = load_case_image(case)
    phys = index_to_physical(r3, origin, spacing)
    return physical_to_iso(phys, pack["origin_iso_xyz"])


def inside(idx: np.ndarray) -> np.ndarray:
    return np.all((idx >= 0.0) & (idx <= (SIZE - 1)), axis=1)


def load_generator(device: torch.device):
    if str(NET) not in sys.path:
        sys.path.insert(0, str(NET))
    from networks.generator_crb_dec import UNetCRBDecoder

    g = UNetCRBDecoder(im_size=SIZE, n_phases=10)
    raw = torch.load(str(CKPT), map_location=device, weights_only=False)
    state = raw["generator"] if isinstance(raw, dict) and "generator" in raw else raw
    g.load_state_dict(state, strict=True)
    g.to(device).eval()
    return g


def predict(g, device, mu: np.ndarray) -> np.ndarray:
    """A1 puts the diaphragm at high Y. TCIA training cubes put the apex there.

    Flip that axis for the network, then flip the field back so landmarks stay
    in the A1 index frame.
    """
    flipped = np.ascontiguousarray(mu[:, ::-1, :])
    x = torch.from_numpy(minmax(flipped)[None, None]).to(device)
    ref = torch.tensor([5], dtype=torch.long, device=device)
    tgt = torch.tensor([0], dtype=torch.long, device=device)
    with torch.no_grad():
        dvf = g(x, ref, tgt)[0].detach().cpu().numpy()
    field = np.moveaxis(dvf, 0, -1).astype(np.float32)
    field = np.ascontiguousarray(field[:, ::-1, :, :])
    field[..., 1] *= -1
    return field


def score_set(dvf, case: int, which: str, pack: dict) -> dict:
    lm00 = landmarks_iso(case, "T00", which, pack)
    lm50 = landmarks_iso(case, "T50", which, pack)
    keep = inside(lm00) & inside(lm50)
    pred = lm00 + sample_u(dvf, lm00)
    err = np.linalg.norm((pred - lm50) * SPACING, axis=1)
    ident = np.linalg.norm((lm00 - lm50) * SPACING, axis=1)
    official_sp = CASE_INFO[case][1]
    off00 = landmarks_75(case, "T00") if which == "75" else landmarks_300(case, "T00")
    off50 = landmarks_75(case, "T50") if which == "75" else landmarks_300(case, "T50")
    official_ident = float(tre_mm(off00, off50, official_sp).mean())
    return {
        "registered": stats(err),
        "registered_inside": stats(err[keep]) if keep.any() else None,
        "identity_iso_mm": float(ident.mean()),
        "identity_official_mm": official_ident,
        "n_inside": int(keep.sum()),
        "n": int(len(err)),
    }


def main() -> None:
    os.environ.setdefault("DIRLAB_ROOT", str(DIR_EXP / "data" / "dirlab_packs"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tcia_mu = np.load(TCIA_MU)
    tcia_lung = np.load(TCIA_LUNG)
    print("brightness TCIA S1", brightness(tcia_mu, tcia_lung), flush=True)

    # Coordinate check on case 1 before the network is asked to matter.
    probe = pack_case(1)
    print("brightness DIR C01", brightness(probe["mu"], (probe["hu_iso"] > -700) & (probe["hu_iso"] < -200)), flush=True)
    ident = score_set(np.zeros((SIZE, SIZE, SIZE, 3), np.float32), 1, "300", probe)
    gap = abs(ident["identity_iso_mm"] - ident["identity_official_mm"])
    print(
        f"C01 identity 300 iso {ident['identity_iso_mm']:.3f} official {ident['identity_official_mm']:.3f}",
        flush=True,
    )
    if gap > 0.05:
        raise SystemExit(f"landmark mapping disagrees with official mm by {gap:.3f}")

    g = load_generator(device)
    rows = []
    for case in range(1, 11):
        pack = probe if case == 1 else pack_case(case)
        dvf = predict(g, device, pack["mu"])
        row = {"case": case}
        for which in ("75", "300"):
            row[which] = score_set(dvf, case, which, pack)
        rows.append(row)
        r75, r300 = row["75"], row["300"]
        print(
            f"C{case:02d} TRE75 {r75['registered']['mean']:.2f} "
            f"TRE300 {r300['registered']['mean']:.2f} "
            f"id75 {r75['identity_official_mm']:.2f} inside {r75['n_inside']}/{r75['n']}",
            flush=True,
        )

    def cohort(which: str) -> dict:
        vals = np.array([r[which]["registered"]["mean"] for r in rows], dtype=np.float64)
        return {
            "mean_mm": float(vals.mean()),
            "std_mm": float(vals.std(ddof=1)),
            "per_case_mm": vals.tolist(),
        }

    report = {
        "model": "TCIA3 UNetCRBDecoder epoch_100",
        "ckpt": str(CKPT),
        "grid": "A1 R3 CT resampled to 2 mm, 160³ centred on the lung mask, µ min-max, superior axis flipped to match TCIA (apex at high Y)",
        "phases": "input CT_06, ref=5, tgt=0; pred = lm00 + u(lm00) in iso-voxels",
        "brightness_tcia_s1": brightness(tcia_mu, tcia_lung),
        "brightness_dir_c01": brightness(
            probe["mu"], (probe["hu_iso"] > -700) & (probe["hu_iso"] < -200)
        ),
        "tre75": cohort("75"),
        "tre300": cohort("300"),
        "cases": rows,
        "when": datetime.now(timezone.utc).astimezone().isoformat(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"cohort TRE75 {report['tre75']['mean_mm']:.3f}±{report['tre75']['std_mm']:.3f}  "
        f"TRE300 {report['tre300']['mean_mm']:.3f}±{report['tre300']['std_mm']:.3f}",
        flush=True,
    )
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
