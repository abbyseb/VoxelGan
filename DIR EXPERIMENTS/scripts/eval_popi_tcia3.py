#!/usr/bin/env python3
"""Pack POPI onto the TCIA3 training grid and score landmark TRE.

Training frame (packed_r3_hu):
  2 mm isotropic, 160³, centred on the lung, R3 patient axes.
  numpy +X left, +Y superior, +Z anterior.
  Network input is per-volume min-max of µ, the same normalisation as
  FastMmapPhasePairDataset. DVF channels (dx, dy, dz) are iso-voxels along
  (x, y, z).

POPI MHD is RAI (+X right, +Y anterior, +Z inferior). Landmarks are already
in that MHD frame. The headline pair is phase 00 to the published reference
phase (bl = 60, the other five = 50). No Elastix field and no breath-size
input are used.

  CUDA_VISIBLE_DEVICES=0 python scripts/eval_popi_tcia3.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage
from scipy.ndimage import map_coordinates

DIR_EXP = Path(__file__).resolve().parents[1]
VOXEL = DIR_EXP.parent
POPI = Path("/media/abhishek/3CCA3CADCA3C6574/POPI/MedPhys11")
PACKED = Path("/media/abhishek/3CCA3CADCA3C6574/POPI/packed_r3_tcia3")
CKPT = (
    VOXEL
    / "PopulationStudy/ClinicalExperiments/Grid160/TCIA3/DecoderCRB/checkpoints/epoch_100.pt"
)
OUT_JSON = DIR_EXP / "popi_tcia3" / "tre_tcia3_ep100.json"
NET_ROOT = VOXEL / "PopulationStudy/ClinicalExperiments/Grid160/Experiment1"

SPACING = 2.0
SIZE = 160
HALF = 0.5 * SIZE * SPACING
MU_WATER = 0.02
# Paper order. Reference phase is the published MHD phase, stored 0-indexed.
PATIENTS = [
    ("bl", 1, 6),
    ("ng", 2, 5),
    ("dx", 3, 5),
    ("gt", 4, 5),
    ("mm2", 5, 5),
    ("bh", 6, 5),
]


def load_pts(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        if len(parts) >= 3:
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.asarray(rows, dtype=np.float64)


def popi_xyz_to_r3_zyx(xyz: np.ndarray) -> np.ndarray:
    """RAI world mm (N,3) xyz → R3 mm (N,3) zyx = (anterior, superior, left)."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    return np.stack([y, -z, -x], axis=1)


def read_mhd(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = sitk.ReadImage(str(path))
    hu = sitk.GetArrayFromImage(img).astype(np.float32)
    origin = np.array(img.GetOrigin(), dtype=np.float64)
    spacing = np.array(img.GetSpacing(), dtype=np.float64)
    return hu, origin, spacing


def lung_mask(hu: np.ndarray) -> np.ndarray:
    raw = (hu > -950.0) & (hu < -300.0)
    lab, n = ndimage.label(raw)
    if n == 0:
        raise RuntimeError("no lung-like voxels")
    counts = np.bincount(lab.ravel())
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = counts[1:] > 5000
    if not keep.any():
        keep[int(np.argmax(counts[1:])) + 1] = True
    return keep[lab]


def lung_centroid_r3(hu: np.ndarray, origin_xyz: np.ndarray, spacing_xyz: np.ndarray) -> np.ndarray:
    mask = lung_mask(hu)
    iz, iy, ix = np.where(mask)
    wx = origin_xyz[0] + ix * spacing_xyz[0]
    wy = origin_xyz[1] + iy * spacing_xyz[1]
    wz = origin_xyz[2] + iz * spacing_xyz[2]
    r3 = popi_xyz_to_r3_zyx(np.stack([wx, wy, wz], axis=1))
    return r3.mean(axis=0)


def sample_phase(hu: np.ndarray, origin_xyz, spacing_xyz, origin_r3) -> np.ndarray:
    ii = np.arange(SIZE, dtype=np.float64)
    zz, yy, xx = np.meshgrid(
        origin_r3[0] + ii * SPACING,
        origin_r3[1] + ii * SPACING,
        origin_r3[2] + ii * SPACING,
        indexing="ij",
    )
    # inverse of popi_xyz_to_r3_zyx
    popi_y = zz
    popi_z = -yy
    popi_x = -xx
    iz = (popi_z - origin_xyz[2]) / spacing_xyz[2]
    iy = (popi_y - origin_xyz[1]) / spacing_xyz[1]
    ix = (popi_x - origin_xyz[0]) / spacing_xyz[0]
    samp = map_coordinates(
        hu, np.stack([iz.ravel(), iy.ravel(), ix.ravel()], axis=0), order=1, mode="constant", cval=-1000.0
    )
    return samp.reshape(SIZE, SIZE, SIZE).astype(np.float32)


def landmarks_to_index(xyz: np.ndarray, origin_r3: np.ndarray) -> np.ndarray:
    r3 = popi_xyz_to_r3_zyx(xyz)
    return (r3 - origin_r3) / SPACING


def hu_to_mu(hu: np.ndarray) -> np.ndarray:
    return ((hu + 1000.0) * (MU_WATER / 1000.0)).astype(np.float32)


def minmax(x: np.ndarray) -> np.ndarray:
    return ((x - x.min()) / (x.max() - x.min() + 1e-8)).astype(np.float32)


def pack_patient(pid: str, number: int, ref_phase: int) -> dict:
    mhd = POPI / pid / "mhd"
    pts = POPI / pid / "pts"
    hu0, origin, spacing = read_mhd(mhd / "00.mhd")
    centroid = lung_centroid_r3(hu0, origin, spacing)
    origin_r3 = centroid - HALF
    out = PACKED / pid
    out.mkdir(parents=True, exist_ok=True)
    phases = []
    for k in range(10):
        tag = f"{10 * k:02d}"
        src = mhd / f"{tag}.mhd"
        if not src.is_file():
            continue
        hu, org, sp = read_mhd(src)
        vol = sample_phase(hu, org, sp, origin_r3)
        np.save(out / f"CT_{k + 1:02d}.npy", vol)
        np.save(out / f"CT_{k + 1:02d}_mu.npy", hu_to_mu(vol))
        phases.append(k)
        pt = pts / f"{tag}.pts"
        if pt.is_file():
            idx = landmarks_to_index(load_pts(pt), origin_r3)
            np.save(out / f"lm_{k:02d}_zyx.npy", idx.astype(np.float64))
    # orientation QC on the packed phase-0 volume
    vol = np.load(out / "CT_01.npy")
    lung = lung_mask(vol)
    bone = vol > 250
    areas = lung.sum(axis=(0, 2))
    occupied = np.where(areas > 0)[0]
    apex_high = float(areas[occupied[-8:]].mean()) < float(areas[occupied[:8]].mean())
    spine_posterior = float(np.where(bone)[0].mean()) < float(np.where(lung)[0].mean())
    meta = {
        "patient": pid,
        "patient_number": number,
        "reference_phase_0idx": ref_phase,
        "spacing_mm": SPACING,
        "size": SIZE,
        "origin_r3_zyx_mm": origin_r3.tolist(),
        "lung_centroid_r3_zyx_mm": centroid.tolist(),
        "native_origin_xyz_mm": origin.tolist(),
        "native_spacing_xyz_mm": spacing.tolist(),
        "frame": "R3_SPARE_orbit",
        "axes": "numpy z anterior, y superior, x left",
        "phases_0idx": phases,
        "qc_apex_at_high_y": apex_high,
        "qc_spine_posterior": spine_posterior,
    }
    (out / "pack_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"[{pid}] packed phases={phases} apex_high_y={apex_high} spine_posterior={spine_posterior}",
        flush=True,
    )
    return meta


def tre_mm(pred_zyx: np.ndarray, gt_zyx: np.ndarray) -> np.ndarray:
    return np.linalg.norm((pred_zyx - gt_zyx) * SPACING, axis=1)


def _sample_u(dvf_zyx3: np.ndarray, idx_zyx: np.ndarray) -> np.ndarray:
    """Sample (dx, dy, dz) at zyx indices. Returns (N, 3) in zyx order."""
    pts = idx_zyx.T
    dx = map_coordinates(dvf_zyx3[..., 0], pts, order=1, mode="nearest")
    dy = map_coordinates(dvf_zyx3[..., 1], pts, order=1, mode="nearest")
    dz = map_coordinates(dvf_zyx3[..., 2], pts, order=1, mode="nearest")
    return np.stack([dz, dy, dx], axis=1)


def warp_landmarks(dvf_zyx3: np.ndarray, idx_zyx: np.ndarray, n_iter: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Invert the pull field. Training warp is target(x) = reference(x + u(x)).

    A reference landmark p therefore lands at q where q + u(q) = p.
    """
    inside = np.all((idx_zyx >= 0.0) & (idx_zyx <= (SIZE - 1)), axis=1)
    q = idx_zyx.copy()
    for _ in range(n_iter):
        ok = inside & np.all((q >= 0.0) & (q <= (SIZE - 1)), axis=1)
        uq = np.zeros_like(q)
        if ok.any():
            uq[ok] = _sample_u(dvf_zyx3, q[ok])
        nxt = idx_zyx - uq
        q[ok] = nxt[ok]
    reliable = inside & np.all((q >= 0.0) & (q <= (SIZE - 1)), axis=1)
    return q, reliable


def load_generator(device: torch.device):
    import sys

    if str(NET_ROOT) not in sys.path:
        sys.path.insert(0, str(NET_ROOT))
    from networks.generator_crb_dec import UNetCRBDecoder

    g = UNetCRBDecoder(im_size=SIZE, n_phases=10)
    ckpt = torch.load(str(CKPT), map_location=device, weights_only=False)
    state = ckpt["generator"] if isinstance(ckpt, dict) and "generator" in ckpt else ckpt
    g.load_state_dict(state, strict=True)
    g.to(device).eval()
    return g


def predict(g, device, mu: np.ndarray, ref: int, tgt: int) -> np.ndarray:
    x = torch.from_numpy(minmax(mu)[None, None]).to(device)
    rp = torch.tensor([ref], dtype=torch.long, device=device)
    tp = torch.tensor([tgt], dtype=torch.long, device=device)
    with torch.no_grad():
        dvf = g(x, rp, tp)[0].detach().cpu().numpy()
    # (3,Z,Y,X) → (Z,Y,X,3) with channels (dx,dy,dz)
    return np.moveaxis(dvf, 0, -1).astype(np.float32)


def summarise(errs: np.ndarray) -> dict:
    return {
        "n": int(errs.size),
        "mean_mm": float(errs.mean()) if errs.size else None,
        "std_mm": float(errs.std()) if errs.size else None,
        "median_mm": float(np.median(errs)) if errs.size else None,
    }


def score_pair(g, device, pid: str, ref: int, tgt: int, save: bool) -> dict:
    folder = PACKED / pid
    mu = np.load(folder / f"CT_{ref + 1:02d}_mu.npy")
    src = np.load(folder / f"lm_{ref:02d}_zyx.npy")
    dst = np.load(folder / f"lm_{tgt:02d}_zyx.npy")
    n = min(len(src), len(dst))
    src, dst = src[:n], dst[:n]
    dvf = predict(g, device, mu, ref, tgt)
    moved, inside = warp_landmarks(dvf, src)
    ident = tre_mm(src[inside], dst[inside])
    err = tre_mm(moved[inside], dst[inside])
    if save:
        np.save(folder / f"dvf_{ref:02d}_to_{tgt:02d}.npy", dvf)
    return {
        "ref": ref,
        "tgt": tgt,
        "n_landmarks": int(n),
        "n_inside": int(inside.sum()),
        "identity": summarise(ident),
        "tcia3": summarise(err),
        "mean_landmark_d_zyx_mm": ((dst[inside] - src[inside]).mean(axis=0) * SPACING).tolist(),
        "mean_pred_d_zyx_mm": ((moved[inside] - src[inside]).mean(axis=0) * SPACING).tolist(),
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()
    metas = []
    for pid, number, ref in PATIENTS:
        meta_path = PACKED / pid / "pack_meta.json"
        if args.score_only and meta_path.is_file():
            metas.append(json.loads(meta_path.read_text()))
        else:
            metas.append(pack_patient(pid, number, ref))
    if not all(m["qc_spine_posterior"] for m in metas):
        raise SystemExit("orientation QC failed — spine is not posterior in the packed frame")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    g = load_generator(device)
    rows = []
    for meta in metas:
        pid = meta["patient"]
        ref = int(meta["reference_phase_0idx"])
        # published pair: phase 0 → reference, and the reverse
        for a, b, save in ((0, ref, True), (ref, 0, False)):
            row = score_pair(g, device, pid, a, b, save=save)
            row["patient"] = pid
            row["patient_number"] = meta["patient_number"]
            row["pair"] = "00_to_reference" if (a, b) == (0, ref) else "reference_to_00"
            rows.append(row)
            t = row["tcia3"]
            i = row["identity"]
            print(
                f"{pid} {a:02d}->{b:02d}  identity {i['mean_mm']:.2f}±{i['std_mm']:.2f}  "
                f"TCIA3 {t['mean_mm']:.2f}±{t['std_mm']:.2f}  n={t['n']}",
                flush=True,
            )

    headline = [r for r in rows if r["pair"] == "00_to_reference"]
    means = np.array([r["tcia3"]["mean_mm"] for r in headline], dtype=np.float64)
    ident = np.array([r["identity"]["mean_mm"] for r in headline], dtype=np.float64)
    # pooled over landmarks, not over patients
    # recompute from per-patient means is the usual cohort figure (unweighted patients)
    report = {
        "model": "TCIA3 UNetCRBDecoder epoch_100",
        "ckpt": str(CKPT),
        "grid": "2 mm 160³ R3, lung-centroid, µ min-max",
        "headline": "phase 00 → published reference (bl 60, others 50)",
        "dvf_application": "inverse of pull field: target(x)=reference(x+u(x)), verified on TCIA S1 Elastix 01→06",
        "cohort_patient_mean_tre_mm": float(means.mean()),
        "cohort_patient_std_tre_mm": float(means.std()),
        "cohort_identity_mean_mm": float(ident.mean()),
        "patients": rows,
        "pack_meta": metas,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"cohort 00→ref  identity {ident.mean():.2f}  TCIA3 {means.mean():.2f}±{means.std():.2f} mm",
        flush=True,
    )
    print(f"wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
