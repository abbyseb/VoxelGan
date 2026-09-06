#!/usr/bin/env python3
"""E4: rescore E3 Both zero-shot QC in millimetres + optional scale correction.

Packing: lung-bbox (native 1 mm) → 128³, Elastix/DVF in packed-voxel units.
mm/voxel_zyx = bbox_size_zyx / 128.

Modes
  voxel     — current QC (baseline)
  mm        — both pred & GT × per-axis spacing (physical score)
  corrected — pred *= (sp_SPARE / sp_patient) per axis, then score in patient voxels
              (maps network's learned scale into this patient's voxel units)

  cd PopulationStudy/ClinicalExperiments
  PYTHONPATH=Experiment3 python scripts/e4_mm_rescoring.py --gpu 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

CE = Path(__file__).resolve().parents[1]
E1 = CE / "Experiment1"
E3 = CE / "Experiment3"
IE1 = CE.parent / "InitialExperiments" / "Experiment1"
sys.path.insert(0, str(IE1))
sys.path.insert(0, str(E3))

from losses.losses import DVFLoss  # noqa: E402
from networks.generator_crb_both import UNetCRBBoth  # noqa: E402

# Import packing helpers
sys.path.insert(0, str(CE / "scripts"))
from prepare_clinical_dvf_library import (  # noqa: E402
    _bbox_from_mask,
    eval_dir,
    morph_pad,
    parse_scan_id,
)

# SPARE MC mean packed mm/vox (zyx), from Training/Validation masks
SPARE_SP_ZYX = np.array([1.5243, 1.7179, 2.1519], dtype=np.float64)

CKPT = E3 / "BothCRB/weights/crb_both_mse_cyclic_fov_aug_generator.pth"
OUT = CE / "plots" / "e4_mm_rescoring"


def pack_spacing_zyx(scan_id: str, mask_pad: int = 8, bbox_pad: int = 8) -> np.ndarray:
    import itk

    patient, _, vendor = parse_scan_id(scan_id)
    ev = eval_dir(patient, scan_id, vendor)
    mask = morph_pad(itk.array_from_image(itk.imread(str(ev / "Mask_Lung.mha"), itk.UC)) > 0, mask_pad)
    z0, z1, y0, y1, x0, x1 = _bbox_from_mask(mask, pad=bbox_pad)
    # native spacing is 1 mm → crop size in mm; DVF comps are ITK x,y,z
    return np.array([z1 - z0, y1 - y0, x1 - x0], dtype=np.float64) / 128.0


def to_cdhw(dvf: np.ndarray) -> np.ndarray:
    if dvf.ndim == 4 and dvf.shape[-1] == 3:
        return np.moveaxis(dvf, -1, 0)
    return dvf


def scale_dvf_cdhw(dvf: np.ndarray, sp_zyx: np.ndarray) -> np.ndarray:
    """dvf channels = (x,y,z) ↔ spacing (sp_x, sp_y, sp_z) = (sp_zyx[2], sp_zyx[1], sp_zyx[0])."""
    out = dvf.astype(np.float32).copy()
    sp_xyz = np.array([sp_zyx[2], sp_zyx[1], sp_zyx[0]], dtype=np.float32)
    out[0] *= sp_xyz[0]
    out[1] *= sp_xyz[1]
    out[2] *= sp_xyz[2]
    return out


def correct_pred_to_patient_vox(pred: np.ndarray, sp_zyx: np.ndarray, ref_zyx: np.ndarray) -> np.ndarray:
    """pred_patient_vox = pred * (sp_ref / sp_patient) per axis (same physical → more voxels if finer)."""
    out = pred.astype(np.float32).copy()
    ref_xyz = np.array([ref_zyx[2], ref_zyx[1], ref_zyx[0]], dtype=np.float32)
    pat_xyz = np.array([sp_zyx[2], sp_zyx[1], sp_zyx[0]], dtype=np.float32)
    scale = ref_xyz / np.maximum(pat_xyz, 1e-6)
    out[0] *= scale[0]
    out[1] *= scale[1]
    out[2] *= scale[2]
    return out


def metrics(pred, gt, mask, dvf_l1):
    m = mask > 0.5
    gt_t = torch.from_numpy(gt)[None].float()
    pr_t = torch.from_numpy(pred)[None].float()
    mk_t = torch.from_numpy(mask)[None, None].float()
    l1 = float(dvf_l1.loss(gt_t, pr_t, mk_t).item())
    l1z = float(dvf_l1.loss(gt_t, torch.zeros_like(gt_t), mk_t).item())
    if m.any() and float(np.abs(gt).max()) > 0:
        a, b = pred[:, m], gt[:, m]
        cos = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-8))
        k = float(np.linalg.norm(pred[:, m]) / (np.linalg.norm(gt[:, m]) + 1e-8))
    else:
        cos, k = float("nan"), float("nan")
    return dict(l1=l1, l1_zero=l1z, ratio=l1 / (l1z + 1e-8), cos=cos, k=k)


def summarize(rows):
    directed = [r for r in rows if r["ref"] != r["tgt"]]
    if not directed:
        return {}
    return {
        "n_directed": len(directed),
        "mean_L1": float(np.mean([r["l1"] for r in directed])),
        "mean_L1_over_zero": float(np.mean([r["ratio"] for r in directed])),
        "mean_cos": float(np.nanmean([r["cos"] for r in directed])),
        "mean_k": float(np.nanmean([r["k"] for r in directed])),
        "beat_zero_pct": float(100.0 * sum(r["ratio"] < 1.0 for r in directed) / len(directed)),
    }


def run_scan(scan: str, g, device, dvf_l1, sp_zyx: np.ndarray):
    data_dir = E1 / "data" / scan / "all"
    mask = (np.load(data_dir / "Mask_Lung.npy") > 0).astype(np.float32)
    stems = [f"{i:02d}_to_{j:02d}" for i in range(1, 11) for j in range(1, 11)]

    modes = {"voxel": [], "mm": [], "corrected": []}
    with torch.no_grad():
        for stem in stems:
            ref, tgt = int(stem[:2]), int(stem[-2:])
            ct_r = np.load(data_dir / f"CT_{ref:02d}.npy").astype(np.float32)
            # match QC norm
            lo, hi = float(ct_r.min()), float(ct_r.max())
            ct_r = np.zeros_like(ct_r) if hi <= lo else (ct_r - lo) / (hi - lo)
            if ref == tgt:
                gt = np.zeros((3,) + ct_r.shape, np.float32)
            else:
                gt = to_cdhw(np.load(data_dir / f"{stem}_pair.npy").astype(np.float32))

            ref_t = torch.from_numpy(ct_r)[None, None].to(device)
            pred = g(
                ref_t,
                torch.tensor([ref - 1], device=device),
                torch.tensor([tgt - 1], device=device),
            )[0].cpu().numpy()

            for mode, pr, gg in [
                ("voxel", pred, gt),
                ("mm", scale_dvf_cdhw(pred, sp_zyx), scale_dvf_cdhw(gt, sp_zyx)),
                (
                    "corrected",
                    correct_pred_to_patient_vox(pred, sp_zyx, SPARE_SP_ZYX),
                    gt,
                ),
            ]:
                m = metrics(pr, gg, mask, dvf_l1)
                modes[mode].append(dict(pair=stem, ref=ref, tgt=tgt, **m))

    return {mode: summarize(rows) for mode, rows in modes.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument(
        "--scans",
        default=",".join(
            [f"CV_P{i}_V_01" for i in range(1, 6)] + [f"CE_P{i}_V_01" for i in range(1, 6)]
        ),
    )
    args = ap.parse_args()
    scans = [s.strip() for s in args.scans.split(",") if s.strip()]

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    g = UNetCRBBoth(im_size=128, n_phases=10).to(device)
    assert g.cond_dim == 4
    g.load_state_dict(torch.load(CKPT, map_location=device))
    g.eval()
    dvf_l1 = DVFLoss()

    OUT.mkdir(parents=True, exist_ok=True)
    results = {"spare_ref_sp_zyx": SPARE_SP_ZYX.tolist(), "scans": {}}

    print(f"E4 mm rescoring  ckpt={CKPT.name}  device={device}", flush=True)
    print(f"SPARE ref mm/vox zyx={SPARE_SP_ZYX}", flush=True)

    for scan in scans:
        sp = pack_spacing_zyx(scan)
        print(f"\n=== {scan}  pack_mm/vox_zyx={sp} ===", flush=True)
        summ = run_scan(scan, g, device, dvf_l1, sp)
        results["scans"][scan] = {
            "pack_mm_per_vox_zyx": sp.tolist(),
            "modes": summ,
        }
        for mode in ("voxel", "mm", "corrected"):
            s = summ[mode]
            print(
                f"  {mode:10s}  L1/z={s['mean_L1_over_zero']:.3f}  cos={s['mean_cos']:.3f}  "
                f"k={s['mean_k']:.3f}  beat={s['beat_zero_pct']:.0f}%",
                flush=True,
            )

    # cohort means
    print("\n======== COHORT MEANS ========", flush=True)
    for vendor, prefix in (("varian", "CV_"), ("elekta", "CE_")):
        subset = [s for s in scans if s.startswith(prefix)]
        print(f"\n-- {vendor} (n={len(subset)}) --")
        for mode in ("voxel", "mm", "corrected"):
            cos = np.mean([results["scans"][s]["modes"][mode]["mean_cos"] for s in subset])
            ratio = np.mean(
                [results["scans"][s]["modes"][mode]["mean_L1_over_zero"] for s in subset]
            )
            beat = np.mean(
                [results["scans"][s]["modes"][mode]["beat_zero_pct"] for s in subset]
            )
            k = np.mean([results["scans"][s]["modes"][mode]["mean_k"] for s in subset])
            print(
                f"  {mode:10s}  L1/z={ratio:.3f}  cos={cos:.3f}  k={k:.3f}  beat={beat:.0f}%",
                flush=True,
            )
        # Elekta valid subset P3-P5
        if vendor == "elekta":
            sub2 = [s for s in subset if any(f"P{i}" in s for i in (3, 4, 5))]
            print(f"  (CE_P3–P5 only)")
            for mode in ("voxel", "mm", "corrected"):
                cos = np.mean([results["scans"][s]["modes"][mode]["mean_cos"] for s in sub2])
                beat = np.mean(
                    [results["scans"][s]["modes"][mode]["beat_zero_pct"] for s in sub2]
                )
                print(f"  {mode:10s}  cos={cos:.3f}  beat={beat:.0f}%", flush=True)

    out_json = OUT / "e4_summary.json"
    out_json.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
