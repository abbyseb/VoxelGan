#!/usr/bin/env python3
"""Prepare DIR C1/C5/C8 volumes + Elastix DVFs as G160-style 160³ npy pool.

Writes under experiments/G160_DIR_FT_C1C5C8/data/train/:
  D0k_CT_01..10.npy, D0k_Mask_Lung.npy, D0k_06_to_XX_pair.npy
  manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

DIR_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIR_EXP / "scripts"))
from prepare_a3_dir_case import dir_hu_to_mu_for_synth  # noqa: E402

A1 = DIR_EXP / "arms" / "A1_oracle_dirlab"
EXP = DIR_EXP / "experiments" / "G160_DIR_FT_C1C5C8"
INFER = 160


def resize_zyx(vol: np.ndarray, size: int, mode: str = "trilinear") -> np.ndarray:
    t = torch.from_numpy(vol[None, None].astype(np.float32))
    if mode == "nearest":
        t = F.interpolate(t, size=(size, size, size), mode="nearest")
    else:
        t = F.interpolate(t, size=(size, size, size), mode="trilinear", align_corners=True)
    return t[0, 0].numpy()


def upsample_dvf_zyx3(dvf_zyx3: np.ndarray, size: int = INFER) -> np.ndarray:
    """(D,H,W,3) at 128 → 160 with displacement scaling."""
    d0, h0, w0, _ = dvf_zyx3.shape
    t = torch.from_numpy(np.moveaxis(dvf_zyx3, -1, 0)[None].astype(np.float32))
    out = F.interpolate(t, size=(size, size, size), mode="trilinear", align_corners=True)
    scales = torch.tensor([size / d0, size / h0, size / w0], dtype=out.dtype).view(1, 3, 1, 1, 1)
    out = out * scales
    return np.moveaxis(out[0].numpy(), 0, -1).astype(np.float32)


def load_dvf_mha(path: Path) -> np.ndarray:
    img = sitk.ReadImage(str(path))
    a = sitk.GetArrayFromImage(img)  # z,y,x,3
    if a.ndim != 4 or a.shape[-1] != 3:
        raise ValueError(f"unexpected DVF shape {a.shape} at {path}")
    return a.astype(np.float32)


def patient_id(case: int) -> str:
    return f"D{case:02d}"


def prepare_case(case: int, out_dir: Path, mu_mode: str) -> dict:
    scan = f"DIR_C{case:02d}"
    train = A1 / "runs" / scan / scan / "train"
    if not train.is_dir():
        raise FileNotFoundError(train)
    pid = patient_id(case)
    meta = {"case": case, "patient": pid, "scan": scan, "mu_mode": mu_mode, "phases": {}}

    mask_path = train / "Mask_Lung.mha"
    mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))).astype(np.float32)
    mask160 = (resize_zyx((mask > 0).astype(np.float32), INFER, mode="nearest") > 0.5).astype(
        np.float32
    )
    np.save(out_dir / f"{pid}_Mask_Lung.npy", mask160)

    for ph in range(1, 11):
        ct_path = train / f"CT_{ph:02d}.mha"
        hu = sitk.GetArrayFromImage(sitk.ReadImage(str(ct_path))).astype(np.float32)
        mu, mu_meta = dir_hu_to_mu_for_synth(hu, mu_mode)
        mu160 = resize_zyx(mu, INFER)
        np.save(out_dir / f"{pid}_CT_{ph:02d}.npy", mu160.astype(np.float32))
        meta["phases"][f"{ph:02d}"] = {
            "hu_range": [float(hu.min()), float(hu.max())],
            "mu_range": [float(mu.min()), float(mu.max())],
            "mu_meta": mu_meta,
        }

    n_pairs = 0
    for tgt in range(1, 11):
        if tgt == 6:
            # identity field
            pair = np.zeros((INFER, INFER, INFER, 3), dtype=np.float32)
        else:
            dvf_path = train / f"DVF_sub_{tgt:02d}.mha"
            if not dvf_path.is_file():
                raise FileNotFoundError(dvf_path)
            pair = upsample_dvf_zyx3(load_dvf_mha(dvf_path), INFER)
        out_name = f"{pid}_06_to_{tgt:02d}_pair.npy"
        np.save(out_dir / out_name, pair)
        n_pairs += 1
        meta.setdefault("pairs", []).append(out_name)

    meta["n_pairs"] = n_pairs
    meta["mask_lung_frac"] = float(mask160.mean())
    print(
        f"C{case} → {pid}: CTs=10 pairs={n_pairs} lung_frac={meta['mask_lung_frac']:.3f}",
        flush=True,
    )
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default="1,5,8")
    ap.add_argument("--mu-mode", choices=("default", "hist_match"), default="hist_match")
    ap.add_argument("--out", type=Path, default=EXP / "data" / "train")
    args = ap.parse_args()
    cases = [int(x) for x in args.cases.split(",") if x.strip()]

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    # clean old npy for these patients
    for case in cases:
        pid = patient_id(case)
        for p in out_dir.glob(f"{pid}_*"):
            p.unlink()

    case_metas = []
    all_pairs = []
    for case in cases:
        m = prepare_case(case, out_dir, args.mu_mode)
        case_metas.append(m)
        all_pairs.extend(m["pairs"])

    # simple val: 20% of pairs stratified by patient
    rng = np.random.default_rng(42)
    by_pat: dict[str, list[str]] = {}
    for p in all_pairs:
        by_pat.setdefault(p.split("_")[0], []).append(p)
    train_pairs, val_pairs = [], []
    for pid, ps in by_pat.items():
        ps = list(ps)
        rng.shuffle(ps)
        n_val = max(1, int(round(0.2 * len(ps))))
        val_pairs.extend(ps[:n_val])
        train_pairs.extend(ps[n_val:])

    man = {
        "experiment": "G160_DIR_FT_C1C5C8",
        "mu_mode": args.mu_mode,
        "infer_size": INFER,
        "pooled_train_dir": str(out_dir.resolve()),
        "train_patients": [patient_id(c) for c in cases],
        "holdout_patients": [],
        "train_pairs": sorted(train_pairs),
        "val_pairs": sorted(val_pairs),
        "n_train_pairs": len(train_pairs),
        "n_val_pairs": len(val_pairs),
        "cases": case_metas,
        "note": "Pairs are 06→XX only (Elastix fixed=T50). Init from SPARE G160-A1 Decoder.",
    }
    man_path = EXP / "manifest.json"
    man_path.write_text(json.dumps(man, indent=2) + "\n")
    (out_dir.parent / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    print(f"Wrote {man_path} | train_pairs={len(train_pairs)} val_pairs={len(val_pairs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
