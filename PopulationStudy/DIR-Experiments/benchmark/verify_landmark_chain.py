#!/usr/bin/env python3
"""Tier-0 sanity checks for DIR-Lab landmark / TRE pipeline.

  cd PopulationStudy/DIR-Experiments
  python benchmark/verify_landmark_chain.py --patient P1_DIR

Checks:
  1. native ↔ packed landmark round-trip (mm)
  2. Elastix 01→06 DVF improves TRE vs identity (correct disp sign)
  3. Optional: packed DVF image-warp residual vs target CT
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DIR = Path(__file__).resolve().parents[1]
PACKED = DIR / "Experiment1" / "packed"
sys.path.insert(0, str(DIR / "benchmark"))

from tre_utils import (  # noqa: E402
    PHASE_T00,
    PHASE_T50,
    eval_tre_from_disp,
    find_300_landmarks,
    load_pack_meta,
    native_to_packed_xyz,
    packed_voxel_to_native_xyz,
    sample_dvf_at_xyz,
    tre_mm,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patient", default="P1_DIR")
    ap.add_argument("--packed-root", type=Path, default=PACKED)
    args = ap.parse_args()

    pid = args.patient
    meta = load_pack_meta(args.packed_root, pid)
    ref_n, tgt_n, ref_p = find_300_landmarks(args.packed_root, pid, meta)
    spacing = meta["native_spacing_xyz_mm"]

    print(f"=== {pid} landmark chain ===\n")

    # 1. Round-trip
    back = packed_voxel_to_native_xyz(ref_p, meta)
    rt_mm = np.linalg.norm((back - ref_n) * np.array(spacing), axis=1)
    print(f"1. native→packed→native  mean={rt_mm.mean():.4e} mm  max={rt_mm.max():.4e} mm")
    ref_from_native = native_to_packed_xyz(ref_n, meta)
    pack_err = np.abs(ref_from_native - ref_p).max()
    print(f"   ref_packed vs native_to_packed max diff = {pack_err:.4e} vox")

    # 2. Identity TRE
    tre_id = tre_mm(ref_n, tgt_n, spacing)
    print(f"\n2. Identity TRE (300 pts T00→T50): mean={tre_id['mean']:.2f} mm")

    # 3. Elastix DVF
    pair = args.packed_root / pid / "all" / f"{PHASE_T00:02d}_to_{PHASE_T50:02d}_pair.npy"
    cache = args.packed_root / pid / "benchmark" / f"elastix_{PHASE_T00:02d}_to_{PHASE_T50:02d}.npy"
    dvf_path = pair if pair.is_file() else cache
    if not dvf_path.is_file():
        print(f"\n3. SKIP Elastix — missing {pair.name}")
        return
    dvf_cdhw = np.moveaxis(np.load(dvf_path), -1, 0)
    disp = sample_dvf_at_xyz(dvf_cdhw, ref_p)
    r = eval_tre_from_disp(ref_n, tgt_n, ref_p, disp, meta, spacing, grid="native_mm")
    tre_e = r["tre_mm"]
    delta = tre_id["mean"] - tre_e["mean"]
    ok = tre_e["mean"] < tre_id["mean"]
    print(f"\n3. Elastix TRE ({dvf_path.name}): mean={tre_e['mean']:.2f} mm  "
          f"Δvs id={delta:+.2f} mm  {'PASS' if ok else 'FAIL'}")

    # 4. Image warp residual (packed µ)
    try:
        import torch

        sys.path.insert(0, str(DIR.parent / "InitialExperiments" / "Experiment1"))
        from utilities.warp import warp  # noqa: E402

        data = args.packed_root / pid / "all"
        ct0 = np.load(data / f"CT_{PHASE_T00:02d}.npy")
        ct6 = np.load(data / f"CT_{PHASE_T50:02d}.npy")
        mask = np.load(data / "Mask_Lung.npy") > 0
        flow = torch.from_numpy(dvf_cdhw)[None].float()
        warped = warp(torch.from_numpy(ct0)[None, None].float(), flow)[0, 0].numpy()
        res_id = (np.abs(ct6 - ct0) * mask)[mask].mean()
        res_w = (np.abs(ct6 - warped) * mask)[mask].mean()
        print(f"\n4. Lung |target−warp| µ: identity={res_id:.4f}  elastix={res_w:.4f}  "
              f"{'PASS' if res_w < res_id else 'FAIL'}")
    except Exception as exc:
        print(f"\n4. Image warp check skipped ({exc})")

    out = {
        "patient": pid,
        "roundtrip_mm_mean": float(rt_mm.mean()),
        "identity_tre_mm": tre_id,
        "elastix_tre_mm": tre_e,
        "elastix_beats_identity": bool(ok),
    }
    out_path = DIR / "benchmark" / "results" / f"verify_{pid}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
