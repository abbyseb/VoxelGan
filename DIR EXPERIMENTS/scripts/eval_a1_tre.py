#!/usr/bin/env python3
"""A1 TRE bridge: sub-grid DVF → official DIR-Lab landmarks.

Reports **both** 75-pt Sampled4D and 300-pt extreme sets (T00→T50 primary;
also T50→T00 for Elastix-direction QA). Arm tables should list both.

Coordinate chain (Case packs with si_axis_flipped in P*_DIR):
  official (x,y,z) ↔ pack: z_p = (nz-1) - z
  pack continuous index → sub_128: s = n * 128 / N
  DVF on 128³ is in *sub-voxels* (LEARN downsample sets spacing to 1).

R3 train volumes (--r3): after pack, map to SPARE orbit indices
  (x,y,z)_r3 = (x, (nz-1)-z, (ny-1)-y), shape (nx,nz,ny), then → sub_128.
  TRE still reported in official DIR-Lab mm (landmarks mapped back).

Convention (Elastix fixed=T50=phase06, moving=T00=phase01; LEARN unit-voxel DVF):
  ITK: moving ≈ fixed + disp(fixed)
  T00→T50: pred = lm00_pack + (-disp) sampled at lm00
  T50→T00: pred = lm50_pack + (+disp) sampled at lm50
  (HU-preserving sub_CT. Old HU-clipped fields were tiny; a flipped sign
   spuriously looked better — do not use that convention anymore.)

  cd "DIR EXPERIMENTS"
  LEARN-GUI/.venv/bin/python scripts/eval_a1_tre.py --case 1 --gpu 1
  LEARN-GUI/.venv/bin/python scripts/eval_a1_tre.py --case 1 --r3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

DIR_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIR_EXP / "scripts"))
from dirlab_tre import (  # noqa: E402
    CASE_INFO,
    landmarks_75,
    landmarks_300,
    sample_dvf,
    stats,
    tre_mm,
)

VMC = Path(os.environ.get("VOXELMAP_CLINICAL_ROOT", "/home/abhishek/Documents/VoxelMap_Clinical"))


def official_to_pack(xyz: np.ndarray, nz: int) -> np.ndarray:
    out = np.asarray(xyz, dtype=np.float64).copy()
    out[:, 2] = (nz - 1) - out[:, 2]
    return out


def pack_to_official(xyz: np.ndarray, nz: int) -> np.ndarray:
    return official_to_pack(xyz, nz)  # involution


def pack_to_r3(xyz: np.ndarray, ny: int, nz: int) -> np.ndarray:
    """Native pack (L,P,S) → R3 itk index (L, S_flip, P_flip)."""
    out = np.empty_like(xyz, dtype=np.float64)
    out[:, 0] = xyz[:, 0]
    out[:, 1] = (nz - 1) - xyz[:, 2]
    out[:, 2] = (ny - 1) - xyz[:, 1]
    return out


def r3_to_pack(xyz: np.ndarray, ny: int, nz: int) -> np.ndarray:
    out = np.empty_like(xyz, dtype=np.float64)
    out[:, 0] = xyz[:, 0]
    out[:, 1] = (ny - 1) - xyz[:, 2]
    out[:, 2] = (nz - 1) - xyz[:, 1]
    return out


def r3_shape_spacing(case: int):
    (nx, ny, nz), (dx, dy, dz) = CASE_INFO[case]
    return (nx, nz, ny), (dx, dz, dy)


def pack_to_sub(xyz: np.ndarray, shape_xyz) -> np.ndarray:
    nx, ny, nz = shape_xyz
    out = np.empty_like(xyz, dtype=np.float64)
    out[:, 0] = xyz[:, 0] * 128.0 / nx
    out[:, 1] = xyz[:, 1] * 128.0 / ny
    out[:, 2] = xyz[:, 2] * 128.0 / nz
    return out


def sub_disp_to_pack(dsub: np.ndarray, shape_xyz) -> np.ndarray:
    nx, ny, nz = shape_xyz
    out = np.empty_like(dsub, dtype=np.float64)
    out[:, 0] = dsub[:, 0] * nx / 128.0
    out[:, 1] = dsub[:, 1] * ny / 128.0
    out[:, 2] = dsub[:, 2] * nz / 128.0
    return out


def npy_hwd_to_zyx(a: np.ndarray) -> np.ndarray:
    """prep_train stores DVF as (H,W,D,3)=(y,x,z,3); sample_dvf wants (z,y,x,3)."""
    return np.transpose(a, (2, 0, 1, 3))


def load_dvf_zyx3(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        a = np.load(path).astype(np.float64)
        if a.ndim != 4 or a.shape[-1] != 3:
            raise ValueError(f"Expected (H,W,D,3) DVF npy, got {a.shape} from {path}")
        a = npy_hwd_to_zyx(a)
    else:
        import SimpleITK as sitk

        a = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float64)
    if a.ndim != 4 or a.shape[-1] != 3:
        raise ValueError(f"Expected (z,y,x,3) DVF, got {a.shape} from {path}")
    if a.shape != (128, 128, 128, 3):
        raise ValueError(f"Expected 128³ DVF, got {a.shape} from {path}")
    if not np.isfinite(a).all():
        raise ValueError(f"Non-finite values in DVF {path}")
    return a


def _load_lms(case: int, which: str):
    if which == "75":
        return landmarks_75(case, "T00"), landmarks_75(case, "T50")
    if which == "300":
        return landmarks_300(case, "T00"), landmarks_300(case, "T50")
    raise ValueError(f"which must be '75' or '300', got {which!r}")


def tre_t00_t50(dvf_zyx3: np.ndarray, case: int, which: str = "75", *, r3: bool = False) -> dict:
    (nx, ny, nz), spacing = CASE_INFO[case]
    lm00, lm50 = _load_lms(case, which)
    lm00_p = official_to_pack(lm00, nz)
    if r3:
        shape, _ = r3_shape_spacing(case)
        lm00_g = pack_to_r3(lm00_p, ny, nz)
        disp = sample_dvf(dvf_zyx3, pack_to_sub(lm00_g, shape))
        pred_g = lm00_g - sub_disp_to_pack(disp, shape)
        pred_p = r3_to_pack(pred_g, ny, nz)
    else:
        shape = (nx, ny, nz)
        disp = sample_dvf(dvf_zyx3, pack_to_sub(lm00_p, shape))
        pred_p = lm00_p - sub_disp_to_pack(disp, shape)
    pred = pack_to_official(pred_p, nz)
    got = stats(tre_mm(pred, lm50, spacing))
    ident = stats(tre_mm(lm00, lm50, spacing))
    return {
        "registered": got,
        "identity": ident,
        "improvement_mm": ident["mean"] - got["mean"],
        "set": which,
        "frame": "r3" if r3 else "native",
    }


def tre_t50_t00(dvf_zyx3: np.ndarray, case: int, which: str = "75", *, r3: bool = False) -> dict:
    (nx, ny, nz), spacing = CASE_INFO[case]
    lm00, lm50 = _load_lms(case, which)
    lm50_p = official_to_pack(lm50, nz)
    if r3:
        shape, _ = r3_shape_spacing(case)
        lm50_g = pack_to_r3(lm50_p, ny, nz)
        disp = sample_dvf(dvf_zyx3, pack_to_sub(lm50_g, shape))
        pred_g = lm50_g + sub_disp_to_pack(disp, shape)
        pred_p = r3_to_pack(pred_g, ny, nz)
    else:
        shape = (nx, ny, nz)
        disp = sample_dvf(dvf_zyx3, pack_to_sub(lm50_p, shape))
        pred_p = lm50_p + sub_disp_to_pack(disp, shape)
    pred = pack_to_official(pred_p, nz)
    got = stats(tre_mm(pred, lm00, spacing))
    ident = stats(tre_mm(lm50, lm00, spacing))
    return {
        "registered": got,
        "identity": ident,
        "improvement_mm": ident["mean"] - got["mean"],
        "set": which,
        "frame": "r3" if r3 else "native",
    }


def infer_voxelmap_dvf_phase(
    ckpt: Path,
    data_dir: Path,
    device: str,
    stride: int = 10,
    *,
    phase: int = 1,
) -> tuple[np.ndarray, int]:
    """Infer one target phase (01–10), conditioned on reference phase 06."""
    if phase not in range(1, 11) or phase == 6:
        raise ValueError("Target phase must be 01–10 excluding reference phase 06")
    sys.path.insert(0, str(VMC))
    import torch
    from ml.utilities import networksFiLM

    def _normalize(x: np.ndarray) -> np.ndarray:
        lo, hi = x.min(), x.max()
        if hi - lo < 1e-8:
            return np.zeros_like(x, dtype=np.float32)
        return ((x - lo) / (hi - lo)).astype(np.float32)

    model = networksFiLM.Model.load(str(ckpt), device)
    model = model.to(device).eval()

    src_vol_path = data_dir / "SourceVolumes" / "sub_CT_06_mha.npy"
    src_vol = _normalize(np.load(src_vol_path).squeeze())
    if src_vol.shape != (128, 128, 128):
        raise ValueError(f"Expected 128³ source volume, got {src_vol.shape} from {src_vol_path}")
    src_vol_t = torch.from_numpy(src_vol[None, None]).to(device)

    angles = None
    ap = data_dir / "Angles.csv"
    if ap.is_file():
        import pandas as pd

        angles = pd.read_csv(ap, header=None).values.squeeze()
        angles = np.atleast_1d(np.asarray(angles, dtype=np.float64)).ravel()

    tgt_files = sorted((data_dir / "TargetProjections").glob(f"{phase:02d}_Proj_*_bin.npy"))
    tgt_files = tgt_files[:: max(1, stride)]
    flows = []
    with torch.no_grad():
        for tgt_file in tgt_files:
            proj_num = int(tgt_file.name.split("_")[2])
            src_file = data_dir / "SourceProjections" / f"06_Proj_{proj_num:03d}_bin.npy"
            if not src_file.is_file():
                continue
            angle_val = float(angles[proj_num - 1]) if angles is not None else 0.0
            src_p = torch.from_numpy(_normalize(np.load(src_file))[None, None]).to(device)
            tgt_p = torch.from_numpy(_normalize(np.load(tgt_file))[None, None]).to(device)
            _, pred_flow = model(src_p, tgt_p, src_vol_t)  # (1,3,D,H,W)
            flows.append(pred_flow[0].detach().cpu().numpy())

    if not flows:
        raise ValueError(f"No phase-{phase:02d} projection pairs found for inference")
    mean_flow = np.mean(np.stack(flows, axis=0), axis=0)  # (3,H,W,D) = GT npy layout
    hwd = np.moveaxis(mean_flow, 0, -1).astype(np.float64)  # (H,W,D,3)
    return npy_hwd_to_zyx(hwd), len(flows)


def infer_voxelmap_dvf_phase01(ckpt, data_dir, device, stride=10):
    """Compatibility entry point for the established T00/T50 evaluator."""
    return infer_voxelmap_dvf_phase(ckpt, data_dir, device, stride, phase=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--stride", type=int, default=10, help="use every Nth phase-01 proj")
    ap.add_argument(
        "--r3",
        action="store_true",
        help="DVF lives on R3-reoriented CT grid (SPARE orbit frame)",
    )
    ap.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="default arms/A1_oracle_dirlab/runs/DIR_C{case:02d}",
    )
    args = ap.parse_args()
    if int(args.gpu) != 1:
        raise SystemExit(f"Refusing --gpu {args.gpu}: use --gpu 1 only")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ.setdefault(
        "DIRLAB_ROOT", str(DIR_EXP / "data" / "dirlab_packs")
    )

    scan_id = f"DIR_C{args.case:02d}"
    run_root = args.run_root or (DIR_EXP / "arms" / "A1_oracle_dirlab" / "runs" / scan_id)
    train = run_root / scan_id / "train"
    mt = run_root / "ModelTraining" / "train" / scan_id
    ckpt = run_root / "checkpoints_nofilm" / "best.pt"
    out_dir = run_root / "tre"
    out_dir.mkdir(parents=True, exist_ok=True)
    # legacy alias
    (run_root / "tre_75").mkdir(parents=True, exist_ok=True)

    frame = "r3" if args.r3 else "native"
    print(f"Case {args.case} | TRE bridge (75 + 300) | frame={frame} | run={run_root}")

    results = {
        "case": args.case,
        "sets": ["75", "300"],
        "scan_id": scan_id,
        "frame": frame,
        "arms": {},
    }

    def _store_arm(label: str, path: Path | None, dvf: np.ndarray, extra: dict | None = None):
        arm = results["arms"].setdefault(label, {})
        if path is not None:
            arm["path"] = str(path)
        if extra:
            arm.update(extra)
        for which in ("75", "300"):
            r00 = tre_t00_t50(dvf, args.case, which, r3=args.r3)
            r50 = tre_t50_t00(dvf, args.case, which, r3=args.r3)
            arm[which] = {"T00_T50": r00, "T50_T00": r50}
            # keep top-level T00_T50 = 75 for older readers
            if which == "75":
                arm["T00_T50"] = r00
                arm["T50_T00"] = r50
            print(
                f"{label:12s} [{which:>3s}] T00→T50 TRE={r00['registered']['mean']:.3f} mm "
                f"(identity {r00['identity']['mean']:.3f}, Δ {r00['improvement_mm']:+.3f})"
            )
            print(
                f"{'':12s}       T50→T00 TRE={r50['registered']['mean']:.3f} mm "
                f"(Δ {r50['improvement_mm']:+.3f})"
            )

    # --- GT Elastix ---
    for label, path in [
        ("elastix_mha", train / "DVF_sub_01.mha"),
        ("elastix_npy", mt / "DVFs" / "DVF_01_mha.npy"),
    ]:
        if not path.is_file():
            print(f"SKIP {label}: missing {path}")
            continue
        _store_arm(label, path, load_dvf_zyx3(path))

    # --- VoxelMap ---
    if not ckpt.is_file():
        print(f"SKIP voxelmap: missing {ckpt}")
    else:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dvf_vm, n = infer_voxelmap_dvf_phase01(ckpt, mt, device, stride=args.stride)
        np.save(out_dir / "voxelmap_dvf_phase01_mean.npy", dvf_vm.astype(np.float32))
        np.save(
            run_root / "tre_75" / "voxelmap_dvf_phase01_mean.npy",
            dvf_vm.astype(np.float32),
        )
        extra = {"n_proj_averaged": n, "checkpoint": str(ckpt)}
        gt_path = mt / "DVFs" / "DVF_01_mha.npy"
        if gt_path.is_file():
            gt = load_dvf_zyx3(gt_path)
            extra["l1_vs_elastix_zyx"] = float(np.mean(np.abs(dvf_vm - gt)))
            print(f"{'voxelmap':12s}       L1 vs elastix (zyx sub-vox): {extra['l1_vs_elastix_zyx']:.4f}")
        _store_arm("voxelmap", None, dvf_vm, extra=extra)
        print(f"{'voxelmap':12s}       averaged over {n} projs (stride={args.stride})")

    out_json = out_dir / "tre_summary.json"
    legacy_json = run_root / "tre_75" / "tre_75_summary.json"

    # Pass criteria on 75-pt: at least one arm beats identity on T00→T50
    ok_bridge = False
    for name, arm in results["arms"].items():
        block = arm.get("75") or {}
        t00 = block.get("T00_T50") or arm.get("T00_T50") or {}
        imp = float(t00.get("improvement_mm", 0.0))
        if imp > 0.05:
            ok_bridge = True
            print(f"VERIFY OK: {name} beats identity by {imp:.3f} mm on 75-pt T00→T50")
    if not ok_bridge:
        print(
            "VERIFY WEAK: no arm beat identity by >0.05 mm on 75-pt — check HU/mask labels."
        )

    results["bridge_verified"] = ok_bridge
    payload = json.dumps(results, indent=2) + "\n"
    out_json.write_text(payload)
    legacy_json.write_text(payload)
    print(f"Wrote {out_json}")
    print(f"Wrote {legacy_json} (alias)")
    return 0 if ok_bridge else 2


if __name__ == "__main__":
    raise SystemExit(main())
