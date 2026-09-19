#!/usr/bin/env python3
"""Compare G160 Encoder vs Decoder synth on DIR-Lab (R3 A1 CT_06).

Metrics per case (phase 06→01 unless noted):
  - TRE75 T00→T50 (--r3), DVF = −u (Elastix convention)
  - cos / L1 vs A1 Elastix sub DVF
  - CT Pearson NCC: warped CT_06 vs real CT_01

  cd "DIR EXPERIMENTS"
  LEARN-GUI/.venv/bin/python scripts/compare_g160_enc_dec_on_dir.py --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

DIR_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIR_EXP / "scripts"))

from eval_a1_tre import load_dvf_zyx3, tre_t00_t50  # noqa: E402
from prepare_a3_dir_case import (  # noqa: E402
    dir_hu_to_mu_for_synth,
    norm_mu,
    resize_volume_zyx,
    upsample_dvf_voxel,
    write_vector_dvf_mha,
)

G160_E1 = DIR_EXP.parent / "PopulationStudy" / "ClinicalExperiments" / "Grid160" / "Experiment1"
NET = DIR_EXP.parent / "PopulationStudy" / "InitialExperiments" / "Experiment6"
A1 = DIR_EXP / "arms" / "A1_oracle_dirlab" / "runs"
OUT = DIR_EXP / "arms" / "A3_synth_conditioned" / "results"

MODELS = {
    "decoder": {
        "label": "G160 Decoder (A3 default)",
        "ckpt": G160_E1 / "DecoderCRB/weights/crb_dec_mse_iso_g160_fov_full_generator.pth",
        "class": "decoder",
    },
    "encoder": {
        "label": "G160 Encoder (CRB in encoder)",
        "ckpt": G160_E1 / "EncoderCRB/weights/crb_enc_mse_iso_g160_fov_full_generator.pth",
        "class": "encoder",
    },
}
INFER = 160
IM_SIZE = 64


def load_gen(kind: str, device: torch.device):
    if str(NET) not in sys.path:
        sys.path.insert(0, str(NET))
    from utilities.warp import warp  # noqa: F401

    if kind == "decoder":
        from networks.generator_crb_dec import UNetCRBDecoder

        g = UNetCRBDecoder(im_size=IM_SIZE, n_phases=10)
    else:
        from networks.generator_crb import UNetCRB

        g = UNetCRB(im_size=IM_SIZE, n_phases=10)
    ckpt = torch.load(MODELS[kind]["ckpt"], map_location=device, weights_only=False)
    state = ckpt.get("generator", ckpt.get("state_dict", ckpt))
    g.load_state_dict(state, strict=True)
    g.to(device).eval()
    return g


def pearson_ncc(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    p = float(np.corrcoef(a, b)[0, 1])
    ac = a - a.mean()
    bc = b - b.mean()
    ncc = float((ac * bc).sum() / (np.linalg.norm(ac) * np.linalg.norm(bc) + 1e-12))
    return ncc, p


def cos_mot(a_cxyz: np.ndarray, b_cxyz: np.ndarray, pct: float = 50.0) -> float:
    am = np.sqrt((a_cxyz**2).sum(0))
    bm = np.sqrt((b_cxyz**2).sum(0))
    m = am > np.percentile(am, pct)
    m &= bm > 1e-6
    return float(((a_cxyz * b_cxyz).sum(0)[m] / (am[m] * bm[m] + 1e-8)).mean())


def dvf_to_sub128(dvf_zyx3: np.ndarray, sub_ref: sitk.Image) -> np.ndarray:
    """(3, Z, Y, X) after prepare-style reorder+negate → (128,128,128,3) for TRE."""
    sd, sh, sw = sitk.GetArrayFromImage(sub_ref).shape
    src = dvf_zyx3.shape[1]
    dvf_zyx = dvf_zyx3
    if (sd, sh, sw) != dvf_zyx.shape[1:]:
        t = torch.from_numpy(dvf_zyx[None].astype(np.float32))
        t = F.interpolate(t, size=(sd, sh, sw), mode="trilinear", align_corners=True)
        scales = torch.tensor([sd / src, sh / src, sw / src], dtype=t.dtype).view(1, 3, 1, 1, 1)
        dvf_zyx = (t * scales)[0].numpy()
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "d.mha"
        write_vector_dvf_mha(dvf_zyx, sub_ref, p)
        return load_dvf_zyx3(p)


def pull_to_elastix_zyx(dvf_nat_cxyz: np.ndarray) -> np.ndarray:
    """Same as prepare_a3 write_synth_dvfs (elastix convention)."""
    return -dvf_nat_cxyz[[2, 1, 0], ...]


def synth_phase01(
    g,
    ct06_hu: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (dvf_infer pull u @160³, dvf_native for warp, warped_hu native zyx)."""
    from utilities.warp import warp

    native_shape = ct06_hu.shape
    mu, _ = dir_hu_to_mu_for_synth(ct06_hu, "default")
    x = torch.from_numpy(norm_mu(resize_volume_zyx(mu, INFER))[None, None]).to(device)
    ref_ph = torch.tensor([5], dtype=torch.long, device=device)
    tgt_ph = torch.tensor([0], dtype=torch.long, device=device)
    with torch.no_grad():
        dvf_inf = g(x, ref_ph, tgt_ph)
    dvf_np = dvf_inf[0].cpu().numpy()  # (3,160,160,160) — same grid as prepare labels
    dvf_nat = upsample_dvf_voxel(dvf_np, native_shape, INFER)
    ct_t = torch.from_numpy(ct06_hu[None, None].astype(np.float32))
    flow_t = torch.from_numpy(dvf_nat[None])
    with torch.no_grad():
        warped = warp(ct_t, flow_t)[0, 0].numpy()
    return dvf_np, dvf_nat, warped.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=int, nargs="*", default=list(range(1, 11)))
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("DIRLAB_ROOT", str(DIR_EXP / "data" / "dirlab_packs"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    gens = {k: load_gen(k, device) for k in MODELS}
    rows = []
    cohort = {k: [] for k in MODELS}

    for case in args.cases:
        sid = f"DIR_C{case:02d}"
        train = A1 / sid / sid / "train"
        ct06_p = train / "CT_06.mha"
        ct01_p = train / "CT_01.mha"
        sub06_p = train / "sub_CT_06.mha"
        el_p = A1 / sid / "ModelTraining" / "train" / sid / "DVFs" / "DVF_01_mha.npy"
        if not all(p.is_file() for p in (ct06_p, ct01_p, sub06_p, el_p)):
            print(f"skip {sid}: missing A1 train files")
            continue
        ct06 = sitk.GetArrayFromImage(sitk.ReadImage(str(ct06_p))).astype(np.float32)
        ct01 = sitk.GetArrayFromImage(sitk.ReadImage(str(ct01_p))).astype(np.float32)
        sub_ref = sitk.ReadImage(str(sub06_p))
        el_sub = load_dvf_zyx3(el_p)
        el_c = np.moveaxis(el_sub, -1, 0)  # 3,z,y,x

        row = {"case": case, "models": {}}
        for kind, g in gens.items():
            u_infer, u_pull, warped = synth_phase01(g, ct06, device)
            dvf_el = pull_to_elastix_zyx(u_infer)
            dvf_sub = dvf_to_sub128(dvf_el, sub_ref)
            tre = tre_t00_t50(dvf_sub, case, "75", r3=True)
            pred_c = np.moveaxis(dvf_sub, -1, 0)
            sl = tuple(slice(0, min(a, b)) for a, b in zip(warped.shape, ct01.shape))
            ncc, pearson = pearson_ncc(warped[sl], ct01[sl])
            m = {
                "tre75_mm": tre["registered"]["mean"],
                "identity_mm": tre["identity"]["mean"],
                "delta_id_mm": tre["improvement_mm"],
                "cos_vs_elastix": cos_mot(pred_c, el_c),
                "l1_vs_elastix_sub": float(np.mean(np.abs(pred_c - el_c))),
                "mean_u_pull": float(np.sqrt((u_pull**2).sum(0)).mean()),
                "mean_u_elastix": float(np.sqrt((dvf_el**2).sum(0)).mean()),
                "mean_elastix_mag": float(np.sqrt((el_c**2).sum(0)).mean()),
                "ct_ncc_vs_real01": ncc,
                "ct_pearson_vs_real01": pearson,
            }
            row["models"][kind] = m
            cohort[kind].append(m["tre75_mm"])
            print(
                f"{sid} {kind:7s} TRE={m['tre75_mm']:.2f} Δid={m['delta_id_mm']:+.2f} "
                f"cos={m['cos_vs_elastix']:.3f} CT_NCC={m['ct_ncc_vs_real01']:.4f}"
            )
        rows.append(row)

    summary = {}
    for kind in MODELS:
        t = cohort[kind]
        summary[kind] = {
            "label": MODELS[kind]["label"],
            "mean_tre75_mm": float(np.mean(t)) if t else None,
            "cases": len(t),
        }
    out = {
        "frame": "r3",
        "phase": "06_to_01",
        "dvf_convention": "elastix (-pull u)",
        "infer_size": INFER,
        "summary": summary,
        "per_case": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "g160_encoder_vs_decoder_dir.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print("\n=== Cohort mean TRE75 (mm) ===")
    for kind in MODELS:
        print(f"  {kind:8s} {summary[kind]['mean_tre75_mm']:.3f}")
    print("Wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
