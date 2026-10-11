#!/usr/bin/env python3
"""Slide figures for the timeline after the small-network tests: DIR-Lab C01, C04 and C08, per step.

Same 6-panel layout as slide 5b (analysis_2026-10-03/explain_figs.py), one PNG per model; steps that compare two
models get _a/_b files. Mid-coronal slice through the lung, fixed 2 mm scorer grid (eval_dir_tcia3_iso2mm_v2). warped(x) = T50(x + u(x)) should match T00(x) (as ncc_dirlab.py).
TRE300 and lung NCC printed per row are for THIS checkpoint only (not the last-5-epoch / 2-seed headline).

  cd "DIR EXPERIMENTS"; CUDA_VISIBLE_DEVICES=1 <LEARN-GUI venv python> slides/make_timeline_figs.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
from scipy.ndimage import binary_dilation, map_coordinates

DIR_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIR_EXP / "scripts"))
os.environ.setdefault("DIRLAB_ROOT", str(DIR_EXP / "data" / "dirlab_packs"))
import eval_dir_tcia3_iso2mm_v2 as v2  # noqa: E402
import mirror_tta_iso2mm as mt  # noqa: E402
from rescore_oracle_iso2mm_v3 import load_ckpt  # noqa: E402

G = Path("/home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Grid160")
L = G / "TCIA_lite"
FIGS = DIR_EXP / "slides" / "figs"
CASES = (1, 4, 8)
VOX_MM = 2.0  # iso grid spacing; u and Elastix fields are in voxels
CK = {
    "full160": L / "run_A2_full160/DecoderCRB/checkpoints/epoch_040.pt",
    "lite": L / "run_A2_full160_aug/DecoderCRB/checkpoints/epoch_040.pt",
    "small_h_s1": L / "run_A2_full160_aug_hybrid/DecoderCRB/checkpoints/epoch_040.pt",
    "small_h_s2": L / "run_A2_full160_aug_hybrid_s2/DecoderCRB/checkpoints/epoch_040.pt",
    "big_old_s1": G / "TCIA3.5/DecoderCRB/checkpoints/epoch_100.pt",
    "big_old_s2": G / "TCIA3.5_s2/DecoderCRB/checkpoints/epoch_100.pt",
    "big_hyb_s1": G / "TCIA3.5_hybrid/DecoderCRB/checkpoints/epoch_100.pt",
    "big_hyb_s2": G / "TCIA3.5_hybrid_s2/DecoderCRB/checkpoints/epoch_100.pt",
}


def warp(img, u):
    zz, yy, xx = np.meshgrid(*(np.arange(n, dtype=np.float32) for n in img.shape), indexing="ij")
    return map_coordinates(img, [zz + u[..., 2], yy + u[..., 1], xx + u[..., 0]], order=1, mode="nearest")


def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def load_cases():
    D = {}
    for c in CASES:
        p = v2.pack_case(c)
        ct = sitk.ReadImage(str(p["train"] / "CT_01.mha"))
        t00 = v2.resample_to_iso(sitk.GetArrayFromImage(ct).astype(np.float32), p["origin_xyz"], p["spacing_xyz"],
                                 p["origin_iso_xyz"], order=1, cval=-1000.0)
        reg = binary_dilation(p["lung_iso"], iterations=5)
        z = int(np.round(np.where(p["hu_iso"] < -400)[0].mean()))  # same slice rule as slide 5b (explain_figs.py)
        D[c] = {"p": p, "t00": t00, "reg": reg, "z": z, "el": v2.elastix_on_iso(p),
                "tre0": tre(np.zeros((*p["hu_iso"].shape, 3), np.float32), c, p)}
    return D


def predict_all(D):
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    U = {}
    for name, path in CK.items():
        g = load_ckpt(path, dev)
        for c in CASES:
            a, m = mt.predict_mirror(g, dev, D[c]["p"]["mu"])
            U[(name, "plain", c)], U[(name, "flip", c)] = a, m
        del g
        torch.cuda.empty_cache()
        print("predicted", name, flush=True)
    for c in CASES:  # pre-set combination: mean of the four flip fields (as combo_small_big.py)
        U[("combo", "flip", c)] = np.mean([U[(k, "flip", c)] for k in ("small_h_s1", "small_h_s2", "big_old_s1",
                                                                         "big_old_s2")], 0)
    return U


def tre(u, c, p):
    """{'75': (mean, sd), '300': (mean, sd)} in mm over the case's landmarks."""
    return {k: (r["mean"], r["std"]) for k in ("75", "300") for r in [v2.score_set(u, c, k, p)["registered"]]}


def row_stats(D, c, u):
    d = D[c]
    w = warp(d["p"]["hu_iso"], u)
    return w, tre(u, c, d["p"]), ncc(w[d["reg"]], d["t00"][d["reg"]])


def pm(t):
    return f"{t[0]:.2f} ± {t[1]:.2f}"


def figure(D, label, ufn, title, fname):
    """Slide-5b layout (as analysis_2026-10-03/explain_figs.py), one model, rows C01/C04/C08.
    Columns: input T50, target T00, warped by model, target - input, target - warped, |u| with model/Elastix arrows."""
    H = 5.0 * len(CASES) + 0.8  # +0.8 in for the two-line title
    fig, ax = plt.subplots(len(CASES), 6, figsize=(24, H), squeeze=False)
    stats = []
    kw = dict(cmap="gray", vmin=-1000, vmax=300)
    for i, c in enumerate(CASES):
        d = D[c]
        z, el, inp, tgt = d["z"], d["el"], d["p"]["hu_iso"], d["t00"]
        u = ufn(c)
        w, t, nc = row_stats(D, c, u)
        stats.append({"case": c, "row": label, "tre75": round(t["75"][0], 3), "tre75_sd": round(t["75"][1], 3),
                      "tre300": round(t["300"][0], 3), "tre300_sd": round(t["300"][1], 3), "lung_ncc": round(nc, 4)})
        ax[i, 0].imshow(inp[z], **kw); ax[i, 0].set_title(f"C{c:02d} input CT_06 (T50, exhale)\nno motion: TRE75 {pm(d['tre0']['75'])} · "
                                          f"TRE300 {pm(d['tre0']['300'])} mm", fontsize=9.5)
        ax[i, 1].imshow(tgt[z], **kw); ax[i, 1].set_title("target CT_01 (T00, inhale)", fontsize=10)
        ax[i, 2].imshow(w[z], **kw); ax[i, 2].set_title(f"warped by model\nTRE75 {pm(t['75'])} · TRE300 {pm(t['300'])} mm", fontsize=9.5)
        ax[i, 3].imshow(tgt[z] - inp[z], cmap="bwr", vmin=-600, vmax=600); ax[i, 3].set_title("target - input (before)")
        ax[i, 4].imshow(tgt[z] - w[z], cmap="bwr", vmin=-600, vmax=600)
        ax[i, 4].set_title("target - warped (after)")
        um, em = u[z] * VOX_MM, el[z] * VOX_MM
        mag, emag = np.linalg.norm(um, axis=-1), np.linalg.norm(em, axis=-1)
        ax[i, 5].imshow(mag, cmap="magma", vmin=0, vmax=max(emag.max(), mag.max(), 1))
        st = 6
        yy, xx = np.mgrid[0:mag.shape[0]:st, 0:mag.shape[1]:st]
        q = dict(angles="xy", scale_units="xy", scale=0.5)
        ax[i, 5].quiver(xx, yy, um[::st, ::st, 0], um[::st, ::st, 1], color="cyan", width=0.003, label="model", **q)
        ax[i, 5].quiver(xx, yy, em[::st, ::st, 0], em[::st, ::st, 1], color="lime", width=0.002, alpha=.6,
                        label="Elastix", **q)
        ax[i, 5].set_title(f"model |DVF| max {mag.max():.1f} mm; Elastix max {emag.max():.1f}")
        ax[i, 5].legend(loc="lower right", fontsize=8)
        for a in ax[i]:
            a.axis("off")
    m75, m300 = (np.mean([s[k] for s in stats]) for k in ("tre75", "tre300"))
    fig.suptitle(f"{title}\n{label} (mean of C01/C04/C08: TRE75 {m75:.2f} mm, TRE300 {m300:.2f} mm; per case "
                 "mean ± SD over landmarks) — coronal slice through lung centre, head at top", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.8 / H), h_pad=2.0)
    fig.savefig(FIGS / fname, dpi=90)
    plt.close(fig)
    print("wrote", fname, flush=True)
    return stats


def main():
    D = load_cases()
    U = predict_all(D)
    E = {c: D[c]["el"] for c in CASES}
    S = {}
    steps = [  # (step, title, [(label, field, file)])
        ("T0", "Reference: Elastix (the label the models learn from)", [("Elastix", lambda c: E[c], "T0_reference.png")]),
        ("T1", "30 Sep · Whole-lung view, epoch 40, no augmentation",
         [("A′-full160", lambda c: U[("full160", "plain", c)], "T1_full160.png")]),
        ("T2", "1 Oct · + augmentation and cosine lr (full160-aug = TCIA-lite, epoch 40)",
         [("TCIA-lite", lambda c: U[("lite", "plain", c)], "T2_lite_aug.png")]),
        ("T3", "1 Oct · Mirror (flip) averaging on the big model (TCIA3.5 seed 1, epoch 100)",
         [("Big, plain", lambda c: U[("big_old_s1", "plain", c)], "T3_flip_a.png"),
          ("Big, flip", lambda c: U[("big_old_s1", "flip", c)], "T3_flip_b.png")]),
        ("T4", "2 Oct · Hybrid loss on the small model (seed 1, epoch 40)",
         [("Small, old loss", lambda c: U[("lite", "plain", c)], "T4_hybrid_small_a.png"),
          ("Small, hybrid", lambda c: U[("small_h_s1", "plain", c)], "T4_hybrid_small_b.png")]),
        ("T5", "2–4 Oct · Hybrid loss on the big model, seed 1 (epoch 100, flip)",
         [("Big old, flip", lambda c: U[("big_old_s1", "flip", c)], "T5_hybrid_big_s1_a.png"),
          ("Big hybrid, flip", lambda c: U[("big_hyb_s1", "flip", c)], "T5_hybrid_big_s1_b.png")]),
        ("T6", "4–6 Oct · Second seed (epoch 100, flip)",
         [("Big old s2, flip", lambda c: U[("big_old_s2", "flip", c)], "T6_seed2_a.png"),
          ("Big hybrid s2, flip", lambda c: U[("big_hyb_s2", "flip", c)], "T6_seed2_b.png")]),
        ("T7", "8 Oct · Pre-set combination: 2 small hybrid + 2 big old models, flip, fields averaged",
         [("Combination", lambda c: U[("combo", "flip", c)], "T7_combination.png")]),
    ]
    for step, title, models in steps:
        S[step] = [r for lab, fn, f in models for r in figure(D, lab, fn, title, f)]
    (FIGS / "timeline_stats.json").write_text(json.dumps(S, indent=1))
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
