#!/usr/bin/env python3
"""Slide figures for the timeline after the small-network tests: DIR-Lab C01 and C08, per step.

Each row = one case and one model: Elastix |u| (reference), model |u|, the input CT (T50) warped by the model,
and |warped - real T00| (the error image). Mid-coronal slice through the lung, fixed 2 mm scorer grid
(eval_dir_tcia3_iso2mm_v2). warped(x) = T50(x + u(x)) should match T00(x) (as ncc_dirlab.py).
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
CASES = (1, 8)
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
HU_WIN = (-1000, 400)
DIFF_MAX = 500.0


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
        z = int(np.round(np.nonzero(p["lung_iso"].any(axis=(1, 2)))[0].mean()))
        D[c] = {"p": p, "t00": t00, "reg": reg, "z": z, "el": v2.elastix_on_iso(p)}
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


def row_stats(D, c, u):
    d = D[c]
    w = warp(d["p"]["hu_iso"], u)
    return w, v2.score_set(u, c, "300", d["p"])["registered"]["mean"], ncc(w[d["reg"]], d["t00"][d["reg"]])


def figure(D, rows, title, fname):
    """rows: list of (case, label, u or None for 'no motion')."""
    fig, ax = plt.subplots(len(rows), 4, figsize=(15, 3.9 * len(rows)), squeeze=False)
    stats = []
    for r, (c, label, u) in enumerate(rows):
        d = D[c]
        z, el = d["z"], d["el"]
        u = np.zeros_like(el) if u is None else u
        w, t300, nc = row_stats(D, c, u)
        stats.append({"case": c, "row": label, "tre300": round(t300, 3), "lung_ncc": round(nc, 4)})
        vmax = float(np.percentile(np.linalg.norm(el[z], axis=-1)[d["p"]["lung_iso"][z]] * 2.0, 99))
        mags = [np.linalg.norm(el[z], axis=-1) * 2.0, np.linalg.norm(u[z], axis=-1) * 2.0]
        for k, (m, t) in enumerate(zip(mags, ["Elastix motion |u| (mm)", f"{label}: motion |u| (mm)"])):
            im = ax[r, k].imshow(m, cmap="magma", vmin=0, vmax=vmax)
            ax[r, k].set_title(t, fontsize=10)
            plt.colorbar(im, ax=ax[r, k], fraction=0.046)
        ax[r, 2].imshow(w[z], cmap="gray", vmin=HU_WIN[0], vmax=HU_WIN[1])
        ax[r, 2].set_title(f"T50 warped toward T00 · lung NCC {nc:.3f}", fontsize=10)
        im = ax[r, 3].imshow(np.abs(w[z] - d["t00"][z]), cmap="inferno", vmin=0, vmax=DIFF_MAX)
        ax[r, 3].set_title(f"|warped − real T00| (HU) · TRE300 {t300:.2f} mm", fontsize=10)
        plt.colorbar(im, ax=ax[r, 3], fraction=0.046)
        ax[r, 0].set_ylabel(f"C{c:02d}", fontsize=13, rotation=0, labelpad=25)
        for a in ax[r]:
            a.set_xticks([]), a.set_yticks([])
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIGS / fname, dpi=100)
    plt.close(fig)
    print("wrote", fname, flush=True)
    return stats


def main():
    D = load_cases()
    U = predict_all(D)
    E = {c: D[c]["el"] for c in CASES}
    S = {}
    rc = lambda rows: [(c, lab, u(c)) for c in CASES for lab, u in rows]  # noqa: E731
    S["T0"] = figure(D, rc([("No motion", lambda c: None), ("Elastix", lambda c: E[c])]),
                     "Reference: no motion vs Elastix (the label the models learn from)", "T0_reference.png")
    S["T1"] = figure(D, rc([("A′-full160", lambda c: U[("full160", "plain", c)])]),
                     "30 Sep · Whole-lung view (A′-full160, epoch 40, no augmentation)", "T1_full160.png")
    S["T2"] = figure(D, rc([("TCIA-lite", lambda c: U[("lite", "plain", c)])]),
                     "1 Oct · + augmentation and cosine lr (full160-aug = TCIA-lite, epoch 40)", "T2_lite_aug.png")
    S["T3"] = figure(D, rc([("Big, plain", lambda c: U[("big_old_s1", "plain", c)]),
                            ("Big, flip", lambda c: U[("big_old_s1", "flip", c)])]),
                     "1 Oct · Mirror (flip) averaging on the big model (TCIA3.5 seed 1, epoch 100)", "T3_flip.png")
    S["T4"] = figure(D, rc([("Small, old loss", lambda c: U[("lite", "plain", c)]),
                            ("Small, hybrid", lambda c: U[("small_h_s1", "plain", c)])]),
                     "2 Oct · Hybrid loss on the small model (seed 1, epoch 40)", "T4_hybrid_small.png")
    S["T5"] = figure(D, rc([("Big old, flip", lambda c: U[("big_old_s1", "flip", c)]),
                            ("Big hybrid, flip", lambda c: U[("big_hyb_s1", "flip", c)])]),
                     "2–4 Oct · Hybrid loss on the big model, seed 1 (epoch 100, flip)", "T5_hybrid_big_s1.png")
    S["T6"] = figure(D, rc([("Big old s2, flip", lambda c: U[("big_old_s2", "flip", c)]),
                            ("Big hybrid s2, flip", lambda c: U[("big_hyb_s2", "flip", c)])]),
                     "4–6 Oct · Second seed: the hybrid gain does not repeat (epoch 100, flip)", "T6_seed2.png")
    S["T7"] = figure(D, rc([("Combination", lambda c: U[("combo", "flip", c)])]),
                     "8 Oct · Pre-set combination: 2 small hybrid + 2 big old models, flip, fields averaged",
                     "T7_combination.png")
    (FIGS / "timeline_stats.json").write_text(json.dumps(S, indent=1))
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
