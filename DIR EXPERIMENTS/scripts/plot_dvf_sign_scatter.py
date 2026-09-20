#!/usr/bin/env python3
"""Scatter: Elastix SI vs synth raw u / −u (lung-masked), with Pearson r.

Uses ModelTraining 128³ DVFs (A3 DVF already elastix-convention −u; raw = −DVF).
R3 ZYX vector layout: (AP, SI, LR) → SI = component 1.

  cd "DIR EXPERIMENTS"
  python scripts/plot_dvf_sign_scatter.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DIR_EXP = Path(__file__).resolve().parents[1]
A1 = DIR_EXP / "arms" / "A1_oracle_dirlab" / "runs"
A3 = DIR_EXP / "arms" / "A3_synth_conditioned" / "runs"
OUT = (
    DIR_EXP
    / "arms"
    / "A3_synth_conditioned"
    / "plots"
    / "synth_phase_panels"
    / "dvf_sign_scatter"
)

SI = 1  # AP=0, SI=1, LR=2


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def load_case(case: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sid = f"DIR_C{case:02d}"
    el = np.load(A1 / sid / "ModelTraining" / "train" / sid / "DVFs" / "DVF_01_mha.npy")
    syn = np.load(A3 / sid / "ModelTraining" / "train" / sid / "DVFs" / "DVF_01_mha.npy")
    mask = np.load(A1 / sid / "ModelTraining" / "train" / sid / "Masks" / "Mask_Lung_mha.npy")
    el = el.astype(np.float32)
    syn = syn.astype(np.float32)  # already −u (elastix)
    raw = -syn
    m = mask > 0.5
    return el[..., SI][m], raw[..., SI][m], syn[..., SI][m]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    per_raw, per_fix = [], []
    el_all, raw_all, fix_all = [], [], []
    for case in range(1, 11):
        el, raw, fix = load_case(case)
        per_raw.append(corr(el, raw))
        per_fix.append(corr(el, fix))
        el_all.append(el)
        raw_all.append(raw)
        fix_all.append(fix)
        print(f"C{case:02d} SI lung  raw={per_raw[-1]:+.3f}  −u={per_fix[-1]:+.3f}  n={el.size}")

    el = np.concatenate(el_all)
    raw = np.concatenate(raw_all)
    fix = np.concatenate(fix_all)
    r_raw = corr(el, raw)
    r_fix = corr(el, fix)
    print(f"pooled SI lung  raw={r_raw:+.3f}  −u={r_fix:+.3f}  n={el.size}")
    print(f"mean per-case    raw={np.mean(per_raw):+.3f}  −u={np.mean(per_fix):+.3f}")

    # Subsample for hexbin density (deterministic)
    rng = np.random.default_rng(0)
    n_show = min(200_000, el.size)
    idx = rng.choice(el.size, size=n_show, replace=False)
    el_s, raw_s, fix_s = el[idx], raw[idx], fix[idx]

    lim = float(np.percentile(np.abs(np.concatenate([el_s, raw_s, fix_s])), 99.5))
    lim = max(lim, 1.0)
    extent = (-lim, lim)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0), constrained_layout=True)
    panels = [
        (axes[0], raw_s, r_raw, r"raw $u$ (pull)", "Reds_r"),
        (axes[1], fix_s, r_fix, r"$-u$ (Elastix convention)", "Greens"),
    ]
    for ax, y, r, ylabel, cmap in panels:
        hb = ax.hexbin(
            el_s,
            y,
            gridsize=80,
            cmap=cmap,
            mincnt=1,
            bins="log",
            extent=(-lim, lim, -lim, lim),
        )
        ax.plot(extent, extent, "k--", lw=1.0, alpha=0.7, label="y = x")
        ax.plot(extent, (-extent[0], -extent[1]), "k:", lw=1.0, alpha=0.5, label="y = −x")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel(r"Elastix $u_{SI}$ [mm]")
        ax.set_ylabel(f"Synth {ylabel} [mm]")
        ax.set_title(rf"Pearson $r$ = {r:+.3f}")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04, label="log10 count")

    fig.suptitle(
        "DIR-Lab C01–C10 · phase 06→01 · lung-masked SI · "
        f"per-case mean r  raw {np.mean(per_raw):+.2f} / −u {np.mean(per_fix):+.2f}",
        fontsize=11,
    )
    out = OUT / "dvf_si_elastix_vs_synth_sign_scatter.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")

    # Also C01-only (matches coronal panel case)
    el1, raw1, fix1 = load_case(1)
    r1_raw, r1_fix = corr(el1, raw1), corr(el1, fix1)
    n_show = min(80_000, el1.size)
    idx = rng.choice(el1.size, size=n_show, replace=False)
    el_s, raw_s, fix_s = el1[idx], raw1[idx], fix1[idx]
    lim = float(np.percentile(np.abs(np.concatenate([el_s, raw_s, fix_s])), 99.5))
    lim = max(lim, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0), constrained_layout=True)
    for ax, y, r, ylabel, cmap in [
        (axes[0], raw_s, r1_raw, r"raw $u$ (pull)", "Reds_r"),
        (axes[1], fix_s, r1_fix, r"$-u$ (Elastix convention)", "Greens"),
    ]:
        hb = ax.hexbin(
            el_s,
            y,
            gridsize=70,
            cmap=cmap,
            mincnt=1,
            bins="log",
            extent=(-lim, lim, -lim, lim),
        )
        ax.plot((-lim, lim), (-lim, lim), "k--", lw=1.0, alpha=0.7, label="y = x")
        ax.plot((-lim, lim), (lim, -lim), "k:", lw=1.0, alpha=0.5, label="y = −x")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel(r"Elastix $u_{SI}$ [mm]")
        ax.set_ylabel(f"Synth {ylabel} [mm]")
        ax.set_title(rf"DIR_C01 · Pearson $r$ = {r:+.3f}")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04, label="log10 count")
    fig.suptitle(
        "Same physical breath, opposite vector convention — negation aligns the cloud",
        fontsize=11,
    )
    out1 = OUT / "DIR_C01_dvf_si_elastix_vs_synth_sign_scatter.png"
    fig.savefig(out1, dpi=160)
    plt.close(fig)
    print(f"wrote {out1}")


if __name__ == "__main__":
    main()
