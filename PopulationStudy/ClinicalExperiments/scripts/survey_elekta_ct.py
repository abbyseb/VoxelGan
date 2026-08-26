#!/usr/bin/env python3
"""Clinical Elekta patient mid-slice survey (mirror of Varian survey).

Reads Evaluation GTVol + Mask_Lung from SpareDVFs ClinicalElektaDatasets.
Writes panels under ClinicalExperiments/plots/elekta_ct_survey/.

  python scripts/survey_elekta_ct.py
"""
from __future__ import annotations

from pathlib import Path

import itk
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/abhishek/SpareDVFs/SPARE_GroundTruth/ClinicalElektaDatasets")
OUT = Path(__file__).resolve().parents[1] / "plots" / "elekta_ct_survey"
OUT.mkdir(parents=True, exist_ok=True)

patients = [f"P{i}" for i in range(1, 6)]
phase = 6  # max exhale — diaphragm often clearest


def load(path: Path) -> np.ndarray:
    return itk.array_from_image(itk.imread(str(path))).astype(np.float32)


def window(x, lo=None, hi=None):
    if lo is None:
        lo = float(np.percentile(x, 1))
    if hi is None:
        hi = float(np.percentile(x, 99))
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((x - lo) / (hi - lo), 0, 1)


def lung_bounds(mask: np.ndarray):
    # mask (AP, SI, LR) — same SPARE Evaluation convention as Varian
    ap = np.where(mask.any(axis=(1, 2)))[0]
    si = np.where(mask.any(axis=(0, 2)))[0]
    lr = np.where(mask.any(axis=(0, 1)))[0]
    return dict(
        ap=(int(ap[0]), int(ap[-1]), int(ap[len(ap) // 2])),
        si=(int(si[0]), int(si[-1]), int(si[len(si) // 2])),
        lr=(int(lr[0]), int(lr[-1]), int(lr[len(lr) // 2])),
    )


def main():
    rows = []
    for pid in patients:
        pdir = ROOT / pid
        scans = sorted(
            [d.name for d in pdir.iterdir() if d.is_dir() and "_V_" in d.name]
        )
        scan = f"CE_{pid}_V_01"
        if scan not in scans:
            if not scans:
                print(f"{pid}: no V scans — skip")
                continue
            scan = scans[0]
        ev = pdir / scan
        ct = load(ev / f"GTVol_{phase:02d}.mha")
        mask = load(ev / "Mask_Lung.mha") > 0
        b = lung_bounds(mask)
        ap0, ap1, ap_mid = b["ap"]
        rows.append(
            dict(pid=pid, scan=scan, ct=ct, mask=mask, b=b, ap_mid=ap_mid)
        )
        print(f'{pid} {scan} shape={ct.shape} lung AP{b["ap"]} SI{b["si"]} LR{b["lr"]}')

    # --- Figure 1: mid-AP + mid-LR sagittal ---
    fig, axs = plt.subplots(2, 5, figsize=(18, 7.5))
    for i, r in enumerate(rows):
        lo = float(np.percentile(r["ct"], 1))
        hi = float(np.percentile(r["ct"], 99))
        sl = window(r["ct"][r["ap_mid"]], lo, hi)
        m = r["mask"][r["ap_mid"]]
        ax = axs[0, i]
        ax.imshow(np.rot90(sl, 2), cmap="gray", origin="upper", aspect="equal", vmin=0, vmax=1)
        ax.contour(np.rot90(m.astype(float), 2), levels=[0.5], colors="lime", linewidths=0.7)
        ax.set_title(f"{r['pid']} mid-AP\n{r['scan']}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

        lr_mid = r["b"]["lr"][2]
        sag = window(r["ct"][:, :, lr_mid], lo, hi)
        ms = r["mask"][:, :, lr_mid]
        ax = axs[1, i]
        ax.imshow(np.rot90(sag, 1), cmap="gray", origin="upper", aspect="equal", vmin=0, vmax=1)
        ax.contour(np.rot90(ms.astype(float), 1), levels=[0.5], colors="lime", linewidths=0.7)
        ax.set_title(f"{r['pid']} mid-LR sagittal\n(SI×AP)", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"Clinical Elekta P1–P5 · V_01 · phase {phase:02d} (max exhale) — green=lung mask\n"
        "Top: mid-AP (SI×LR, coronal-like — look for diaphragm at caudal lung)\n"
        "Bottom: mid-LR sagittal (diaphragm dome along SI)",
        fontsize=11,
    )
    fig.tight_layout()
    out1 = OUT / "all_patients_mid_diaphragm.png"
    fig.savefig(out1, dpi=140)
    plt.close()
    print("wrote", out1)

    # --- Figure 2: mid-AP only ---
    fig, axs = plt.subplots(1, 5, figsize=(18, 4))
    for i, r in enumerate(rows):
        lo = float(np.percentile(r["ct"], 1))
        hi = float(np.percentile(r["ct"], 99))
        sl = window(r["ct"][r["ap_mid"]], lo, hi)
        m = r["mask"][r["ap_mid"]]
        axs[i].imshow(np.rot90(sl, 2), cmap="gray", origin="upper", aspect="equal", vmin=0, vmax=1)
        axs[i].contour(np.rot90(m.astype(float), 2), levels=[0.5], colors="lime", linewidths=0.8)
        axs[i].set_title(f"{r['pid']}", fontsize=12)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    fig.suptitle(
        f"Clinical Elekta · mid-AP lung slice · phase {phase:02d} (look caudal / bottom for diaphragm)",
        fontsize=12,
    )
    fig.tight_layout()
    out2 = OUT / "all_patients_midAP_phase06.png"
    fig.savefig(out2, dpi=150)
    plt.close()
    print("wrote", out2)

    # --- Figure 3: all V scans per patient ---
    max_v = max(
        len([d for d in (ROOT / pid).iterdir() if d.is_dir() and "_V_" in d.name])
        for pid in patients
        if (ROOT / pid).is_dir()
    )
    ncols = max(max_v, 1)
    fig, axs = plt.subplots(5, ncols, figsize=(2.8 * ncols, 14))
    if ncols == 1:
        axs = np.array([[axs[i]] for i in range(5)])
    for ri, pid in enumerate(patients):
        scans = sorted(
            [d.name for d in (ROOT / pid).iterdir() if d.is_dir() and "_V_" in d.name]
        )
        for ci in range(ncols):
            ax = axs[ri, ci]
            if ci >= len(scans):
                ax.axis("off")
                continue
            scan = scans[ci]
            ev = ROOT / pid / scan
            ct = load(ev / f"GTVol_{phase:02d}.mha")
            mask = load(ev / "Mask_Lung.mha") > 0
            b = lung_bounds(mask)
            lo = float(np.percentile(ct, 1))
            hi = float(np.percentile(ct, 99))
            sl = window(ct[b["ap"][2]], lo, hi)
            m = mask[b["ap"][2]]
            ax.imshow(np.rot90(sl, 2), cmap="gray", origin="upper", aspect="equal", vmin=0, vmax=1)
            ax.contour(np.rot90(m.astype(float), 2), levels=[0.5], colors="lime", linewidths=0.5)
            ax.set_title(scan.replace("CE_", ""), fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if ci == 0:
                ax.set_ylabel(pid, fontsize=11, fontweight="bold")
    fig.suptitle(f"All Clinical Elekta validation scans · mid-AP · phase {phase:02d}", fontsize=12)
    fig.tight_layout()
    out3 = OUT / "all_scans_midAP_phase06.png"
    fig.savefig(out3, dpi=120)
    plt.close()
    print("wrote", out3)
    print("OUT=", OUT)


if __name__ == "__main__":
    main()
