#!/usr/bin/env python3
"""Visual check of a synthesiser's own motion field on DIR-Lab (verified v2 geometry).

One figure per case:
  top    : T00 landmarks on T00 CT | T50 landmarks on T50 CT | network motion size |u| (mm)
  bottom : arrows at real length (true T00->T50 orange, Elastix green, network cyan)
           | T00 - T50 before | T00 - T50 warped by the NETWORK field
Coronal slab through the landmark centre. Nothing is written except PNGs + a small JSON.

  cd "DIR EXPERIMENTS"
  CUDA_VISIBLE_DEVICES=1 /home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python \
      scripts/viz_network_fields.py --ckpt TCIA3=<.../TCIA3/DecoderCRB/checkpoints/epoch_100.pt> --cases 1 8
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
from scipy.ndimage import map_coordinates

DIR_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIR_EXP / "scripts"))
import eval_dir_tcia3_iso2mm_v2 as v2  # noqa: E402
from dirlab_tre import landmarks_300  # noqa: E402
from rescore_oracle_iso2mm_v3 import load_ckpt  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="LABEL=PATH")
    ap.add_argument("--cases", type=int, nargs="*", default=[1, 8])
    args = ap.parse_args()
    label, path = args.ckpt.split("=", 1)
    os.environ.setdefault("DIRLAB_ROOT", str(DIR_EXP / "data" / "dirlab_packs"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    g = load_ckpt(Path(path), device)
    out_dir = v2.OUT_DIR / f"viz_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for case in args.cases:
        pack = v2.pack_case(case)
        ct06 = pack["hu_iso"]
        img = sitk.ReadImage(str(pack["train"] / "CT_01.mha"))
        ct01 = v2.resample_to_iso(sitk.GetArrayFromImage(img).astype(np.float32), pack["origin_xyz"],
                                  pack["spacing_xyz"], pack["origin_iso_xyz"], 1, -1000.0)
        l00 = v2.official_to_iso(landmarks_300(case, "T00"), pack)
        l50 = v2.official_to_iso(landmarks_300(case, "T50"), pack)
        u_net = v2.predict(g, device, pack["mu"], flip=False)
        u_el = v2.elastix_on_iso(pack)
        p_net = l00 + v2.sample_u(u_net, l00)
        p_el = l00 + v2.sample_u(u_el, l00)
        tre_net = float(np.linalg.norm((p_net - l50) * 2, axis=1).mean())
        tre_el = float(np.linalg.norm((p_el - l50) * 2, axis=1).mean())
        true_len = np.linalg.norm((l50 - l00) * 2, axis=1)
        net_len = np.linalg.norm((p_net - l00) * 2, axis=1)
        summary[case] = {"tre300_net": tre_net, "tre300_elastix": tre_el,
                         "mean_true_motion_mm": float(true_len.mean()),
                         "mean_net_motion_mm": float(net_len.mean()),
                         "size_ratio_net_over_true": float(net_len.mean() / max(true_len.mean(), 1e-6))}
        print(f"C{case:02d} {summary[case]}", flush=True)

        zc = int(np.round(np.median(l00[:, 2])))
        slab = np.abs(l00[:, 2] - zc) <= 6
        s50 = np.abs(l50[:, 2] - zc) <= 6
        kw = dict(cmap="gray", vmin=-1000, vmax=300, origin="upper")
        fig, ax = plt.subplots(2, 3, figsize=(16, 11))
        ax[0, 0].imshow(ct01[zc], **kw)
        ax[0, 0].scatter(l00[slab, 0], l00[slab, 1], s=14, c="lime", edgecolors="k", lw=.4)
        ax[0, 0].set_title(f"C{case:02d} T00 CT + T00 landmarks")
        ax[0, 1].imshow(ct06[zc], **kw)
        ax[0, 1].scatter(l50[s50, 0], l50[s50, 1], s=14, c="orange", edgecolors="k", lw=.4)
        ax[0, 1].set_title("T50 CT (network input) + T50 landmarks")
        mag = np.linalg.norm(u_net[zc], axis=-1) * 2
        im = ax[0, 2].imshow(mag, cmap="magma", origin="upper", vmin=0, vmax=max(15, float(true_len.max())))
        fig.colorbar(im, ax=ax[0, 2], fraction=.046, label="mm")
        ax[0, 2].set_title(f"{label}: motion size |u|")
        ax[1, 0].imshow(ct01[zc], **kw)
        for p, col, w, lab in ((l50, "orange", .004, "true"), (p_el, "lime", .0025, f"Elastix ({tre_el:.2f} mm)"),
                               (p_net, "cyan", .0025, f"{label} ({tre_net:.2f} mm)")):
            ax[1, 0].quiver(l00[slab, 0], l00[slab, 1], (p - l00)[slab, 0], (p - l00)[slab, 1], color=col,
                            angles="xy", scale_units="xy", scale=1, width=w, label=lab)
        ax[1, 0].legend(loc="lower right", fontsize=8)
        ax[1, 0].set_title(f"Arrows at real length (net/true size {summary[case]['size_ratio_net_over_true']:.2f})")
        zz, yy, xx = np.meshgrid(*(np.arange(s, dtype=np.float32) for s in ct06.shape), indexing="ij")
        warped = map_coordinates(ct06, np.stack([zz + u_net[..., 2], yy + u_net[..., 1], xx + u_net[..., 0]]),
                                 order=1, mode="nearest")
        dk = dict(cmap="RdBu_r", vmin=-400, vmax=400, origin="upper")
        ax[1, 1].imshow((ct01 - ct06)[zc], **dk)
        ax[1, 1].set_title("BEFORE: T00 − T50")
        ax[1, 2].imshow((ct01 - warped)[zc], **dk)
        ax[1, 2].set_title(f"AFTER: T00 − T50 warped by {label}")
        for a in ax.ravel():
            a.set_xticks([])
            a.set_yticks([])
        fig.tight_layout()
        fig.savefig(out_dir / f"verify_C{case:02d}.png", dpi=80)
        plt.close(fig)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
