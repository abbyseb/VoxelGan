#!/usr/bin/env python3
"""Re-score the SPARE-trained synthesisers on the corrected DIR path (v2 geometry).

SPARE data_iso may use a different numpy axis layout from TCIA R3 (its meta.json
says DVF channels are (LR, AP, SI), TCIA is (LR, SI, AP)). So the layout is found
from anatomy, not assumed:

  For every axis permutation (6) x flip pattern (8), transform the DIR C01 lung
  mask and measure Dice with the training lung mask (both lung-centred 160^3, 2 mm).
  The best of the 48 is the layout. The same search is run against TCIA S1 as a
  control: it must pick "no permutation, no flip" (the verified TCIA path).

Then each SPARE model gets DIR in its own layout, and its field is transformed
back to the TCIA/R3 frame (spatial axes AND vector components, with sign flips)
before the verified landmark scoring. Mirror averaging is reported too.
Weights are only read. Also writes an orientation picture.

  cd "DIR EXPERIMENTS"
  CUDA_VISIBLE_DEVICES=1 python scripts/rescore_spare_iso2mm.py
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

DIR_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIR_EXP / "scripts"))
import eval_dir_tcia3_iso2mm_v2 as v2  # noqa: E402

POP = v2.VOXEL / "PopulationStudy"
G160 = POP / "ClinicalExperiments/Grid160"
ISO_E2 = POP / "IsoExperiments/Experiment2"
SPARE_P1 = POP / "data_iso/P1/all"
NET = v2.NET  # networks: generator_crb_dec / generator_crb / generator_crb_both

MODELS = [
    # label, path, architecture
    ("SPARE_A1_dec_fov_full", G160 / "Experiment1/DecoderCRB/weights/crb_dec_mse_iso_g160_fov_full_generator.pth", "dec"),
    ("SPARE_A1_enc_fov_full", G160 / "Experiment1/EncoderCRB/weights/crb_enc_mse_iso_g160_fov_full_generator.pth", "enc"),
    ("SPARE_A1_both_fov_full", G160 / "Experiment1/BothCRB/weights/crb_both_mse_iso_g160_fov_full_generator.pth", "both"),
    ("SPARE_A0_dec_full", ISO_E2 / "DecoderCRB/weights/crb_dec_mse_iso_e2_full_generator.pth", "dec"),
    ("SPARE_A0_enc_full", ISO_E2 / "EncoderCRB/weights/crb_enc_mse_iso_e2_full_generator.pth", "enc"),
    ("SPARE_A0_both_full", ISO_E2 / "BothCRB/weights/crb_both_mse_iso_e2_full_generator.pth", "both"),
    ("SPARE_A0h_dec_holdout", ISO_E2 / "DecoderCRB/weights/crb_dec_mse_iso_e2_generator.pth", "dec"),
    ("SPARECRB_mae_p2p8_best", G160 / "SPARECRB/DecoderCRB/weights/crb_dec_mae_full160_spare_p2p8_generator.pth", "dec"),
]
REFERENCE = ("TCIA3_ep100", G160 / "TCIA3/DecoderCRB/checkpoints/epoch_100.pt", "dec")


# ------------------------------------------------------------------ layout search
def transform(vol, perm, flips):
    out = np.transpose(vol, perm)
    for ax, f in enumerate(flips):
        if f:
            out = np.flip(out, axis=ax)
    return np.ascontiguousarray(out)


def field_back(u_m, perm, flips):
    """Field predicted in model layout -> R3 layout.

    u_m: (Z,Y,X,3) in model layout, channel k is displacement along model numpy
    axis (2 - k). Undo flips (spatial + negate that component), undo the
    transpose (spatial), then move each component to the R3 axis it came from.
    """
    u = u_m.copy()
    for ax, f in enumerate(flips):
        if f:
            u = np.flip(u, axis=ax)
            u[..., 2 - ax] *= -1.0
    inv = np.argsort(perm)
    u = np.transpose(u, list(inv) + [3])
    out = np.empty_like(u)
    for a_model in range(3):          # model numpy axis a_model came from R3 axis perm[a_model]
        out[..., 2 - perm[a_model]] = u[..., 2 - a_model]
    return np.ascontiguousarray(out)


def centre_on_lung(mask):
    """Shift a 160^3 mask so its lung centroid sits at the cube centre (integer roll)."""
    c = np.array(np.nonzero(mask)).mean(axis=1)
    shift = np.round((np.array(mask.shape) - 1) / 2 - c).astype(int)
    return np.roll(mask, tuple(shift), axis=(0, 1, 2))


def dice(a, b):
    return float(2 * (a & b).sum() / (a.sum() + b.sum() + 1e-9))


def find_layout(dir_lung, train_lung):
    ref = centre_on_lung(train_lung)
    scores = []
    for perm in itertools.permutations(range(3)):
        for flips in itertools.product((0, 1), repeat=3):
            m = centre_on_lung(transform(dir_lung, perm, flips))
            scores.append((dice(m, ref), perm, flips))
    scores.sort(reverse=True)
    return scores


# ------------------------------------------------------------------ models
def load_model(path, arch, device):
    if str(NET) not in sys.path:
        sys.path.insert(0, str(NET))
    if arch == "dec":
        from networks.generator_crb_dec import UNetCRBDecoder as C
    elif arch == "enc":
        from networks.generator_crb import UNetCRB as C
    else:
        from networks.generator_crb_both import UNetCRBBoth as C
    g = C(im_size=160, n_phases=10)
    raw = torch.load(str(path), map_location=device, weights_only=False)
    if isinstance(raw, dict):
        for k in ("generator", "state_dict", "model"):
            if k in raw and isinstance(raw[k], dict):
                raw = raw[k]
                break
    g.load_state_dict(raw, strict=True)
    return g.to(device).eval()


def predict_model_layout(g, device, mu_r3, perm, flips, mirror):
    """Run in model layout, return R3 field. mirror: also average the LR-mirrored pass."""
    def one(mu):
        x = torch.from_numpy(v2.minmax(transform(mu, perm, flips))[None, None]).to(device)
        with torch.no_grad():
            d = g(x, torch.tensor([5], device=device), torch.tensor([0], device=device))[0]
        return field_back(np.moveaxis(d.cpu().numpy(), 0, -1).astype(np.float32), perm, flips)

    a = one(mu_r3)
    if not mirror:
        return a
    b = one(np.ascontiguousarray(mu_r3[:, :, ::-1]))          # mirror in R3 (X = left-right)
    b = np.ascontiguousarray(b[:, :, ::-1, :])
    b[..., 0] *= -1.0
    return 0.5 * (a + b)


def save_png(spare_ct, spare_lung, dir_mu_model, dir_lung_model, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def mids(v, m):
        c = np.round(np.array(np.nonzero(m)).mean(axis=1)).astype(int)
        return v[c[0]], v[:, c[1], :], v[:, :, c[2]]

    rows = [("SPARE P1 (train)", mids(spare_ct, spare_lung)),
            ("DIR C01 in SPARE layout", mids(dir_mu_model, dir_lung_model))]
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    for r, (t, sl) in enumerate(rows):
        for c, im in enumerate(sl):
            ax[r, c].imshow(im, cmap="gray", origin="upper")
            ax[r, c].set_title(f"{t} — slice along axis {c}")
    fig.suptitle("Both rows must look the same way round")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main() -> None:
    os.environ.setdefault("DIRLAB_ROOT", str(DIR_EXP / "data" / "dirlab_packs"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cases = list(range(1, 11))
    packs = {c: v2.pack_case(c) for c in cases}
    dir_lung = packs[1]["lung_iso"]

    # control: TCIA S1 must give identity
    tcia_lung = np.load(v2.TCIA_LUNG) > 0
    t_scores = find_layout(dir_lung, tcia_lung)
    t_best = t_scores[0]
    print(f"[layout] TCIA S1 control best: perm {t_best[1]} flips {t_best[2]} dice {t_best[0]:.3f} "
          f"(2nd {t_scores[1][0]:.3f})", flush=True)
    if t_best[1] != (0, 1, 2) or t_best[2] != (0, 0, 0):
        raise SystemExit("control failed: layout search does not recover the verified TCIA frame")

    spare_lung = np.load(SPARE_P1 / "Mask_Lung.npy") > 0
    spare_ct = np.load(SPARE_P1 / "CT_06.npy").astype(np.float32)
    s_scores = find_layout(dir_lung, spare_lung)
    s_best = s_scores[0]
    perm, flips = s_best[1], s_best[2]
    print(f"[layout] SPARE P1 best: perm {perm} flips {flips} dice {s_best[0]:.3f} "
          f"(2nd {s_scores[1][0]:.3f} perm {s_scores[1][1]} flips {s_scores[1][2]})", flush=True)
    png = v2.OUT_DIR / "orientation_spare_p1_vs_dir_c01.png"
    save_png(spare_ct, spare_lung, transform(packs[1]["mu"], perm, flips),
             transform(dir_lung, perm, flips), png)
    print(f"[layout] picture {png}", flush=True)

    rows = {}
    for label, path, arch in MODELS + [REFERENCE]:
        if not path.is_file():
            print(f"skip {label}: missing {path}")
            continue
        p, f = ((0, 1, 2), (0, 0, 0)) if label.startswith("TCIA") else (perm, flips)
        try:
            g = load_model(path, arch, device)
        except Exception as e:  # noqa: BLE001
            print(f"skip {label}: {e}")
            continue
        for mirror in (False, True):
            t75, t300 = [], []
            per = {}
            for c in cases:
                u = predict_model_layout(g, device, packs[c]["mu"], p, f, mirror)
                a = v2.score_set(u, c, "75", packs[c])["registered"]["mean"]
                b = v2.score_set(u, c, "300", packs[c])["registered"]["mean"]
                t75.append(a)
                t300.append(b)
                per[str(c)] = b
            key = f"{label}|{'mirror' if mirror else 'plain'}"
            rows[key] = {"tre75": float(np.mean(t75)), "tre300": float(np.mean(t300)),
                         "per_case_tre300": per, "layout": {"perm": p, "flips": f}}
            print(f"{key:40s} TRE75 {rows[key]['tre75']:.3f}  TRE300 {rows[key]['tre300']:.3f}  "
                  f"C08 {per['8']:.2f}", flush=True)
        del g
        torch.cuda.empty_cache()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = v2.OUT_DIR / f"rescore_spare_iso2mm_{stamp}.json"
    out.write_text(json.dumps({
        "when": datetime.now(timezone.utc).astimezone().isoformat(),
        "tcia_control": {"perm": t_best[1], "flips": t_best[2], "dice": t_best[0]},
        "spare_layout": {"perm": perm, "flips": flips, "dice": s_best[0],
                         "runner_up": {"perm": s_scores[1][1], "flips": s_scores[1][2], "dice": s_scores[1][0]}},
        "rows": rows,
    }, indent=2, default=list) + "\n")

    print("\n============ SPARE vs TCIA on corrected DIR (mm) ============")
    for k in sorted(rows, key=lambda k: rows[k]["tre300"]):
        r = rows[k]
        print(f"{k:40s} TRE75 {r['tre75']:.3f}  TRE300 {r['tre300']:.3f}  C08 {r['per_case_tre300']['8']:.2f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
