#!/usr/bin/env python3
"""C08 projection check: is the breathing cue visible, and does brightness matching change VoxelMap?

No new training. Real A1 projections are DRRs of the real DIR phases. Synth projections
are DRRs of the amplitude-scaled TCIA3 motion. DIR Elastix is not used.

  cd "DIR EXPERIMENTS"
  CUDA_VISIBLE_DEVICES=0 python scripts/probe_c08_proj_breathing.py --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIR_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIR_EXP / "scripts"))
from eval_a1_tre import infer_voxelmap_dvf_phase01  # noqa: E402

A1_MT = (
    DIR_EXP
    / "arms"
    / "A1_oracle_dirlab"
    / "runs"
    / "DIR_C08"
    / "ModelTraining"
    / "train"
    / "DIR_C08"
)
AMP = DIR_EXP / "arms" / "A3_synth_conditioned" / "runs" / "DIR_C08_tcia3_amp4"
CKPT = AMP / "checkpoints_nofilm" / "best.pt"
OUT = AMP / "probe_proj"
SID, SDD = 1000.0, 1500.0
MM_PER_PX = SID / SDD  # 1 detector px = 1 mm; isocenter shift = px * SID/SDD
AMPS = (("a080", 0.8), ("a100", 1.0), ("a130", 1.3), ("a200", 2.0))


def proj_path(root: Path, phase: int, index0: int) -> Path:
    return root / ("TargetProjections" if phase != 6 else "SourceProjections") / (
        f"{phase:02d}_Proj_{index0 + 1:03d}_bin.npy"
    )


def shared_ap_indices() -> np.ndarray:
    """Real indices kept by the stride-4 synth geometry, near AP or PA."""
    ang = np.arange(680) * (360.0 / 680.0)
    keep = []
    for i, a in enumerate(ang):
        if i % 4 != 0:
            continue
        wrapped = min(a % 360.0, 360.0 - (a % 360.0))
        if wrapped < 25.0 or abs((a % 360.0) - 180.0) < 25.0:
            keep.append(i)
    return np.array(keep, dtype=int)


def load_pair(root: Path, real_index: int, renumber: bool) -> tuple[np.ndarray, np.ndarray]:
    j = real_index // 4 if renumber else real_index
    p1 = np.load(proj_path(root, 1, j)).astype(np.float32)
    p6 = np.load(proj_path(root, 6, j)).astype(np.float32)
    return p1, p6


def phase_mae(root: Path, indices: np.ndarray, renumber: bool) -> float:
    vals = []
    for i in indices:
        a, b = load_pair(root, int(i), renumber)
        vals.append(float(np.mean(np.abs(a - b))))
    return float(np.median(vals))


def dome_shift_mm(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Row shift of the steepest falling edge in the central columns. + means A is lower."""

    def peak(img: np.ndarray) -> float:
        gy = np.gradient(img.astype(np.float64), axis=0)
        prof = np.convolve(gy[:, 36:92].mean(1), np.ones(5) / 5.0, mode="same")
        lo, hi = 30, 115
        seg = -prof[lo:hi]
        i = int(np.argmax(seg))
        if 0 < i < len(seg) - 1:
            y0, y1, y2 = seg[i - 1], seg[i], seg[i + 1]
            den = y0 - 2 * y1 + y2
            delta = 0.5 * (y0 - y2) / den if abs(den) > 1e-8 else 0.0
        else:
            delta = 0.0
        return lo + i + float(np.clip(delta, -1.0, 1.0))

    return float((peak(img_a) - peak(img_b)) * MM_PER_PX)


def shift_stats(root: Path, indices: np.ndarray, renumber: bool) -> dict:
    sh = []
    for i in indices:
        a, b = load_pair(root, int(i), renumber)
        sh.append(dome_shift_mm(a, b))
    sh = np.array(sh)
    return {
        "n": int(len(sh)),
        "median_mm": float(np.median(sh)),
        "mean_mm": float(sh.mean()),
        "p25_mm": float(np.percentile(sh, 25)),
        "p75_mm": float(np.percentile(sh, 75)),
    }


def radial_spectrum(img: np.ndarray, n_bins: int = 16) -> np.ndarray:
    f = np.fft.fftshift(np.fft.fft2(img.astype(np.float64)))
    mag = np.abs(f)
    cy = cx = mag.shape[0] // 2
    yy, xx = np.ogrid[: mag.shape[0], : mag.shape[1]]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    bins = np.linspace(0, r.max() + 1e-6, n_bins + 1)
    out = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_bins):
        m = (r >= bins[i]) & (r < bins[i + 1])
        out[i] = float(mag[m].mean()) if m.any() else 0.0
    return np.log(out + 1e-6)


def appearance(indices: np.ndarray) -> dict:
    real_root = A1_MT
    syn_root = AMP / "a100" / "ModelTraining" / "train" / "DIR_C08"
    means_r, means_s, stds_r, stds_s = [], [], [], []
    grad_r, grad_s = [], []
    spec_r, spec_s = [], []
    wass_raw, wass_norm = [], []
    for i in indices:
        r1, _ = load_pair(real_root, int(i), False)
        s1, _ = load_pair(syn_root, int(i), True)
        means_r.append(float(r1.mean()))
        means_s.append(float(s1.mean()))
        stds_r.append(float(r1.std()))
        stds_s.append(float(s1.std()))
        grad_r.append(float(np.mean(np.abs(np.gradient(r1.astype(np.float64), axis=0)))))
        grad_s.append(float(np.mean(np.abs(np.gradient(s1.astype(np.float64), axis=0)))))
        spec_r.append(radial_spectrum(r1))
        spec_s.append(radial_spectrum(s1))

        def _norm(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-8)

        wass_raw.append(_wasserstein(r1, s1))
        wass_norm.append(_wasserstein(_norm(r1), _norm(s1)))
    spec_r = np.mean(spec_r, axis=0)
    spec_s = np.mean(spec_s, axis=0)
    return {
        "n_views": int(len(indices)),
        "real_mean": float(np.mean(means_r)),
        "synth_a100_mean": float(np.mean(means_s)),
        "real_std": float(np.mean(stds_r)),
        "synth_a100_std": float(np.mean(stds_s)),
        "real_grad_energy": float(np.mean(grad_r)),
        "synth_a100_grad_energy": float(np.mean(grad_s)),
        "spectrum_l1": float(np.mean(np.abs(spec_r - spec_s))),
        "wasserstein_raw": float(np.median(wass_raw)),
        "wasserstein_after_per_image_minmax": float(np.median(wass_norm)),
    }


def _wasserstein(a: np.ndarray, b: np.ndarray) -> float:
    qa = np.quantile(a, np.linspace(0, 1, 256))
    qb = np.quantile(b, np.linspace(0, 1, 256))
    return float(np.mean(np.abs(qa - qb)))


def fit_global_map(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One curve: real C08 intensities → pooled synth-training intensities."""
    rng = np.random.default_rng(0)
    real_s, syn_s = [], []
    for i in indices[::2]:
        r1, r6 = load_pair(A1_MT, int(i), False)
        real_s.append(r1.ravel()[::16])
        real_s.append(r6.ravel()[::16])
        for tag, _ in AMPS:
            root = AMP / tag / "ModelTraining" / "train" / "DIR_C08"
            s1, s6 = load_pair(root, int(i), True)
            syn_s.append(s1.ravel()[::16])
            syn_s.append(s6.ravel()[::16])
    real_all = np.concatenate(real_s)
    syn_all = np.concatenate(syn_s)
    real = rng.choice(real_all, size=min(200_000, real_all.size), replace=False)
    syn = rng.choice(syn_all, size=min(200_000, syn_all.size), replace=False)
    qs = np.linspace(0, 1, 2048)
    return np.quantile(real, qs), np.quantile(syn, qs)


def lung_mask(mt: Path) -> np.ndarray:
    m = np.load(mt / "Masks" / "Mask_Lung_mha.npy")
    m = np.squeeze(m)
    if m.shape != (128, 128, 128):
        raise SystemExit(f"unexpected lung mask shape {m.shape}")
    return m > 0.5


def p95_of(dvf_zyx3: np.ndarray, mask: np.ndarray) -> float:
    mag = np.linalg.norm(dvf_zyx3, axis=-1)
    return float(np.percentile(mag[mask], 95))


def label_p95(mt: Path, mask: np.ndarray) -> float:
    dvf = np.load(mt / "DVFs" / "DVF_01_mha.npy")
    # stored (H,W,D,3); mask from SimpleITK is (z,y,x). Use the same axis order as inference.
    from eval_a1_tre import npy_hwd_to_zyx

    return p95_of(npy_hwd_to_zyx(dvf.astype(np.float64)), mask)


def save_panel(indices: np.ndarray) -> None:
    i = int(indices[0])
    r1, r6 = load_pair(A1_MT, i, False)
    s1, s6 = load_pair(AMP / "a100" / "ModelTraining" / "train" / "DIR_C08", i, True)
    t1, _ = load_pair(AMP / "a200" / "ModelTraining" / "train" / "DIR_C08", i, True)
    cols = [
        (r6, "real phase 06"),
        (r1, "real phase 01"),
        (np.abs(r1 - r6), "|real 01−06|"),
        (s1, "synth ×1.0 phase 01"),
        (np.abs(s1 - s6), "|synth ×1.0 01−06|"),
        (np.abs(t1 - s6), "|synth ×2.0 01−06|"),
    ]
    fig, axes = plt.subplots(1, 6, figsize=(14, 3.2))
    for ax, (im, title) in zip(axes, cols):
        ax.imshow(im, cmap="gray")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "c08_proj_phase_diff.png", dpi=120)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    OUT.mkdir(parents=True, exist_ok=True)
    indices = shared_ap_indices()
    print(f"[probe] AP/PA views shared with synth: {len(indices)}", flush=True)

    mae = {"real": phase_mae(A1_MT, indices, False)}
    shifts = {"real": shift_stats(A1_MT, indices, False)}
    for tag, amp in AMPS:
        root = AMP / tag / "ModelTraining" / "train" / "DIR_C08"
        mae[tag] = phase_mae(root, indices, True)
        shifts[tag] = shift_stats(root, indices, True)
        print(
            f"[probe] {tag} phase-diff MAE {mae[tag]:.2f} | dome shift median {shifts[tag]['median_mm']:+.2f} mm",
            flush=True,
        )
    print(
        f"[probe] real phase-diff MAE {mae['real']:.2f} | dome shift median {shifts['real']['median_mm']:+.2f} mm",
        flush=True,
    )
    # amplitude whose synth MAE is closest to real
    closest = min(AMPS, key=lambda ta: abs(mae[ta[0]] - mae["real"]))
    app = appearance(indices)
    print(
        f"[probe] appearance wasserstein raw {app['wasserstein_raw']:.2f} | "
        f"after per-image minmax {app['wasserstein_after_per_image_minmax']:.3f}",
        flush=True,
    )
    save_panel(indices)

    src_vals, ref_vals = fit_global_map(indices)
    # how much does the global curve change an image after the network's own min-max?
    r1, _ = load_pair(A1_MT, int(indices[0]), False)
    mapped = np.interp(r1, src_vals, ref_vals).astype(np.float32)

    def _n(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    map_effect = float(np.mean(np.abs(_n(r1) - _n(mapped))))
    print(f"[probe] global map changes per-image-normalized view by mean |Δ| {map_effect:.4f}", flush=True)

    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    mask = lung_mask(A1_MT)
    print(f"[probe] lung frac {float(mask.mean()):.3f} device {device}", flush=True)

    pred = {}
    labels = {}
    # synth depths and unmatched real: existing folders
    # Synth folders are already every 4th angle (170 views). Real has all 680.
    # stride 4 on real hits the same angles as stride 1 on synth.
    jobs = [("real", A1_MT, args.stride)] + [
        (tag, AMP / tag / "ModelTraining" / "train" / "DIR_C08", 1) for tag, _ in AMPS
    ]
    for name, mt, stride in jobs:
        dvf, n = infer_voxelmap_dvf_phase01(CKPT, mt, device, stride=stride)
        # mask axis: inference returns (z,y,x,3). If lung frac looks empty, try a transpose.
        if mask.shape != dvf.shape[:3]:
            raise SystemExit(f"mask {mask.shape} vs dvf {dvf.shape}")
        pred[name] = p95_of(dvf, mask)
        labels[name] = label_p95(mt, mask)
        print(f"[probe] {name:5s} views {n:3d}  label p95 {labels[name]:.2f}  pred p95 {pred[name]:.2f}", flush=True)

    # matched real: on the fly, by writing nothing — monkeypatch load via a temp tree of symlinks
    # plus replaced phase-01/06 arrays is heavy. Apply the map inside a copied infer instead.
    pred["real_matched"] = p95_matched(CKPT, device, mask, src_vals, ref_vals, args.stride)
    print(f"[probe] real_matched pred p95 {pred['real_matched']:.2f}", flush=True)

    out = {
        "ckpt": str(CKPT),
        "note": "Diagnostic. Real = A1 DRRs of DIR phases. No Elastix, no landmarks.",
        "mm_per_detector_px_at_isocenter": MM_PER_PX,
        "phase_diff_mae_median": mae,
        "closest_synth_amplitude": {"tag": closest[0], "amplitude": closest[1], "mae": mae[closest[0]]},
        "dome_shift_mm_ap": shifts,
        "appearance_vs_a100": app,
        "global_map_mean_abs_change_after_minmax": map_effect,
        "label_p95": labels,
        "predicted_p95": pred,
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"[probe] wrote {OUT / 'results.json'}", flush=True)
    return 0


def p95_matched(ckpt, device, mask, src_vals, ref_vals, stride: int) -> float:
    """Same forward as infer_voxelmap, with one global intensity curve on real views."""
    sys.path.insert(0, "/home/abhishek/Documents/VoxelMap_Clinical")
    import torch
    from ml.utilities import networksFiLM

    def _normalize(x: np.ndarray) -> np.ndarray:
        lo, hi = float(x.min()), float(x.max())
        if hi - lo < 1e-8:
            return np.zeros_like(x, dtype=np.float32)
        return ((x - lo) / (hi - lo)).astype(np.float32)

    model = networksFiLM.Model.load(str(ckpt), device).to(device).eval()
    src_vol = _normalize(np.load(A1_MT / "SourceVolumes" / "sub_CT_06_mha.npy").squeeze())
    src_vol_t = torch.from_numpy(src_vol[None, None]).to(device)
    angles = np.loadtxt(A1_MT / "Angles.csv", dtype=np.float64).ravel()
    tgt_files = sorted((A1_MT / "TargetProjections").glob("01_Proj_*_bin.npy"))[:: max(1, stride)]
    flows = []
    with torch.no_grad():
        for tgt_file in tgt_files:
            proj_num = int(tgt_file.name.split("_")[2])
            src_file = A1_MT / "SourceProjections" / f"06_Proj_{proj_num:03d}_bin.npy"
            if not src_file.is_file():
                continue
            src = np.interp(np.load(src_file), src_vals, ref_vals).astype(np.float32)
            tgt = np.interp(np.load(tgt_file), src_vals, ref_vals).astype(np.float32)
            angle_val = float(angles[proj_num - 1]) if proj_num - 1 < len(angles) else 0.0
            src_p = torch.from_numpy(_normalize(src)[None, None]).to(device)
            tgt_p = torch.from_numpy(_normalize(tgt)[None, None]).to(device)
            _ = angle_val
            _, pred_flow = model(src_p, tgt_p, src_vol_t)
            flows.append(pred_flow[0].detach().cpu().numpy())
    if not flows:
        raise SystemExit("no matched views")
    from eval_a1_tre import npy_hwd_to_zyx

    mean_flow = np.mean(np.stack(flows, axis=0), axis=0)
    dvf = npy_hwd_to_zyx(np.moveaxis(mean_flow, 0, -1).astype(np.float64))
    print(f"[probe] real_matched views {len(flows)}", flush=True)
    return p95_of(dvf, mask)


if __name__ == "__main__":
    raise SystemExit(main())
