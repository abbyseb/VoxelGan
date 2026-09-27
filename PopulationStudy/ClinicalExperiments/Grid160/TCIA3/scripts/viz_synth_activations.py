#!/usr/bin/env python3
"""Activation / Grad-CAM maps for the G160 DecoderCRB synthesizer.

Produces, for phase 06→01:
  - channel-mean |feat| at enc1, bottleneck, up1
  - Grad-CAM on up1 (score = mean |DVF|)
  - predicted ‖u‖

  python PopulationStudy/ClinicalExperiments/Grid160/TCIA3/scripts/viz_synth_activations.py \\
      --case 1 --case 8 --gpu 0
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

DIR_EXP = Path("/home/abhishek/Voxel_GAN/DIR EXPERIMENTS")
sys.path.insert(0, str(DIR_EXP / "scripts"))
from prepare_a3_dir_case import (  # noqa: E402
    dir_hu_to_mu_for_synth,
    norm_mu,
    resize_volume_zyx,
)

ROOT = Path("/home/abhishek/Voxel_GAN")
NET = ROOT / "PopulationStudy/InitialExperiments/Experiment6"
T3 = ROOT / "PopulationStudy/ClinicalExperiments/Grid160/TCIA3"
A1 = DIR_EXP / "arms/A1_oracle_dirlab/runs"
CKPT = T3 / "DecoderCRB/checkpoints/epoch_100.pt"
OUT = T3 / "DecoderCRB/plots/activations"
INFER = 160


def load_decoder(ckpt: Path, device: torch.device):
    sys.path.insert(0, str(NET))
    from networks.generator_crb_dec import UNetCRBDecoder

    g = UNetCRBDecoder(im_size=INFER, n_phases=10)
    raw = torch.load(str(ckpt), map_location=device, weights_only=False)
    state = raw.get("generator", raw.get("state_dict", raw)) if isinstance(raw, dict) else raw
    g.load_state_dict(state, strict=True)
    g.to(device).eval()
    return g


def mid_slices(vol: np.ndarray):
    z, y, x = vol.shape
    return {
        "axial": vol[z // 2],
        "coronal": vol[:, y // 2, :],
        "sagittal": vol[:, :, x // 2],
    }


def heat_on_ct(ct: np.ndarray, heat: np.ndarray) -> np.ndarray:
    if heat.shape != ct.shape:
        t = torch.from_numpy(heat[None, None].astype(np.float32))
        t = F.interpolate(t, size=ct.shape, mode="trilinear", align_corners=True)
        heat = t[0, 0].numpy()
    h = heat - heat.min()
    return h / (h.max() + 1e-8)


def overlay_panel(ct: np.ndarray, heat: np.ndarray, title: str, path: Path) -> None:
    h = heat_on_ct(ct, heat)
    ct_views = mid_slices(ct)
    h_views = mid_slices(h)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    for i, name in enumerate(("axial", "coronal", "sagittal")):
        axes[0, i].imshow(ct_views[name], cmap="gray", vmin=-800, vmax=200)
        axes[0, i].set_title(f"CT {name}")
        axes[0, i].axis("off")
        axes[1, i].imshow(ct_views[name], cmap="gray", vmin=-800, vmax=200)
        axes[1, i].imshow(h_views[name], cmap="hot", alpha=0.55, vmin=0, vmax=1)
        axes[1, i].set_title(f"map {name}")
        axes[1, i].axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def gradcam_trio(ct: np.ndarray, maps: dict[str, np.ndarray], title: str, path: Path) -> None:
    """Encoder, bottleneck, and decoder Grad-CAM on one figure."""
    layers = (
        ("Encoder (enc1)", "gradcam_enc1"),
        ("Bottleneck", "gradcam_bottleneck"),
        ("Decoder (up1)", "gradcam_up1"),
    )
    missing = [key for _lab, key in layers if key not in maps]
    if missing:
        raise RuntimeError(f"missing Grad-CAM maps: {missing}")
    planes = ("axial", "coronal", "sagittal")
    ct_views = mid_slices(ct)
    heats = {key: mid_slices(heat_on_ct(ct, maps[key])) for _lab, key in layers}
    fig, axes = plt.subplots(4, 3, figsize=(12, 14))
    for i, name in enumerate(planes):
        axes[0, i].imshow(ct_views[name], cmap="gray", vmin=-800, vmax=200)
        axes[0, i].set_title(f"CT {name}")
        axes[0, i].axis("off")
    for r, (lab, key) in enumerate(layers, start=1):
        for i, name in enumerate(planes):
            axes[r, i].imshow(ct_views[name], cmap="gray", vmin=-800, vmax=200)
            axes[r, i].imshow(heats[key][name], cmap="hot", alpha=0.55, vmin=0, vmax=1)
            axes[r, i].set_title(f"{lab} · {name}")
            axes[r, i].axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_case(g, case: int, device: torch.device, tag: str, out_root: Path) -> None:
    sid = f"DIR_C{case:02d}"
    ct_p = A1 / sid / sid / "train" / "CT_06.mha"
    if not ct_p.is_file():
        raise SystemExit(f"missing {ct_p}")

    ct06 = sitk.GetArrayFromImage(sitk.ReadImage(str(ct_p))).astype(np.float32)
    mu, _ = dir_hu_to_mu_for_synth(ct06, "default")
    x_np = norm_mu(resize_volume_zyx(mu, INFER))
    x = torch.from_numpy(x_np[None, None]).to(device)
    ref_ph = torch.tensor([5], dtype=torch.long, device=device)
    tgt_ph = torch.tensor([0], dtype=torch.long, device=device)

    acts: dict[str, torch.Tensor] = {}

    def fwd_hook(name):
        def _fn(_m, _i, o):
            o.retain_grad()
            acts[name] = o

        return _fn

    hs = [
        g.enc1.register_forward_hook(fwd_hook("enc1")),
        g.down5.register_forward_hook(fwd_hook("bottleneck")),
        g.up1.register_forward_hook(fwd_hook("up1")),
    ]
    g.zero_grad(set_to_none=True)
    dvf = g(x, ref_ph, tgt_ph)
    score = dvf.abs().mean()
    score.backward()

    maps: dict[str, np.ndarray] = {}
    for name, a in acts.items():
        maps[name] = a.detach()[0].abs().mean(0).cpu().numpy()
        if a.grad is not None:
            w = a.grad[0].mean(dim=(1, 2, 3), keepdim=True)
            cam = (w * a.detach()[0]).sum(0).clamp_min(0).cpu().numpy()
            maps[f"gradcam_{name}"] = cam
    mag = dvf.detach()[0].norm(dim=0).cpu().numpy()
    for h in hs:
        h.remove()

    ct_inf = resize_volume_zyx(ct06, INFER)
    out_dir = out_root / sid
    overlay_panel(ct_inf, maps["enc1"], f"{sid} {tag} · enc1 mean|feat|", out_dir / "enc1_meanabs.png")
    overlay_panel(
        ct_inf, maps["bottleneck"], f"{sid} {tag} · bottleneck mean|feat|", out_dir / "bottleneck_meanabs.png"
    )
    overlay_panel(ct_inf, maps["up1"], f"{sid} {tag} · up1 mean|feat|", out_dir / "up1_meanabs.png")
    if "gradcam_up1" in maps:
        overlay_panel(
            ct_inf,
            maps["gradcam_up1"],
            f"{sid} {tag} · Grad-CAM up1 (score=mean|DVF|)",
            out_dir / "gradcam_up1.png",
        )
    gradcam_trio(
        ct_inf,
        maps,
        f"{sid} {tag} · Grad-CAM enc / bottleneck / decoder (score=mean|DVF|)",
        out_dir / "gradcam_enc_bottleneck_decoder.png",
    )
    overlay_panel(ct_inf, mag, f"{sid} {tag} · predicted ‖u‖ (06→01)", out_dir / "pred_mag_u.png")
    print(f"wrote maps under {out_dir}", flush=True)
    for p in sorted(out_dir.glob("*.png")):
        print(" ", p.name, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", type=int, action="append", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ckpt", type=Path, default=CKPT)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--tag", default="TCIA3")
    args = ap.parse_args()
    cases = args.case or [1, 8]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not args.ckpt.is_file():
        raise SystemExit(f"missing {args.ckpt}")
    print(f"device {device} | tag={args.tag} | ckpt={args.ckpt.name}", flush=True)
    g = load_decoder(args.ckpt, device)
    for case in cases:
        run_case(g, case, device, args.tag, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
