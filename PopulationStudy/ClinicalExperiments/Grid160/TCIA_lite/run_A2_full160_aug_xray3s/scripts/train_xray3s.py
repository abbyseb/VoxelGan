#!/usr/bin/env python3
"""TCIA-lite run X3S = X2R with TWO FIXED VIEWS, FRONT (0 deg) + SIDE (90 deg), each in its OWN channels. NEW COPY of
run_A2_full160_aug_xray2r/scripts/train_xray2r.py (original untouched).
Why: X3R (random angles, views averaged into one channel, 0/1/2 views) barely used the X-rays (front helped only ~0.15 mm
vs no X-ray at epoch 15; stopped). X3S keeps X2R's consistent geometry and adds the side view without mixing:
  * encoder input = [ref CT, bp(std Dt_front), bp(std Dt_side), bp(diff_front), bp(diff_side)]  (5 channels)
  * breath code: the 2-D CNN reads the two difference images as 2 channels (front, side) -> 8 numbers
  * every training step has BOTH views; each view is an independent clinic-like X-ray (80 %) or perfect DRR (20 %);
    the planning-CT X-ray Dr at each angle is a clean DRR.
Everything else as X2R (loss, data, aug, 40 epochs, cosine lr, seed). Original X2R docstring follows.

TCIA-lite run X2R: X2 trained with REALISTIC target X-rays. NEW COPY of run_A2_full160_aug_xray2/scripts/train_xray2.py.
Why: X2 (perfect DRRs) collapses on realistic X-rays (DIR-Lab 2.95 -> 9.32 mm with the clinic-like setting; scatter
breaks its difference image). X2R = same network, inputs, loss and recipe; ONE change: the target X-ray.

Target X-ray in training (per step): 20 % perfect DRR (as X2), 80 % clinic-like (xray/realistic_xray.clinic_projection:
polychromatic 120 kVp, soft/bone from HU, random mild scatter SPR 0-0.5, light Poisson + electronic noise, blur, gamma,
+-1 px shift, sometimes couch rails / cables; settings chosen against a real SPARE Varian projection).
HU of the (augmented) target = x * mu_max(scan) / 2e-5 - 1000 (raw TCIA files are mu = (HU + 1000) * 2e-5, min 0;
per-scan max in xray/tables/tcia_ct_mu_minmax.json). The planning-CT X-ray Dr stays a clean software DRR (as in clinic).

Logged: train-eval and val LNCC with CLINIC-LIKE X-rays (fixed seeds per pair), val LNCC with PERFECT X-rays (every 5th
pair), val with a shuffled clinic-like X-ray, DVF MAE vs Elastix, fold %, std log|det J|.
GATE: as X2 + (f) HU round-trip of a training target == raw file HU + (g) clinic projection finite, not constant.

  cd PopulationStudy/ClinicalExperiments/Grid160/TCIA_lite/run_A2_full160_aug_xray2r
  python -u scripts/train_xray2r.py --gpu 1 --check-only
  python -u scripts/train_xray2r.py --gpu 1
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

RUN = Path(__file__).resolve().parents[1]
LITE = RUN.parent
E1 = LITE.parent / "Experiment1"
T35 = LITE.parent / "TCIA3.5"
FILENAME = "crb_dec_tcia_lite_run_a2_full160_aug_xray3s"
LABEL = "TCIA-lite X3S (front + side, separate channels, realistic) · 160³ · CT + AP/LAT DRR + DRR difference (back-projected) + breath code (FiLM) · local NCC only · from scratch"
N_CODE = 8
XRAY_ANGLES = [0.0, 90.0]                # front, side (fixed); one set of channels per view
LNCC_WIN = 9
CKPT_DIR = RUN / "DecoderCRB" / "checkpoints"
WEIGHTS_DIR = RUN / "DecoderCRB" / "weights"
PLOTS_DIR = RUN / "DecoderCRB" / "plots"
AUG_CFG = {"flip_p": 0.5, "shift_p": 0.5, "shift_max": 8, "zoom_p": 0.3, "zoom": [0.92, 1.08],
           "int_p": 0.5, "a": [0.9, 1.1], "b": [-0.05, 0.05], "gamma": [0.8, 1.25],
           "noise_p": 0.3, "noise_sd": 0.01}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


if str(E1) not in sys.path:
    sys.path.insert(0, str(E1))
from networks.generator_crb_dec import ResidualBlock, UNetCRBDecoder, UpCRB  # noqa: E402
import torch.nn as nn  # noqa: E402
sys.path.insert(0, str(LITE / "xray"))
import drr_torch as dt  # noqa: E402
import realistic_xray as rx  # noqa: E402
MU_MAX = {k: v[1] for k, v in json.loads((LITE / "xray" / "tables" / "tcia_ct_mu_minmax.json").read_text()).items()}
P_CLEAN = 0.2
from utilities.seed_utils import set_seeds  # noqa: E402

DVFLoss = _load("tcia_lite_a2_full160_aug_losses", E1 / "losses" / "losses.py").DVFLoss
FastMmapPhasePairDataset = _load("tcia_lite_a2_full160_aug_fast_dataset",
                                 T35 / "scripts" / "fast_dataset.py").FastMmapPhasePairDataset

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------ geometry helpers
# Tensors are (B,C,Z,Y,X). DVF channels are (x, y, z) in voxels: channel 0 moves along X
# (last axis, LR), channel 2 along Z. Pull field: target(p) = reference(p + u(p)).
def warp(img: torch.Tensor, dvf: torch.Tensor) -> torch.Tensor:
    b, _, d, h, w = img.shape
    zz, yy, xx = torch.meshgrid(*(torch.arange(n, device=img.device, dtype=img.dtype) for n in (d, h, w)),
                                indexing="ij")
    gx = 2 * (xx + dvf[:, 0]) / (w - 1) - 1
    gy = 2 * (yy + dvf[:, 1]) / (h - 1) - 1
    gz = 2 * (zz + dvf[:, 2]) / (d - 1) - 1
    grid = torch.stack([gx, gy, gz], -1)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)


def shift(t: torch.Tensor, dz: int, dy: int, dx: int, fill: float) -> torch.Tensor:
    out = torch.full_like(t, fill)
    D, H, W = t.shape[-3:]

    def rng(d, n):
        return (slice(max(d, 0), n + min(d, 0)), slice(max(-d, 0), n + min(-d, 0)))

    (oz, iz), (oy, iy), (ox, ix) = rng(dz, D), rng(dy, H), rng(dx, W)
    out[..., oz, oy, ox] = t[..., iz, iy, ix]
    return out


def zoom(t: torch.Tensor, s: float, mode: str) -> torch.Tensor:
    """Magnify content by s about the centre: out(p) = in(c + (p - c)/s)."""
    b = t.shape[0]
    theta = torch.zeros(b, 3, 4, device=t.device, dtype=t.dtype)
    theta[:, 0, 0] = theta[:, 1, 1] = theta[:, 2, 2] = 1.0 / s
    grid = F.affine_grid(theta, list(t.shape), align_corners=True)
    return F.grid_sample(t, grid, mode=mode, padding_mode="border" if mode != "nearest" else "zeros",
                         align_corners=True)


def augment(ref, tgt, mask, dvf, g: torch.Generator, force: dict | None = None):
    """Same geometric transform on ref/tgt/mask/dvf; intensity on ref only. force: tests."""
    c = AUG_CFG

    def coin(p, key):
        if force is not None:
            return bool(force.get(key, False))
        return torch.rand(1, generator=g).item() < p

    def uni(lo, hi):
        return lo + (hi - lo) * torch.rand(1, generator=g).item()

    if coin(c["flip_p"], "flip"):
        ref, tgt, mask, dvf = (x.flip(-1) for x in (ref, tgt, mask, dvf))
        dvf = torch.cat([-dvf[:, :1], dvf[:, 1:]], 1)
    if coin(c["shift_p"], "shift"):
        m = c["shift_max"]
        dz, dy, dx = (int(torch.randint(-m, m + 1, (1,), generator=g)) for _ in range(3))
        if force is not None:
            dz, dy, dx = 5, -6, 7
        ref, tgt = shift(ref, dz, dy, dx, float(ref.min())), shift(tgt, dz, dy, dx, float(tgt.min()))
        mask, dvf = shift(mask, dz, dy, dx, 0.0), shift(dvf, dz, dy, dx, 0.0)
    if coin(c["zoom_p"], "zoom"):
        s = uni(*c["zoom"]) if force is None else 1.08
        ref, tgt = zoom(ref, s, "bilinear"), zoom(tgt, s, "bilinear")
        mask = zoom(mask, s, "nearest")
        dvf = zoom(dvf, s, "bilinear") * s
    ref_geo = ref
    if force is None:
        if coin(c["int_p"], "int"):
            ref = (uni(*c["a"]) * ref + uni(*c["b"])).clamp(0, 1) ** uni(*c["gamma"])
        if coin(c["noise_p"], "noise"):
            ref = ref + c["noise_sd"] * torch.randn(ref.shape, generator=g, device="cpu").to(ref.device)
    return ref, tgt, mask, dvf, ref_geo


# ------------------------------------------------------------------ unsupervised loss
def dilate(mask: torch.Tensor, r: int = 2) -> torch.Tensor:
    return F.max_pool3d((mask > 0.5).float(), 2 * r + 1, stride=1, padding=r)


def box_mean(x: torch.Tensor, k: int = LNCC_WIN) -> torch.Tensor:
    """k^3 box mean with zero padding (== conv3d with ones/k^3), as 3 separable 1-D passes, per channel."""
    c = x.shape[1]
    for ks in ((k, 1, 1), (1, k, 1), (1, 1, k)):
        x = F.conv3d(x, torch.full((c, 1, *ks), 1.0 / k, device=x.device, dtype=x.dtype),
                     padding=tuple(s // 2 for s in ks), groups=c)
    return x


def lncc_term(ref_geo, tgt, pred, mask, k: int = LNCC_WIN) -> torch.Tensor:
    w = warp(ref_geo, pred); m = dilate(mask)
    a, b, wt, ww, tt = box_mean(torch.cat([w, tgt, w * tgt, w * w, tgt * tgt], 1), k).unbind(1)
    cc = (wt - a * b) / torch.sqrt((ww - a * a).clamp_min(1e-6) * (tt - b * b).clamp_min(1e-6))
    return 1 - (cc[:, None] * m).sum() / m.sum().clamp_min(1.0)


def lncc_term_conv(ref_geo, tgt, pred, mask, k: int = LNCC_WIN) -> torch.Tensor:
    """Reference 9^3 conv3d version (LNCC-only run); used only by the gate."""
    w = warp(ref_geo, pred); m = dilate(mask); o = torch.ones(1, 1, k, k, k, device=w.device) / k ** 3
    mu = lambda x: F.conv3d(x, o, padding=k // 2)
    a, b = mu(w), mu(tgt)
    cc = (mu(w * tgt) - a * b) / torch.sqrt((mu(w * w) - a * a).clamp_min(1e-6) * (mu(tgt * tgt) - b * b).clamp_min(1e-6))
    return 1 - (cc * m).sum() / m.sum().clamp_min(1.0)


def jac_stats(pred, mask) -> tuple[float, float]:
    """% lung voxels with det(I + grad u) <= 0, and std of log|det J| in the lung. u in voxels, (x,y,z) channels."""
    u = pred[0]
    g = [torch.gradient(u[c], dim=(2, 1, 0)) for c in range(3)]      # d/dx, d/dy, d/dz of each channel
    J = torch.stack([torch.stack(r, 0) for r in g], 0)               # J[c, d] = du_c/dx_d
    J = J + torch.eye(3, device=u.device)[:, :, None, None, None]
    det = torch.linalg.det(J.permute(2, 3, 4, 0, 1))
    m = mask[0, 0] > 0.5
    d = det[m]
    return float((d <= 0).float().mean() * 100), float(torch.log(d.abs().clamp_min(1e-6)).std())


def unsup_gate(dataset, device) -> dict:
    """a) label beats zero motion on LNCC (none/flip/shift/zoom); b) LNCC gives a gradient."""
    out = {}
    for name, force in (("none", {}), ("flip", {"flip": True}), ("shift", {"shift": True}), ("zoom", {"zoom": True})):
        lab, zero = [], []
        for i in (0, len(dataset) // 2, len(dataset) - 1):
            e = dataset[i]
            r, t, m, v = (e[k][None].to(device) for k in ("reference_ct", "target_ct", "lung_mask", "target_dvf"))
            r, t, m, v, rg = augment(r, t, m, v, None, force=force)
            lab.append(float(lncc_term(rg, t, v, m)))
            zero.append(float(lncc_term(rg, t, torch.zeros_like(v), m)))
        out[f"a_{name}_label"], out[f"a_{name}_zero"] = float(np.mean(lab)), float(np.mean(zero))
    i = next(i for i in range(len(dataset)) if float(dataset[i]["target_dvf"].abs().mean()) > 0.1)  # skip self-pairs (zero DVF)
    d = dataset[i]
    out["moving_pair_index"] = i
    ref, tgt, mask, dvf = (d[k][None].to(device) for k in ("reference_ct", "target_ct", "lung_mask", "target_dvf"))
    p = (dvf + 0.3 * torch.randn_like(dvf)).detach().requires_grad_(True)
    lncc_term(ref, tgt, p, mask).backward()
    out["b_grad_norm"] = float(p.grad.norm())
    out["label_jac_fold_pct"], out["label_jac_logstd"] = jac_stats(dvf, mask)
    out["ok"] = out["b_grad_norm"] > 0 and all(out[f"a_{k}_label"] < out[f"a_{k}_zero"]
                                                for k in ("none", "flip", "shift", "zoom"))
    return out


# ------------------------------------------------------------------ X-ray conditioning
class BreathEncoder(nn.Module):
    """2-D difference image (B,1,128,128) -> n_code numbers (global summary of how much / where it moved)."""

    def __init__(self, n_code: int = N_CODE):
        super().__init__()
        ch = [len(XRAY_ANGLES), 16, 32, 32, 32]
        layers = []
        for i in range(4):
            layers += [nn.Conv2d(ch[i], ch[i + 1], 3, stride=2, padding=1), nn.ReLU(inplace=True)]
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(ch[-1], n_code)

    def forward(self, d):
        assert d.shape[1] == len(XRAY_ANGLES), "BreathEncoder reads one difference image per view"
        return self.fc(self.conv(d).mean((2, 3)))


class XrayUNet2(UNetCRBDecoder):
    """UNetCRBDecoder: encoder reads [CT, bp(X-ray), bp(difference)]; decoder conditioned on [phases, breath code]."""

    def __init__(self, n_code: int = N_CODE, **kw):
        super().__init__(**kw)
        cd = 2 + n_code
        self.enc1 = ResidualBlock(1 + 2 * len(XRAY_ANGLES), 16)
        self.up4 = UpCRB(64, 64, 64, cond_dim=cd)
        self.up3 = UpCRB(64, 32, 32, cond_dim=cd)
        self.up2 = UpCRB(32, 32, 32, cond_dim=cd)
        self.up1 = UpCRB(32, 16, 16, cond_dim=cd)
        self.breath = BreathEncoder(n_code)

    def forward(self, reference_ct, ref_phase, target_phase, vol, diff2d):
        cond = torch.cat([self._phase_vec(ref_phase, target_phase), self.breath(diff2d)], 1)
        s1 = self.enc1(torch.cat([reference_ct, vol], 1))
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        s4 = self.down4(s3)
        x = self.down5(s4)
        x = self.up4(x, s4, cond)
        x = self.up3(x, s3, cond)
        x = self.up2(x, s2, cond)
        x = self.up1(x, s1, cond)
        return self.out_conv(x)


def _std2d(p: torch.Tensor) -> torch.Tensor:
    return (p - p.mean((2, 3), keepdim=True)) / p.std((2, 3), keepdim=True).clamp_min(1e-6)


def ct_key(data, i: int = 0) -> str:
    return f"{data['patient'][i]}_CT_{int(data['target_phase'][i]) + 1:02d}"


def to_hu(x: torch.Tensor, key: str) -> torch.Tensor:
    return x * (MU_MAX[key] / 2e-5) - 1000.0


def target_projection(tgt: torch.Tensor, key: str | None, gen: torch.Generator | None, p_clean: float,
                      angles=XRAY_ANGLES) -> torch.Tensor:
    """(B,V,n,n), one per angle: each a perfect DRR with prob p_clean (or if key is None), else an independent
    clinic-like realistic X-ray of the target."""
    outs, sb = [], None
    for a in angles:
        if key is None or gen is None or float(torch.rand(1, generator=gen)) < p_clean:
            outs.append(dt.drr(tgt, [a]))
        else:
            if sb is None:
                sb = rx.materials(to_hu(tgt, key))
            outs.append(rx.clinic_projection(dt.drr(sb[0], [a]) / 10.0, dt.drr(sb[1], [a]) / 10.0, gen))
    return torch.cat(outs, 1)


def xray_inputs(ref: torch.Tensor, tgt: torch.Tensor, key: str | None = None, gen: torch.Generator | None = None,
                p_clean: float = 1.0, angles=XRAY_ANGLES):
    """-> (vol (B,2V,D,H,W) = [bp(std Dt_v) for v] + [bp(std Dt_v - std Dr_v) for v], diff2d (B,V,n,n)). Dr = clean DRR of ref."""
    with torch.no_grad():
        pt, pr = _std2d(target_projection(tgt, key, gen, p_clean, angles)), _std2d(dt.drr(ref, angles))
        d = pt - pr
        vol = torch.cat([dt.backproject(pt, angles, tgt.shape[2:]), dt.backproject(d, angles, tgt.shape[2:])], 1)
        return vol, d


def zero_inputs(ref, tgt, *a, **k):
    return _zero_inputs(ref, tgt)


def _zero_inputs(ref: torch.Tensor, tgt: torch.Tensor, angles=XRAY_ANGLES):
    b = tgt.shape[0]
    return (torch.zeros(b, 2 * len(angles), *tgt.shape[2:], device=tgt.device),
            torch.zeros(b, len(angles), dt.N_DET, dt.N_DET, device=tgt.device))


# ------------------------------------------------------------------ consistency check
def consistency_check(dataset, device, n_pairs: int = 3) -> dict:
    """Mean |target - warp(ref, dvf)| in the lung, with and without each geometric aug."""
    out = {}
    idxs = [0, len(dataset) // 2, len(dataset) - 1][:n_pairs]
    for name, force in (("none", {}), ("flip", {"flip": True}), ("shift", {"shift": True}),
                        ("zoom", {"zoom": True}), ("identity_no_motion", None)):
        errs = []
        for i in idxs:
            d = dataset[i]
            ref, tgt, mask, dvf = (d[k][None].to(device) for k in ("reference_ct", "target_ct", "lung_mask", "target_dvf"))
            if name == "identity_no_motion":
                dvf = torch.zeros_like(dvf)
            else:
                ref, tgt, mask, dvf, _ = augment(ref, tgt, mask, dvf, None, force=force)
            m = mask > 0.5
            errs.append(float((tgt - warp(ref, dvf)).abs()[m].mean()))
        out[name] = float(np.mean(errs))
    return out


def load_manifest() -> dict:
    p = LITE / "data_allscans" / "manifest.json"
    man = json.loads(p.read_text())
    if man.get("n_train_pairs") != 6500 or man.get("n_val_pairs") != 1700:
        raise SystemExit(f"unexpected manifest sizes: train {man.get('n_train_pairs')} val {man.get('n_val_pairs')}")
    return man


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-min", type=float, default=1e-6)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--zero-xray", action="store_true", help="matched control: X-ray channel all zeros")
    args = ap.parse_args()
    global RUN, CKPT_DIR, WEIGHTS_DIR, PLOTS_DIR, FILENAME
    if args.zero_xray:
        RUN = RUN.parent / "run_A2_full160_aug_xray3s_ctrl"
        FILENAME += "_ctrl"
        CKPT_DIR, WEIGHTS_DIR, PLOTS_DIR = (RUN / "DecoderCRB" / x for x in ("checkpoints", "weights", "plots"))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if RUN.name not in ("run_A2_full160_aug_xray3s", "run_A2_full160_aug_xray3s_ctrl") or "TCIA_lite" not in str(CKPT_DIR):
        raise SystemExit(f"refuse writing outside TCIA_lite/run_A2_full160_aug_xray3s[_ctrl]: {CKPT_DIR}")

    set_seeds(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    man = load_manifest()
    im_dir, im_size, n_phases = man["pooled_train_dir"], 160, 10
    kw = dict(im_dir=im_dir, im_size=im_size, random_crop=False, patches_per_pair=1, fov_aug=False)
    trainset = FastMmapPhasePairDataset(pair_files=man["train_pairs"], aug_seed=args.seed, **kw)
    valset = FastMmapPhasePairDataset(pair_files=man["val_pairs"], **kw)
    for d in (CKPT_DIR, WEIGHTS_DIR, PLOTS_DIR, RUN / "logs"):
        d.mkdir(parents=True, exist_ok=True)

    # ---- consistency check (gate) ----
    chk = consistency_check(trainset, device)
    print(f"[{FILENAME}] consistency |tgt - warp(ref,dvf)| in lung: "
          + "  ".join(f"{k} {v:.4f}" for k, v in chk.items()), flush=True)
    (RUN / "logs" / "consistency_check.json").write_text(json.dumps(chk, indent=2) + "\n")
    base = chk["none"]
    bad = [k for k in ("flip", "shift", "zoom") if chk[k] > 1.15 * base]
    if chk["none"] >= chk["identity_no_motion"]:
        raise SystemExit("check failed: labels do not improve alignment even without augmentation "
                         "(DVF channel order or sign convention differs from warp()); fix before training")
    if bad:
        raise SystemExit(f"check failed for {bad}: augmented DVF no longer matches augmented images")
    print(f"[{FILENAME}] consistency check passed", flush=True)
    hg = unsup_gate(trainset, device)
    print(f"[{FILENAME}] unsup gate: {json.dumps(hg)}", flush=True)
    (RUN / "logs" / "unsup_gate.json").write_text(json.dumps(hg, indent=2) + "\n")
    if not hg["ok"]:
        raise SystemExit("UNSUP GATE FAILED - not training")
    print(f"[{FILENAME}] unsup gate passed | LNCC_WIN {LNCC_WIN}", flush=True)
    d0 = valset[len(valset) // 2]
    r0, t0_, m0, v0 = (d0[k][None].to(device) for k in ("reference_ct", "target_ct", "lung_mask", "target_dvf"))
    rp0, tp0 = d0["ref_phase"][None].to(device), d0["target_phase"][None].to(device)
    with torch.no_grad():
        lf, lc = float(lncc_term(r0, t0_, v0, m0)), float(lncc_term_conv(r0, t0_, v0, m0))
    xm = XrayUNet2(im_size=im_size, n_phases=n_phases).to(device).train()
    v0_, d0_ = xray_inputs(r0, t0_)
    lncc_term(r0, t0_, xm(r0, rp0, tp0, v0_, d0_), m0).backward()
    nv_ = len(XRAY_ANGLES)
    gx = float(min(xm.enc1.conv1.weight.grad[:, 1 + v].norm() for v in range(nv_)))            # every X-ray channel
    gd = float(min(xm.enc1.conv1.weight.grad[:, 1 + nv_ + v].norm() for v in range(nv_)))      # every difference channel
    gb = float(sum(p.grad.norm() for p in xm.breath.parameters()))
    vs_, ds_ = xray_inputs(t0_, t0_)
    hg.update({"c_fast_lncc": lf, "c_conv_lncc": lc, "c_rel_diff": abs(lf - lc) / max(abs(lc), 1e-12),
               "d_grad_xray_ch": gx, "d_grad_diff_ch": gd, "d_grad_breath_encoder": gb,
               "e_same_ct_diff_max": float(ds_.abs().max()), "e_same_ct_diff_vol_max": float(vs_[:, len(XRAY_ANGLES):].abs().max()),
               "h_front_vs_side_channels_differ": float((v0_[:, 0] - v0_[:, 1]).abs().mean()),
               "diff2d_std_real_pair": float(d0_.std()), "n_params": sum(p.numel() for p in xm.parameters())})
    (RUN / "logs" / "unsup_gate.json").write_text(json.dumps(hg, indent=2) + "\n")
    print(f"[{FILENAME}] fast LNCC {lf:.7f} vs conv {lc:.7f} | grads X-ray {gx:.3e} diff {gd:.3e} breath {gb:.3e} | "
          f"same-CT diff max {hg['e_same_ct_diff_max']:.2e} | real-pair diff std {hg['diff2d_std_real_pair']:.3f}", flush=True)
    dtr = trainset[37]
    key = ct_key({"patient": [dtr["patient"]], "target_phase": dtr["target_phase"][None]})
    raw_hu = np.load(os.path.join(im_dir, key + ".npy")) / 2e-5 - 1000.0
    hg["f_hu_roundtrip_max_abs"] = float(np.abs(to_hu(dtr["target_ct"][0], key).numpy() - raw_hu).max())
    pc = target_projection(dtr["target_ct"][None].to(device), key, torch.Generator().manual_seed(0), 0.0)
    hg["g_clinic_proj_finite"] = bool(torch.isfinite(pc).all()); hg["g_clinic_proj_std"] = float(pc.std())
    print(f"[{FILENAME}] HU round-trip max |diff| {hg['f_hu_roundtrip_max_abs']:.3e} HU | clinic proj std {hg['g_clinic_proj_std']:.3f}", flush=True)
    (RUN / "logs" / "unsup_gate.json").write_text(json.dumps(hg, indent=2) + "\n")
    if not (hg["c_rel_diff"] < 1e-5 and min(gx, gd, gb) > 0 and hg["e_same_ct_diff_max"] == 0.0
            and hg["e_same_ct_diff_vol_max"] == 0.0 and hg["f_hu_roundtrip_max_abs"] < 0.5
            and hg["g_clinic_proj_finite"] and hg["g_clinic_proj_std"] > 0
            and hg["h_front_vs_side_channels_differ"] > 1e-3):
        raise SystemExit("X-RAY GATE FAILED - not training")
    del xm
    torch.cuda.empty_cache()
    if args.check_only:
        return 0

    g_dl = torch.Generator().manual_seed(args.seed)
    g_aug = torch.Generator().manual_seed(args.seed + 1)
    g_xr = torch.Generator().manual_seed(args.seed + 2)
    dl_kw = dict(batch_size=1, num_workers=4, pin_memory=device.type == "cuda",
                 persistent_workers=True, prefetch_factor=2)
    trainloader = DataLoader(trainset, shuffle=True, generator=g_dl, **dl_kw)
    valloader = DataLoader(valset, shuffle=False, **dl_kw)
    trevalloader = DataLoader(Subset(trainset, list(range(0, len(trainset), 19))), shuffle=False, **dl_kw)

    generator = XrayUNet2(im_size=im_size, n_phases=n_phases).to(device)
    xin = zero_inputs if args.zero_xray else xray_inputs
    mae_loss = DVFLoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr)
    total_steps = args.epochs * len(trainloader)
    sched = optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=total_steps, eta_min=args.lr_min)

    n_g = sum(p.numel() for p in generator.parameters()) / 1e6
    print(f"[{FILENAME}] device={device} | UNetCRBDecoder params={n_g:.2f}M | {LABEL} | seed={args.seed} | "
          f"lr cosine {args.lr}->{args.lr_min} over {args.epochs} epochs | physical GPU {args.gpu}", flush=True)
    print(f"[{FILENAME}] aug {AUG_CFG}", flush=True)
    print(f"[{FILENAME}] train steps/epoch {len(trainset)} | val steps/epoch {len(valset)} | full {im_size}³",
          flush=True)

    train_losses, val_losses, hist = [], [], []
    min_val, best_epoch, t0 = float("inf"), 0, time.time()
    for epoch in range(1, args.epochs + 1):
        generator.train()
        tr, n = 0.0, 0
        for data in trainloader:
            ref, tgt, mask, dvf = (data[k].to(device, non_blocking=True)
                                   for k in ("reference_ct", "target_ct", "lung_mask", "target_dvf"))
            ref, tgt, mask, dvf, ref_geo = augment(ref, tgt, mask, dvf, g_aug)
            rp = data["ref_phase"].to(device)
            tp = data["target_phase"].to(device)
            optimizer_g.zero_grad(set_to_none=True)
            loss = lncc_term(ref_geo, tgt, generator(ref, rp, tp, *xin(ref_geo, tgt, ct_key(data), g_xr, P_CLEAN)), mask)
            loss.backward()
            optimizer_g.step()
            sched.step()
            tr += float(loss.item())
            n += 1

        generator.eval()
        te, nte = 0.0, 0
        with torch.no_grad():
            for data in trevalloader:
                ref, tgt, mask = (data[k].to(device) for k in ("reference_ct", "target_ct", "lung_mask"))
                pred = generator(ref, data["ref_phase"].to(device), data["target_phase"].to(device),
                                 *xin(ref, tgt, ct_key(data), torch.Generator().manual_seed(50_000 + nte), 0.0))
                te += float(lncc_term(ref, tgt, pred, mask)); nte += 1
        va, va_mae, fold, lstd, nv = 0.0, 0.0, 0.0, 0.0, 0
        va_sh, nsh, prev_t, prev_k = 0.0, 0, None, None
        va_cl, ncl = 0.0, 0
        with torch.no_grad():
            for data in valloader:
                ref, tgt, mask, dvf = (data[k].to(device) for k in ("reference_ct", "target_ct", "lung_mask", "target_dvf"))
                rp_, tp_ = data["ref_phase"].to(device), data["target_phase"].to(device)
                k_ = ct_key(data)
                pred = generator(ref, rp_, tp_, *xin(ref, tgt, k_, torch.Generator().manual_seed(90_000 + nv), 0.0))
                if nv % 5 == 2:
                    va_cl += float(lncc_term(ref, tgt, generator(ref, rp_, tp_, *xin(ref, tgt)), mask)); ncl += 1
                va += float(lncc_term(ref, tgt, pred, mask))
                if nv % 5 == 0 and prev_t is not None:
                    va_sh += float(lncc_term(ref, tgt, generator(ref, rp_, tp_, *xin(
                        ref, prev_t, prev_k, torch.Generator().manual_seed(70_000 + nv), 0.0)), mask)); nsh += 1
                if nv % 5 == 4:
                    prev_t, prev_k = tgt, k_
                va_mae += float(mae_loss.loss(dvf, pred, mask).item())
                f, s_ = jac_stats(pred, mask)
                fold += f; lstd += s_
                nv += 1
        k_ = max(nv, 1)
        hist.append({"epoch": epoch, "train_lncc_loss": tr / max(n, 1), "train_eval_lncc_loss": te / max(nte, 1),
                     "val_lncc_loss": va / k_,
                     "val_lncc_loss_shuffled_xray": va_sh / max(nsh, 1),
                     "val_lncc_loss_perfect_xray": va_cl / max(ncl, 1),
                     "val_dvf_mae_vs_elastix": va_mae / k_, "val_fold_pct": fold / k_, "val_logjac_std": lstd / k_})
        (RUN / "logs" / "history.json").write_text(json.dumps(hist, indent=2) + "\n")

        tr_m, va_m = tr / max(n, 1), va / max(nv, 1)
        train_losses.append(tr_m)
        val_losses.append(va_m)
        el = time.time() - t0
        mark = ""
        if va_m < min_val:
            min_val, best_epoch, mark = va_m, epoch, " *best*"
            torch.save(generator.state_dict(), str(WEIGHTS_DIR / f"{FILENAME}_generator.pth"))
        mem = f" | gpu_mem {torch.cuda.max_memory_allocated() / 1024**2:.0f} MiB" if device.type == "cuda" else ""
        print("Epoch: %d | train LNCC loss: %.6f | val LNCC loss: %.6f | lr %.2e | total time: %d hours %d minutes%s%s"
              % (epoch, tr_m, va_m, optimizer_g.param_groups[0]["lr"], el // 3600, (el % 3600) // 60, mem, mark),
              flush=True)
        h = hist[-1]
        print(f"   gap: train-eval {h['train_eval_lncc_loss']:.6f} vs val {h['val_lncc_loss']:.6f} "
              f"-> gap {h['val_lncc_loss'] - h['train_eval_lncc_loss']:.6f}", flush=True)
        print(f"   diag: val LNCC shuffled X-ray {h['val_lncc_loss_shuffled_xray']:.6f} | "
              f"val DVF MAE vs Elastix {h['val_dvf_mae_vs_elastix']:.4f} | fold {h['val_fold_pct']:.3f} % "
              f"| std log|detJ| {h['val_logjac_std']:.4f}", flush=True)

        xs = np.arange(1, epoch + 1)
        fig, ax = plt.subplots()
        ax.plot(xs, train_losses, "b-o", markersize=3, label="Train LNCC loss (augmented)")
        ax.plot(xs, [h_["train_eval_lncc_loss"] for h_ in hist], "g-s", markersize=3,
                label="Train-eval LNCC loss (343 train pairs, no aug, end of epoch)")
        ax.plot(xs, val_losses, "r-o", markersize=3, label="Val LNCC loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("1 - lung LNCC")
        ax.set_title(FILENAME)
        ax.legend()
        fig.tight_layout()
        fig.savefig(str(PLOTS_DIR / f"{FILENAME}.png"), dpi=120)
        plt.close(fig)

        payload = {"epoch_done": epoch, "generator": generator.state_dict(),
                   "optimizer": optimizer_g.state_dict(), "scheduler": sched.state_dict(),
                   "train_losses": train_losses, "val_losses": val_losses, "min_val_loss": min_val,
                   "best_epoch": best_epoch, "elapsed_s": el,
                   "cfg": {"torch_seed": args.seed, "lr": args.lr, "lr_min": args.lr_min, "schedule": "cosine",
                           "run": RUN.name, "loss": "lncc_only", "zero_xray": args.zero_xray,
                           "xray_angles": XRAY_ANGLES, "init": "scratch", "n_code": N_CODE,
                           "inputs": "ct, bp(std Dt), bp(std Dt - std Dr); breath code from std Dt - std Dr",
                           "lncc_win": LNCC_WIN, "unsup_gate": hg, "history": hist, "im_size": im_size, "aug": AUG_CFG,
                           "manifest": "data_allscans/manifest.json", "consistency_check": chk}}
        torch.save(payload, str(CKPT_DIR / f"epoch_{epoch:03d}.pt"))
        torch.save(payload, str(CKPT_DIR / "latest.pt"))

    print(f"finished best_epoch={best_epoch} best_val_lncc_loss={min_val:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
