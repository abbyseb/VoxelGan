#!/usr/bin/env python3
"""TCIA-lite run X1: CT + ONE AP X-ray (DRR of the target), LNCC loss. NEW COPY of
run_A2_full160_aug_unsup_lncc/scripts/train_unsup_lncc.py (original untouched). Plan: DIR EXPERIMENTS/docs/PLAN_xray_conditioning.md.

Changes vs LNCC-only (one idea: give the network the target X-ray):
  * input = concat[ ref CT , backproject( standardise( DRR(tgt, 0 deg) ) ) ]  -> 2 channels (encoder, enc1 only).
    DRR/back-projection: xray/drr_torch.py (cone-beam, RTK geometry, checked vs RTK Joseph). Made on the fly from the
    (geometrically augmented) target CT, no grad. Standardised per image (zero mean, unit std) so the scale of real
    X-rays does not matter.
  * trained FROM SCRATCH, same recipe as LNCC-only: 40 epochs, cosine lr 1e-4 -> 1e-6, seed 20260929, same aug.
    The LNCC-only run IS the matched no-X-ray baseline.
  * LNCC computed with a separable box filter (3 x 1-D passes, 5 maps in one grouped conv): same loss to 8 digits,
    35x faster than the 9^3 conv3d (4.6 vs 162 ms fwd+bwd at 160^3). Pure speed-up.
Loss unchanged: 1 - lung LNCC_9(tgt, warp(ref_geo, pred)). No labels, no regulariser.

--zero-xray: optional sanity run with the X-ray channel all zeros -> run_A2_full160_aug_xray1_ctrl (not queued).

Logged each epoch: val LNCC (real X-ray), val LNCC with a SHUFFLED X-ray (another val pair's; every 5th pair),
val DVF MAE vs Elastix, fold %, std log|det J|.

GATE: consistency check + LNCC gate (as before) + (c) fast LNCC == 9^3-conv LNCC on a real pair (rel diff < 1e-5)
      + (d) the X-ray input weights receive a non-zero gradient.

  cd PopulationStudy/ClinicalExperiments/Grid160/TCIA_lite/run_A2_full160_aug_xray1
  python -u scripts/train_xray1.py --gpu 1 --check-only
  python -u scripts/train_xray1.py --gpu 1                 # X1 (from scratch, 40 epochs)
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
from torch.utils.data import DataLoader

RUN = Path(__file__).resolve().parents[1]
LITE = RUN.parent
E1 = LITE.parent / "Experiment1"
T35 = LITE.parent / "TCIA3.5"
FILENAME = "crb_dec_tcia_lite_run_a2_full160_aug_xray1"
LABEL = "TCIA-lite X1 · 160³ · CT + 1 AP DRR (back-projected) · loss = local NCC only · aug · cosine lr · from scratch"
XRAY_ANGLES = [0.0]
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
from networks.generator_crb_dec import ResidualBlock, UNetCRBDecoder  # noqa: E402
sys.path.insert(0, str(LITE / "xray"))
import drr_torch as dt  # noqa: E402
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
class XrayUNet(UNetCRBDecoder):
    """UNetCRBDecoder whose first block reads [CT, back-projected X-ray(s)]."""

    def __init__(self, n_xray: int = 1, **kw):
        super().__init__(**kw)
        self.n_xray = n_xray
        self.enc1 = ResidualBlock(1 + n_xray, 16)

    def forward(self, reference_ct, ref_phase, target_phase, xray):
        return super().forward(torch.cat([reference_ct, xray], 1), ref_phase, target_phase)


def xray_channel(tgt: torch.Tensor, angles=XRAY_ANGLES) -> torch.Tensor:
    """target CT (B,1,D,H,W) -> back-projected, per-image standardised DRR(s) (B,len(angles),D,H,W). No grad."""
    with torch.no_grad():
        p = dt.drr(tgt, angles)
        p = (p - p.mean((2, 3), keepdim=True)) / p.std((2, 3), keepdim=True).clamp_min(1e-6)
        return dt.backproject(p, angles, tgt.shape[2:])


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
        RUN = RUN.parent / "run_A2_full160_aug_xray1_ctrl"
        FILENAME += "_ctrl"
        CKPT_DIR, WEIGHTS_DIR, PLOTS_DIR = (RUN / "DecoderCRB" / x for x in ("checkpoints", "weights", "plots"))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if RUN.name not in ("run_A2_full160_aug_xray1", "run_A2_full160_aug_xray1_ctrl") or "TCIA_lite" not in str(CKPT_DIR):
        raise SystemExit(f"refuse writing outside TCIA_lite/run_A2_full160_aug_xray1[_ctrl]: {CKPT_DIR}")

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
    xm = XrayUNet(n_xray=len(XRAY_ANGLES), im_size=im_size, n_phases=n_phases).to(device).train()
    xr0 = xray_channel(t0_)
    lncc_term(r0, t0_, xm(r0, rp0, tp0, xr0), m0).backward()
    gnew = float(xm.enc1.conv1.weight.grad[:, 1:].norm())
    hg.update({"c_fast_lncc": lf, "c_conv_lncc": lc, "c_rel_diff": abs(lf - lc) / max(abs(lc), 1e-12),
               "d_xray_weight_grad_norm": gnew, "xray_channel_std": float(xr0.std())})
    (RUN / "logs" / "unsup_gate.json").write_text(json.dumps(hg, indent=2) + "\n")
    print(f"[{FILENAME}] fast LNCC {lf:.7f} vs conv {lc:.7f} | X-ray weight grad {gnew:.3e}", flush=True)
    if not (hg["c_rel_diff"] < 1e-5 and gnew > 0):
        raise SystemExit("X-RAY GATE FAILED - not training")
    del xm
    torch.cuda.empty_cache()
    if args.check_only:
        return 0

    g_dl = torch.Generator().manual_seed(args.seed)
    g_aug = torch.Generator().manual_seed(args.seed + 1)
    dl_kw = dict(batch_size=1, num_workers=4, pin_memory=device.type == "cuda",
                 persistent_workers=True, prefetch_factor=2)
    trainloader = DataLoader(trainset, shuffle=True, generator=g_dl, **dl_kw)
    valloader = DataLoader(valset, shuffle=False, **dl_kw)

    generator = XrayUNet(n_xray=len(XRAY_ANGLES), im_size=im_size, n_phases=n_phases).to(device)
    xin = (lambda t: torch.zeros(t.shape[0], len(XRAY_ANGLES), *t.shape[2:], device=t.device)) if args.zero_xray \
        else xray_channel
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
            loss = lncc_term(ref_geo, tgt, generator(ref, rp, tp, xin(tgt)), mask)
            loss.backward()
            optimizer_g.step()
            sched.step()
            tr += float(loss.item())
            n += 1

        generator.eval()
        va, va_mae, fold, lstd, nv = 0.0, 0.0, 0.0, 0.0, 0
        va_sh, nsh, prev_x = 0.0, 0, None
        with torch.no_grad():
            for data in valloader:
                ref, tgt, mask, dvf = (data[k].to(device) for k in ("reference_ct", "target_ct", "lung_mask", "target_dvf"))
                rp_, tp_ = data["ref_phase"].to(device), data["target_phase"].to(device)
                x_ = xin(tgt)
                pred = generator(ref, rp_, tp_, x_)
                va += float(lncc_term(ref, tgt, pred, mask))
                if nv % 5 == 0 and prev_x is not None:
                    va_sh += float(lncc_term(ref, tgt, generator(ref, rp_, tp_, prev_x), mask)); nsh += 1
                if nv % 5 == 4:
                    prev_x = x_
                va_mae += float(mae_loss.loss(dvf, pred, mask).item())
                f, s_ = jac_stats(pred, mask)
                fold += f; lstd += s_
                nv += 1
        k_ = max(nv, 1)
        hist.append({"epoch": epoch, "train_lncc_loss": tr / max(n, 1), "val_lncc_loss": va / k_,
                     "val_lncc_loss_shuffled_xray": va_sh / max(nsh, 1),
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
        print(f"   diag: val LNCC shuffled X-ray {h['val_lncc_loss_shuffled_xray']:.6f} | "
              f"val DVF MAE vs Elastix {h['val_dvf_mae_vs_elastix']:.4f} | fold {h['val_fold_pct']:.3f} % "
              f"| std log|detJ| {h['val_logjac_std']:.4f}", flush=True)

        xs = np.arange(1, epoch + 1)
        fig, ax = plt.subplots()
        ax.plot(xs, train_losses, "b-o", markersize=3, label="Train LNCC loss (augmented)")
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
                           "xray_angles": XRAY_ANGLES, "init": "scratch", "n_xray": len(XRAY_ANGLES),
                           "lncc_win": LNCC_WIN, "unsup_gate": hg, "history": hist, "im_size": im_size, "aug": AUG_CFG,
                           "manifest": "data_allscans/manifest.json", "consistency_check": chk}}
        torch.save(payload, str(CKPT_DIR / f"epoch_{epoch:03d}.pt"))
        torch.save(payload, str(CKPT_DIR / "latest.pt"))

    print(f"finished best_epoch={best_epoch} best_val_lncc_loss={min_val:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
