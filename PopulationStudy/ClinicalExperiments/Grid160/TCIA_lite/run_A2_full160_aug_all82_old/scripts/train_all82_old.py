#!/usr/bin/env python3
"""OLD-LOSS ALL-82. Copy of run_A2_full160_aug/scripts/train_aug.py (old loss: lung-masked L1 only) with: folder
run_A2_full160_aug_all82_old, file names, seed 20261002 (= hybrid all-82 run), and ONE data change: train on all 82 scans
(data_allscans train + val pairs, 8200). "val MAE" not held out. Gate fix as in the hybrid all-82 run: skip same-phase pairs.
Question: does adding the 17 scans also hurt the OLD loss? Compare with full160-aug on 65 scans (4.47, seed 20260929; lite
seeds differ by ~0.01) and hybrid all-82 (4.50). DIR-Lab + POPI, ep 36-40.

Original header: TCIA-lite Experiments, run A′-full160-aug. A′-full160 + variety + cosine learning rate.

Same data, network (UNetCRBDecoder 1.07M) and loss (lung-masked L1) as A′-full160.
Two changes (RUN_SPEC_A2_FULL160_AUG.md):
  1. Training-only augmentation, done on the GPU per pair:
       LR flip (p .5, negate DVF x), shift ±8 vox (p .5), zoom 0.92-1.08 (p .3, DVF x s),
       brightness/contrast/gamma (p .5, CT only), Gaussian noise sigma .01 (p .3, CT only).
     Validation and DIR scoring see NO augmentation.
  2. Cosine learning rate 1e-4 -> 1e-6 over exactly 40 epochs (per step). No extension rule.

Before training, a consistency check warps the (augmented) reference CT with the
(augmented) DVF and compares it with the identically augmented target CT. If flip or
zoom makes the match clearly worse than no augmentation, training does NOT start.

  cd PopulationStudy/ClinicalExperiments/Grid160/TCIA_lite/run_A2_full160_aug
  python -u scripts/train_aug.py --gpu 1            # check + train
  python -u scripts/train_aug.py --gpu 1 --check-only
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
FILENAME = "crb_dec_mae_tcia_lite_run_a2_full160_aug_all82_old"
LABEL = "TCIA-lite Experiments run A′-full160-aug · 160³ · MAE · aug · cosine lr · all scans · full volume"
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
from networks.generator_crb_dec import UNetCRBDecoder  # noqa: E402
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
    if force is None:
        if coin(c["int_p"], "int"):
            ref = (uni(*c["a"]) * ref + uni(*c["b"])).clamp(0, 1) ** uni(*c["gamma"])
        if coin(c["noise_p"], "noise"):
            ref = ref + c["noise_sd"] * torch.randn(ref.shape, generator=g, device="cpu").to(ref.device)
    return ref, tgt, mask, dvf


# ------------------------------------------------------------------ consistency check
def consistency_check(dataset, device, n_pairs: int = 3) -> dict:
    """Mean |target - warp(ref, dvf)| in the lung, with and without each geometric aug."""
    out = {}
    def moving(i):  # all-82 list: fixed indices land on same-phase pairs -> step to the next real pair
        while dataset.pair_files[i].split("_")[1] == dataset.pair_files[i].split("_")[3]:
            i = (i + 1) % len(dataset)
        return i
    idxs = [moving(i) for i in (0, len(dataset) // 2, len(dataset) - 1)][:n_pairs]
    for name, force in (("none", {}), ("flip", {"flip": True}), ("shift", {"shift": True}),
                        ("zoom", {"zoom": True}), ("identity_no_motion", None)):
        errs = []
        for i in idxs:
            d = dataset[i]
            ref, tgt, mask, dvf = (d[k][None].to(device) for k in ("reference_ct", "target_ct", "lung_mask", "target_dvf"))
            if name == "identity_no_motion":
                dvf = torch.zeros_like(dvf)
            else:
                ref, tgt, mask, dvf = augment(ref, tgt, mask, dvf, None, force=force)
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
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if RUN.name != "run_A2_full160_aug_all82_old" or "TCIA_lite" not in str(CKPT_DIR):
        raise SystemExit(f"refuse writing outside TCIA_lite/run_A2_full160_aug_all82_old: {CKPT_DIR}")

    set_seeds(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    man = load_manifest()
    im_dir, im_size, n_phases = man["pooled_train_dir"], 160, 10
    kw = dict(im_dir=im_dir, im_size=im_size, random_crop=False, patches_per_pair=1, fov_aug=False)
    all_pairs = list(man["train_pairs"]) + list(man["val_pairs"])
    if len(all_pairs) != 8200 or len(set(all_pairs)) != 8200 or len({p.split("_")[0] for p in all_pairs}) != 82:
        raise SystemExit("all-82 set wrong")
    print(f"[{FILENAME}] train on ALL 82 scans / 8200 pairs (val scans included; val MAE not held out)", flush=True)
    trainset = FastMmapPhasePairDataset(pair_files=all_pairs, aug_seed=args.seed, **kw)
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
    if args.check_only:
        return 0

    g_dl = torch.Generator().manual_seed(args.seed)
    g_aug = torch.Generator().manual_seed(args.seed + 1)
    dl_kw = dict(batch_size=1, num_workers=4, pin_memory=device.type == "cuda",
                 persistent_workers=True, prefetch_factor=2)
    trainloader = DataLoader(trainset, shuffle=True, generator=g_dl, **dl_kw)
    valloader = DataLoader(valset, shuffle=False, **dl_kw)

    generator = UNetCRBDecoder(im_size=im_size, n_phases=n_phases).to(device)
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

    train_losses, val_losses = [], []
    min_val, best_epoch, t0 = float("inf"), 0, time.time()
    for epoch in range(1, args.epochs + 1):
        generator.train()
        tr, n = 0.0, 0
        for data in trainloader:
            ref, tgt, mask, dvf = (data[k].to(device, non_blocking=True)
                                   for k in ("reference_ct", "target_ct", "lung_mask", "target_dvf"))
            ref, _, mask, dvf = augment(ref, tgt, mask, dvf, g_aug)
            rp = data["ref_phase"].to(device)
            tp = data["target_phase"].to(device)
            optimizer_g.zero_grad(set_to_none=True)
            loss = mae_loss.loss(dvf, generator(ref, rp, tp), mask)
            loss.backward()
            optimizer_g.step()
            sched.step()
            tr += float(loss.item())
            n += 1

        generator.eval()
        va, nv = 0.0, 0
        with torch.no_grad():
            for data in valloader:
                ref, mask, dvf = (data[k].to(device) for k in ("reference_ct", "lung_mask", "target_dvf"))
                pred = generator(ref, data["ref_phase"].to(device), data["target_phase"].to(device))
                va += float(mae_loss.loss(dvf, pred, mask).item())
                nv += 1

        tr_m, va_m = tr / max(n, 1), va / max(nv, 1)
        train_losses.append(tr_m)
        val_losses.append(va_m)
        el = time.time() - t0
        mark = ""
        if va_m < min_val:
            min_val, best_epoch, mark = va_m, epoch, " *best*"
            torch.save(generator.state_dict(), str(WEIGHTS_DIR / f"{FILENAME}_generator.pth"))
        mem = f" | gpu_mem {torch.cuda.max_memory_allocated() / 1024**2:.0f} MiB" if device.type == "cuda" else ""
        print("Epoch: %d | train MAE: %.6f | val MAE: %.6f | lr %.2e | total time: %d hours %d minutes%s%s"
              % (epoch, tr_m, va_m, optimizer_g.param_groups[0]["lr"], el // 3600, (el % 3600) // 60, mem, mark),
              flush=True)

        xs = np.arange(1, epoch + 1)
        fig, ax = plt.subplots()
        ax.plot(xs, train_losses, "b-o", markersize=3, label="Train MAE (augmented)")
        ax.plot(xs, val_losses, "r-o", markersize=3, label="Val MAE")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Lung-masked MAE (iso-vox)")
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
                           "run": "A2_full160_aug_all82_old", "im_size": im_size, "aug": AUG_CFG,
                           "manifest": "data_allscans/manifest.json", "consistency_check": chk}}
        torch.save(payload, str(CKPT_DIR / f"epoch_{epoch:03d}.pt"))
        torch.save(payload, str(CKPT_DIR / "latest.pt"))

    print(f"finished best_epoch={best_epoch} best_val_mae={min_val:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
