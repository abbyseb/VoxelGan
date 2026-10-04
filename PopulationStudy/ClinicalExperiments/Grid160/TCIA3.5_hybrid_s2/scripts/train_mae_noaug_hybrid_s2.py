#!/usr/bin/env python3
"""SEED 2 of TCIA3.5-hybrid. Identical to TCIA3.5_hybrid/scripts/train_mae_noaug_hybrid.py except: folder
TCIA3.5_hybrid_s2, file names, default seed 20261002, write guard also refuses TCIA3.5_hybrid.
Rule (fixed 2026-10-04): mean of seeds 1+2, last-5 (ep 96-100), plain AND mirror, hybrid > 0.1 mm better than
TCIA3.5 (seeds 1+2) AND >= 7/10 cases lower.

Original header: TCIA3.5-HYBRID: NEW COPY of TCIA3.5/scripts/train_mae_noaug.py (original untouched).

ONE CHANGE, the loss (same as TCIA_lite run_A2_full160_aug_hybrid, which passed its rule: 4.174 vs 4.470):
    loss = L1_dvf + LAMBDA_IMG * |tgt - warp(ref, pred)| (lung mask dilated 2 vox) + LAMBDA_SMOOTH * mean|grad pred|^2
Weights FIXED, same as the lite run, not tuned: LAMBDA_IMG = 10, LAMBDA_SMOOTH = 0.1.
Everything else = TCIA3.5: pooled manifest (82 scans, 7380/820 pairs), 64^3 lung-biased crops x16 per pair,
100 epochs, Adam 1e-4 constant, no augmentation, seed 20260918, batch 1.
No augmentation here, so the image term uses the same ref the network sees. Crops: the image term only sees the
crop (warp uses border padding), so it may be a bit weaker near crop edges than on full volumes.
Logged "train MAE"/"val MAE" = DVF L1 only (comparable with TCIA3.5). Parts line shows all three terms.

Comparison FIXED in advance (not by validation): epoch 92 and last-5 (96-100), plain and mirror, STANDARD TRE300,
vs TCIA3.5 4.18 plain / 4.00 mirror (ep 92). Keep only if last-5 > 0.1 mm better AND >= 7/10 cases lower.

GATE before training (stops if any fails): a) lambdas 0 == DVFLoss; b) on fixed AND random crops, the Elastix label
beats zero motion on the image term; c) extra terms give a non-zero gradient; d) CT and label crops come from the
same place (crop drawn exactly once per item).

  cd PopulationStudy/ClinicalExperiments/Grid160/TCIA3.5_hybrid
  $PY -u scripts/train_mae_noaug_hybrid.py --gpu 0 --check-only
  $PY -u scripts/train_mae_noaug_hybrid.py --gpu 0 2>&1 | tee logs/train.log
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

T = Path(__file__).resolve().parents[1]
E1 = T.parent / "Experiment1"
T3 = T.parent / "TCIA3"
T35 = T.parent / "TCIA3.5"          # read only
FILENAME = "crb_dec_mae_iso_g160_tcia35_noaug_hybrid_s2"
LABEL = "G160-TCIA3.5-HYBRID decoder · MAE + 10·image + 0.1·smooth · no aug · 82 scans"
LAMBDA_IMG = 10.0
LAMBDA_SMOOTH = 0.1
CKPT_DIR = T / "DecoderCRB" / "checkpoints"
WEIGHTS_DIR = T / "DecoderCRB" / "weights"
PLOTS_DIR = T / "DecoderCRB" / "plots"


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

_loss_mod = _load("tcia35_losses", E1 / "losses" / "losses.py")
DVFLoss = _loss_mod.DVFLoss
_fd = _load("tcia35h_fast_dataset", T35 / "scripts" / "fast_dataset.py")
_BaseDS = _fd.FastMmapPhasePairDataset


class FastMmapPhasePairDataset(_BaseDS):
    """TCIA3.5's dataset unchanged; only counts crop draws so the gate can prove CT and label share one crop."""

    def _crop_params(self, meta, patch_idx):
        self._n_crop_calls = getattr(self, "_n_crop_calls", 0) + 1
        return super()._crop_params(meta, patch_idx)

    def __getitem__(self, idx):
        self._n_crop_calls = 0
        d = super().__getitem__(idx)
        if self._n_crop_calls != 1:
            raise RuntimeError(f"crop drawn {self._n_crop_calls} times - expected exactly 1")
        return d


def warp(img, dvf):
    """Pull: out(p) = img(p + dvf(p)). (B,1,Z,Y,X); dvf (B,3,Z,Y,X) channels (x,y,z) in voxels."""
    b, _, d, h, w = img.shape
    zz, yy, xx = torch.meshgrid(*(torch.arange(n, device=img.device, dtype=img.dtype) for n in (d, h, w)),
                                indexing="ij")
    grid = torch.stack([2 * (xx + dvf[:, 0]) / (w - 1) - 1, 2 * (yy + dvf[:, 1]) / (h - 1) - 1,
                        2 * (zz + dvf[:, 2]) / (d - 1) - 1], -1)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)


def dilate(mask, r=2):
    return F.max_pool3d((mask > 0.5).float(), 2 * r + 1, stride=1, padding=r)


def image_term(ref, tgt, pred, mask):
    m = dilate(mask)
    return ((tgt - warp(ref, pred)).abs() * m).sum() / m.sum().clamp_min(1.0)


def smooth_term(pred):
    dz = pred[:, :, 1:] - pred[:, :, :-1]
    dy = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dx = pred[..., 1:] - pred[..., :-1]
    return (dz.pow(2).mean() + dy.pow(2).mean() + dx.pow(2).mean()) / 3.0


def hybrid_loss(mae, dvf, pred, mask, ref, tgt, li=LAMBDA_IMG, ls=LAMBDA_SMOOTH):
    l_dvf, l_img, l_sm = mae.loss(dvf, pred, mask), image_term(ref, tgt, pred, mask), smooth_term(pred)
    return l_dvf + li * l_img + ls * l_sm, (l_dvf.item(), l_img.item(), l_sm.item())


def hybrid_gate(man, im_dir, device, n=12) -> dict:
    mae = DVFLoss()
    rng = np.random.default_rng(0)
    moving = [p for p in man["val_pairs"] if p.split("_")[1] != p.split("_")[3]]
    pick = list(rng.choice(moving, size=min(n, len(moving)), replace=False))
    out, lab, zero = {}, {}, {}
    for rc in (False, True):
        ds = FastMmapPhasePairDataset(im_dir=im_dir, pair_files=pick, im_size=64, random_crop=rc,
                                      patches_per_pair=1, fov_aug=False, aug_seed=0)
        L, Z = [], []
        for i in range(len(pick)):
            d = ds[i]
            r, t, m, v = (d[k][None].to(device) for k in ("reference_ct", "target_ct", "lung_mask", "target_dvf"))
            if m.sum() < 100:
                continue
            L.append(float(image_term(r, t, v, m)))
            Z.append(float(image_term(r, t, torch.zeros_like(v), m)))
        key = "random" if rc else "fixed"
        out[f"b_{key}_label"] = float(np.mean(L)) if L else None
        out[f"b_{key}_zero"] = float(np.mean(Z)) if Z else None
    d = ds[0]
    r, t, m, v = (d[k][None].to(device) for k in ("reference_ct", "target_ct", "lung_mask", "target_dvf"))
    pred = (v + 0.3 * torch.randn_like(v)).detach().requires_grad_(True)
    l0, _ = hybrid_loss(mae, v, pred, m, r, t, 0.0, 0.0)
    out["a_zero_lambda_diff"] = abs(float(l0.detach()) - float(mae.loss(v, pred, m).detach()))
    (LAMBDA_IMG * image_term(r, t, pred, m) + LAMBDA_SMOOTH * smooth_term(pred)).backward()
    out["c_grad_norm"] = float(pred.grad.norm())
    out["d_single_crop"] = True                      # __getitem__ raises otherwise
    ok_b = all(out[f"b_{k}_label"] is not None and out[f"b_{k}_label"] < out[f"b_{k}_zero"] for k in ("fixed", "random"))
    out["ok"] = bool(ok_b and out["a_zero_lambda_diff"] < 1e-6 and out["c_grad_norm"] > 0)
    return out

warnings.filterwarnings("ignore")


def load_manifest() -> dict:
    media = Path(
        "/media/abhishek/3CCA3CADCA3C6574/TCIA_4D-Lung/synth_g160_r3/pooled/manifest.json"
    )
    if media.is_file():
        return json.loads(media.read_text())
    p = T35 / "data" / "manifest.json"
    if p.is_file() and p.stat().st_size > 10:
        return json.loads(p.read_text())
    raise FileNotFoundError("manifest.json missing")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if T.name != "TCIA3.5_hybrid_s2":
        raise SystemExit(f"this script must live in TCIA3.5_hybrid_s2/scripts (found {T})")
    for other in (T3, T35, T.parent / "TCIA3.5_hybrid", T.parent / "TCIA3.5_push", T.parent / "TCIA3.5_push_v2"):
        od = (other / "DecoderCRB").resolve()
        if CKPT_DIR.resolve() == od or od in CKPT_DIR.resolve().parents:
            raise SystemExit(f"refuse writing into {other.name}: {CKPT_DIR}")

    set_seeds(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    man = load_manifest()
    im_dir = man["pooled_train_dir"]
    (T / "logs").mkdir(parents=True, exist_ok=True)
    hg = hybrid_gate(man, im_dir, device)
    print(f"[{FILENAME}] hybrid gate: {json.dumps(hg)}", flush=True)
    (T / "logs" / "hybrid_gate.json").write_text(json.dumps(hg, indent=1) + "\n")
    if not hg["ok"]:
        raise SystemExit("HYBRID GATE FAILED - not training")
    print(f"[{FILENAME}] hybrid gate passed | LAMBDA_IMG {LAMBDA_IMG} LAMBDA_SMOOTH {LAMBDA_SMOOTH}", flush=True)
    if args.check_only:
        return 0
    im_size = 64
    n_phases = 10
    patches_train = 16
    patches_val = 8
    batch_size = 1
    num_workers = 4

    trainset = FastMmapPhasePairDataset(
        im_dir=im_dir,
        pair_files=man["train_pairs"],
        im_size=im_size,
        random_crop=True,
        patches_per_pair=patches_train,
        fov_aug=False,
        aug_seed=args.seed,
    )
    valset = FastMmapPhasePairDataset(
        im_dir=im_dir,
        pair_files=man["val_pairs"],
        im_size=im_size,
        random_crop=False,
        patches_per_pair=patches_val,
        fov_aug=False,
    )
    g_dl = torch.Generator().manual_seed(args.seed)
    dl_kw = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    trainloader = DataLoader(trainset, shuffle=True, generator=g_dl, **dl_kw)
    valloader = DataLoader(valset, shuffle=False, **dl_kw)

    generator = UNetCRBDecoder(im_size=im_size, n_phases=n_phases).to(device)
    mae_loss = DVFLoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    (T / "logs").mkdir(exist_ok=True)

    n_g = sum(p.numel() for p in generator.parameters()) / 1e6
    n_scans = len(man.get("train_scans") or man.get("train_patients") or [])
    print(
        f"[{FILENAME}] fast mmap DVF crops | num_workers={num_workers} | "
        f"patches_train={patches_train}",
        flush=True,
    )
    print(
        f"[{FILENAME}] device={device} | UNetCRBDecoder params={n_g:.2f}M | no FOV | "
        f"{LABEL} | seed={args.seed}",
        flush=True,
    )
    print(
        f"[{FILENAME}] train {len(trainset)} | val {len(valset)} | scans={n_scans} | "
        f"pairs train/val={man.get('n_train_pairs')}/{man.get('n_val_pairs')} | "
        f"epochs 1..{args.epochs}",
        flush=True,
    )
    print(f"[{FILENAME}] checkpoints → {CKPT_DIR}  (TCIA3 untouched)", flush=True)

    train_losses: list[float] = []
    val_losses: list[float] = []
    min_val_loss = float("inf")
    t0 = time.time()
    best_path = WEIGHTS_DIR / f"{FILENAME}_generator.pth"

    for epoch in range(1, args.epochs + 1):
        generator.train()
        train_loss = 0.0
        n_batches = 0
        parts = np.zeros(3)
        for data in trainloader:
            reference_ct = data["reference_ct"].to(device, non_blocking=True)
            lung_mask = data["lung_mask"].to(device, non_blocking=True)
            ref_phase = data["ref_phase"].to(device, non_blocking=True)
            target_phase = data["target_phase"].to(device, non_blocking=True)
            target_dvf = data["target_dvf"].to(device, non_blocking=True)
            target_ct = data["target_ct"].to(device, non_blocking=True)
            optimizer_g.zero_grad(set_to_none=True)
            fake_dvf = generator(reference_ct, ref_phase, target_phase)
            loss, pt = hybrid_loss(mae_loss, target_dvf, fake_dvf, lung_mask, reference_ct, target_ct)
            loss.backward()
            optimizer_g.step()
            parts += pt
            train_loss += pt[0]          # DVF part only, comparable with TCIA3.5
            n_batches += 1

        generator.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for data in valloader:
                reference_ct = data["reference_ct"].to(device, non_blocking=True)
                lung_mask = data["lung_mask"].to(device, non_blocking=True)
                ref_phase = data["ref_phase"].to(device, non_blocking=True)
                target_phase = data["target_phase"].to(device, non_blocking=True)
                target_dvf = data["target_dvf"].to(device, non_blocking=True)
                fake_dvf = generator(reference_ct, ref_phase, target_phase)
                val_loss += float(mae_loss.loss(target_dvf, fake_dvf, lung_mask).item())
                n_val += 1

        tr_m = train_loss / max(n_batches, 1)
        va_m = val_loss / max(n_val, 1)
        train_losses.append(tr_m)
        val_losses.append(va_m)
        elapsed_s = time.time() - t0
        hours, minutes = int(elapsed_s // 3600), int((elapsed_s % 3600) // 60)
        mark = ""
        if va_m < min_val_loss:
            min_val_loss = va_m
            mark = " *best*"
            torch.save(generator.state_dict(), str(best_path))

        print(
            "Epoch: %d | train MAE: %.6f | val MAE: %.6f | total time: %d hours %d minutes%s"
            % (epoch, tr_m, va_m, hours, minutes, mark),
            flush=True,
        )
        p = parts / max(n_batches, 1)
        print(f"   parts: dvf {p[0]:.4f} | img {p[1]:.5f} (x{LAMBDA_IMG}) | smooth {p[2]:.5f} (x{LAMBDA_SMOOTH})", flush=True)

        xs = np.arange(1, epoch + 1)
        fig, ax = plt.subplots()
        ax.plot(xs, train_losses, "b-o", markersize=3, label="Train MAE")
        ax.plot(xs, val_losses, "r-o", markersize=3, label="Val MAE")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Lung-masked MAE (iso-vox)")
        ax.set_title(FILENAME)
        ax.legend()
        fig.tight_layout()
        fig.savefig(str(PLOTS_DIR / f"{FILENAME}.png"), dpi=120)
        plt.close(fig)

        if epoch % args.save_every == 0:
            payload = {
                "epoch_done": epoch,
                "generator": generator.state_dict(),
                "optimizer": optimizer_g.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "min_val_loss": min_val_loss,
                "elapsed_s": elapsed_s,
                "cfg": {"fov_aug": False, "torch_seed": args.seed, "lr": args.lr, "run": "TCIA3.5_hybrid_s2",
                        "lambda_img": LAMBDA_IMG, "lambda_smooth": LAMBDA_SMOOTH, "hybrid_gate": hg},
            }
            torch.save(payload, str(CKPT_DIR / f"epoch_{epoch:03d}.pt"))
            torch.save(payload, str(CKPT_DIR / "latest.pt"))

    print(f"finished best_val_mae={min_val_loss:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
