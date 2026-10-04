#!/usr/bin/env python3
"""SEED 2 of TCIA3.5 (old loss). Identical to TCIA3.5/scripts/train_mae_noaug.py except: folder TCIA3.5_s2,
file names, default seed 20261002, write guard also refuses TCIA3.5. fast_dataset.py copied unchanged.
Purpose: seed-2 baseline for the TCIA3.5-hybrid re-check (rule fixed 2026-10-04, see Diary).

Original header: TCIA3.5: same MAE decoder as TCIA3, FOV/CBCT augmentation off.

Writes only under TCIA3.5/. Does not read or write TCIA3 checkpoints.

  cd PopulationStudy/ClinicalExperiments/Grid160/TCIA3.5
  LEARN-GUI/.venv/bin/python -u scripts/train_mae_noaug.py --gpu 0
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
import torch.optim as optim
from torch.utils.data import DataLoader

T = Path(__file__).resolve().parents[1]
E1 = T.parent / "Experiment1"
T3 = T.parent / "TCIA3"
FILENAME = "crb_dec_mae_iso_g160_tcia35_noaug_s2"
LABEL = "G160-TCIA3.5 decoder · R3+µ · MAE · no FOV aug · 82 scans"
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
_fd = _load("tcia35_fast_dataset", T / "scripts" / "fast_dataset.py")
FastMmapPhasePairDataset = _fd.FastMmapPhasePairDataset

warnings.filterwarnings("ignore")


def load_manifest() -> dict:
    media = Path(
        "/media/abhishek/3CCA3CADCA3C6574/TCIA_4D-Lung/synth_g160_r3/pooled/manifest.json"
    )
    if media.is_file():
        return json.loads(media.read_text())
    p = T / "data" / "manifest.json"
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
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    t3_dec = (T3 / "DecoderCRB").resolve()
    if CKPT_DIR.resolve() == t3_dec or t3_dec in CKPT_DIR.resolve().parents:
        raise SystemExit(f"refuse writing into TCIA3: {CKPT_DIR}")
    if T.name != "TCIA3.5_s2":
        raise SystemExit(f"this script must live in TCIA3.5_s2/scripts (found {T})")

    set_seeds(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    man = load_manifest()
    im_dir = man["pooled_train_dir"]
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
        for data in trainloader:
            reference_ct = data["reference_ct"].to(device, non_blocking=True)
            lung_mask = data["lung_mask"].to(device, non_blocking=True)
            ref_phase = data["ref_phase"].to(device, non_blocking=True)
            target_phase = data["target_phase"].to(device, non_blocking=True)
            target_dvf = data["target_dvf"].to(device, non_blocking=True)
            optimizer_g.zero_grad(set_to_none=True)
            fake_dvf = generator(reference_ct, ref_phase, target_phase)
            loss = mae_loss.loss(target_dvf, fake_dvf, lung_mask)
            loss.backward()
            optimizer_g.step()
            train_loss += float(loss.item())
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
                "cfg": {"fov_aug": False, "torch_seed": args.seed, "lr": args.lr},
            }
            torch.save(payload, str(CKPT_DIR / f"epoch_{epoch:03d}.pt"))
            torch.save(payload, str(CKPT_DIR / "latest.pt"))

    print(f"finished best_val_mae={min_val_loss:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
