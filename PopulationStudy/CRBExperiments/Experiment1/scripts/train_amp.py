"""Train Decoder-CRB + oracle amplitude (CRBExperiments Experiment 1).

  PYTHONPATH=. python DecoderCRB/train_crb_dec_amp_mse.py --gpu 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

E1 = Path(__file__).resolve().parents[1]
if str(E1) not in sys.path:
    sys.path.insert(0, str(E1))

from losses.losses import DVFMSELoss
from networks.generator_crb_dec_amp import UNetCRBDecoderAmp
from utilities.amplitude import load_amplitude
from utilities.dataset import MultiPatientPhasePairDatasetAmp
from utilities.seed_utils import load_seed_config, set_seeds

warnings.filterwarnings("ignore")


def load_manifest():
    with open(E1 / "data" / "pooled" / "manifest.json") as f:
        return json.load(f)


def train_one(epochs: int, lr: float, gpu: int, seed: int | None):
    cfg = load_seed_config()
    if seed is None:
        seed = int(cfg["torch_seed"])
    set_seeds(seed)
    amp_table = load_amplitude()

    run_dir = E1 / "DecoderCRB"
    filename = "crb_dec_amp_oracle_e1"
    man = load_manifest()
    train_dir = man["pooled_train_dir"]

    im_size = 64
    batch_size = 1
    patches_per_pair_train = 16
    patches_per_pair_val = 8
    device = (
        torch.device(f"cuda:{gpu}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)

    trainset = MultiPatientPhasePairDatasetAmp(
        im_dir=train_dir,
        pair_files=man["train_pairs"],
        im_size=im_size,
        random_crop=True,
        patches_per_pair=patches_per_pair_train,
        amplitude_table=amp_table,
    )
    valset = MultiPatientPhasePairDatasetAmp(
        im_dir=train_dir,
        pair_files=man["val_pairs"],
        im_size=im_size,
        random_crop=False,
        patches_per_pair=patches_per_pair_val,
        amplitude_table=amp_table,
    )
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0)

    g = UNetCRBDecoderAmp(im_size=im_size, n_phases=10, cond_dim=3).to(device)
    opt = optim.Adam(g.parameters(), lr=lr)
    crit = DVFMSELoss()
    n_train = sum(p.numel() for p in g.parameters() if p.requires_grad)
    print(
        f"[{filename}] device={device} | params={n_train/1e6:.2f}M trainable | "
        f"shape-MSE + oracle log(r_p) | seed={seed}",
        flush=True,
    )
    print(
        f"[{filename}] A_train_mm={amp_table['A_train_mean_mm']:.3f} | "
        f"grid={amp_table.get('grid','data_iso')} | "
        f"train {len(trainset)} val {len(valset)} | "
        f"train={cfg['train_patients']} hold-out={cfg['holdout_patients']}",
        flush=True,
    )

    train_hist, val_hist = [], []
    best_val = float("inf")
    t0 = time.time()
    weights_dir = run_dir / "weights"
    plots_dir = run_dir / "plots"
    weights_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        g.train()
        tr = []
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            ref = batch["reference_ct"].to(device)
            mask = batch["lung_mask"].to(device)
            gt = batch["target_dvf"].to(device)
            log_r = batch["log_r_p"].to(device)
            pred = g(
                ref,
                batch["ref_phase"].to(device),
                batch["target_phase"].to(device),
                log_r,
            )
            loss = crit.loss(gt, pred, mask)
            loss.backward()
            opt.step()
            tr.append(float(loss.item()))

        g.eval()
        va = []
        with torch.no_grad():
            for batch in val_loader:
                ref = batch["reference_ct"].to(device)
                mask = batch["lung_mask"].to(device)
                gt = batch["target_dvf"].to(device)
                log_r = batch["log_r_p"].to(device)
                pred = g(
                    ref,
                    batch["ref_phase"].to(device),
                    batch["target_phase"].to(device),
                    log_r,
                )
                va.append(float(crit.loss(gt, pred, mask).item()))

        tr_m, va_m = float(np.mean(tr)), float(np.mean(va))
        train_hist.append(tr_m)
        val_hist.append(va_m)
        elapsed = time.time() - t0
        h, m = int(elapsed // 3600), int((elapsed % 3600) // 60)
        print(
            f"Epoch: {epoch} | train MSE: {tr_m:.6f} | val MSE: {va_m:.6f} | "
            f"total time: {h} hours {m} minutes",
            flush=True,
        )
        if va_m < best_val:
            best_val = va_m
            torch.save(g.state_dict(), weights_dir / f"{filename}_generator.pth")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(train_hist, label="train")
        ax.plot(val_hist, label="val")
        ax.set_xlabel("epoch")
        ax.set_ylabel("shape-space lung MSE")
        ax.set_title(filename)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / f"{filename}.png", dpi=120)
        plt.close(fig)

    print(f"finished best_val_mse={best_val:.6f}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)
    train_one(args.epochs, args.lr, args.gpu, args.seed)


if __name__ == "__main__":
    main()
