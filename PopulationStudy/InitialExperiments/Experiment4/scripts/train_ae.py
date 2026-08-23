#!/usr/bin/env python3
"""Train masked CT autoencoder on LIDC 128³ packs (Experiment 4).

Saves full AE + encoder-only weights. Encoder is what Decoder-CRB loads.

  PYTHONPATH=. python scripts/train_ae.py --gpu 0
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

E4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E4))

from networks.anatomy_ae import AnatomyAutoencoder
from utilities.lidc_dataset import LidcAnatomyDataset, lidc_split
from utilities.seed_utils import load_seed_config, set_seeds


def masked_mse(pred, target, mask):
    m = mask.expand_as(target)
    err = (pred - target) ** 2 * m
    return err.sum() / m.sum().clamp_min(1.0)


def train_one(epochs, lr, gpu, seed, im_size, patches_train, patches_val):
    cfg = load_seed_config()
    if seed is None:
        seed = int(cfg["torch_seed"])
    set_seeds(seed)
    train_ids, val_ids = lidc_split()
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    trainset = LidcAnatomyDataset(
        train_ids, im_size=im_size, random_crop=True, patches_per_vol=patches_train
    )
    valset = LidcAnatomyDataset(
        val_ids, im_size=im_size, random_crop=False, patches_per_vol=patches_val
    )
    g_dl = torch.Generator()
    g_dl.manual_seed(seed)
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, generator=g_dl)
    valloader = DataLoader(valset, batch_size=1, shuffle=False)

    model = AnatomyAutoencoder().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    run_dir = E4 / "AnatomyAE"
    os.makedirs(run_dir / "weights", exist_ok=True)
    os.makedirs(run_dir / "plots", exist_ok=True)
    ckpt = run_dir / "weights" / "anatomy_ae.pth"
    enc_ckpt = run_dir / "weights" / "anatomy_encoder.pth"

    n = sum(p.numel() for p in model.parameters())
    print(
        f"[anatomy_ae] device={device} params={n / 1e6:.2f}M | "
        f"train={len(train_ids)} val={len(val_ids)} | {im_size}³ | seed={seed}",
        flush=True,
    )

    best = float("inf")
    tr_hist, va_hist = [], []
    tic = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        tr = 0.0
        for batch in trainloader:
            ct = batch["ct"].to(device)
            mask = batch["lung_mask"].to(device)
            pred = model(ct)
            opt.zero_grad()
            loss = masked_mse(pred, ct, mask)
            loss.backward()
            opt.step()
            tr += loss.item()
        model.eval()
        va = 0.0
        with torch.no_grad():
            for batch in valloader:
                ct = batch["ct"].to(device)
                mask = batch["lung_mask"].to(device)
                va += masked_mse(model(ct), ct, mask).item()
        tr /= max(len(trainset), 1)
        va /= max(len(valset), 1)
        tr_hist.append(tr)
        va_hist.append(va)
        elapsed = time.time() - tic
        print(
            f"Epoch: {epoch} | train MSE: {tr:.6f} | val MSE: {va:.6f} | "
            f"total time: {int(elapsed // 3600)} hours {int((elapsed % 3600) // 60)} minutes",
            flush=True,
        )
        if va < best:
            best = va
            torch.save(
                {"ae": model.state_dict(), "encoder": model.encoder.state_dict()},
                ckpt,
            )
            torch.save(model.encoder.state_dict(), enc_ckpt)
        plt.figure()
        xs = np.arange(1, epoch + 1)
        plt.plot(xs, tr_hist, "b-o", markersize=3, label="Train MSE")
        plt.plot(xs, va_hist, "r-o", markersize=3, label="Val MSE")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Lung-masked CT MSE")
        plt.title("anatomy_ae")
        plt.savefig(str(run_dir / "plots" / "anatomy_ae.png"))
        plt.close()
    print(f"finished best_val_mse={best:.6f} encoder={enc_ckpt}", flush=True)
    return best


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--im_size", type=int, default=64)
    ap.add_argument("--patches_train", type=int, default=16)
    ap.add_argument("--patches_val", type=int, default=4)
    args = ap.parse_args(argv)
    train_one(
        args.epochs,
        args.lr,
        args.gpu,
        args.seed,
        args.im_size,
        args.patches_train,
        args.patches_val,
    )


if __name__ == "__main__":
    main()
