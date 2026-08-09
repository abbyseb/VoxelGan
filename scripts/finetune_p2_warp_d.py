#!/usr/bin/env python3
"""Fine-tune P1 FiLM warp-D model on P2 Elastix pairs (no architecture changes)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utilities.dataset import PhasePairDataset
from utilities.discriminator import PatchDiscriminator
from utilities.generator import UNetFiLM
from utilities.svf import warp
from utilities import losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", default="P2/data/train")
    ap.add_argument("--val_dir", default="P2/data/val")
    ap.add_argument("--init_g", default="weights/spare_mc_p1_scenario_warp_d_generator.pth")
    ap.add_argument("--init_d", default="weights/spare_mc_p1_scenario_warp_d_discriminator.pth")
    ap.add_argument("--out_prefix", default="spare_mc_p2_warp_d_finetune")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr_g", type=float, default=3e-5)
    ap.add_argument("--lr_d", type=float, default=1e-5)
    ap.add_argument("--adv_warmup", type=int, default=2)
    ap.add_argument("--im_size", type=int, default=64)
    ap.add_argument("--patches_train", type=int, default=16)
    ap.add_argument("--patches_val", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainset = PhasePairDataset(
        im_dir=args.train_dir,
        im_size=args.im_size,
        random_crop=True,
        patches_per_pair=args.patches_train,
    )
    valset = PhasePairDataset(
        im_dir=args.val_dir,
        im_size=args.im_size,
        random_crop=False,
        patches_per_pair=args.patches_val,
    )
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True)
    valloader = DataLoader(valset, batch_size=1, shuffle=False)

    generator = UNetFiLM(im_size=args.im_size, n_phases=10, int_steps=6).to(device)
    discriminator = PatchDiscriminator(in_channels=2, base_channels=32).to(device)

    generator.load_state_dict(torch.load(args.init_g, map_location=device))
    discriminator.load_state_dict(torch.load(args.init_d, map_location=device))

    dvf_loss = losses.DVFLoss()
    img_loss = losses.ImageSimilarityLoss()
    smooth_loss = losses.SmoothnessLoss()
    gan_loss = losses.GANLoss()

    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr_g)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr_d)

    Path("weights").mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    tic = time.time()

    print(
        f"[{args.out_prefix}] device={device} train={len(trainset)} val={len(valset)} "
        f"lr_g={args.lr_g} lr_d={args.lr_d}"
    )

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        g_sum, d_sum = 0.0, 0.0
        use_adv = epoch > args.adv_warmup

        for i, data in enumerate(trainloader):
            reference_ct = data["reference_ct"].to(device)
            target_ct = data["target_ct"].to(device)
            lung_mask = data["lung_mask"].to(device)
            ref_phase = data["ref_phase"].to(device)
            target_phase = data["target_phase"].to(device)
            target_dvf = data["target_dvf"].to(device)

            fake_dvf = generator(reference_ct, ref_phase, target_phase)

            if use_adv and (i % 2 == 0):
                optimizer_d.zero_grad()
                real_a = warp(reference_ct, target_dvf)
                fake_a = warp(reference_ct, fake_dvf.detach())
                loss_d = gan_loss.discriminator_loss(discriminator, real_a, target_ct, fake_a, target_ct)
                loss_d.backward()
                optimizer_d.step()
                d_sum += float(loss_d.item())

            optimizer_g.zero_grad()
            loss_g = (
                dvf_loss.loss(target_dvf, fake_dvf, lung_mask)
                + 0.5 * img_loss.loss(reference_ct, target_ct, fake_dvf, lung_mask)
                + 0.1 * smooth_loss.loss(fake_dvf)
            )
            if use_adv:
                fake_a = warp(reference_ct, fake_dvf)
                loss_g = loss_g + 0.05 * gan_loss.generator_loss(discriminator, fake_a, target_ct)
            loss_g.backward()
            optimizer_g.step()
            g_sum += float(loss_g.item())

        generator.eval()
        val_g = 0.0
        with torch.no_grad():
            for data in valloader:
                reference_ct = data["reference_ct"].to(device)
                target_ct = data["target_ct"].to(device)
                lung_mask = data["lung_mask"].to(device)
                ref_phase = data["ref_phase"].to(device)
                target_phase = data["target_phase"].to(device)
                target_dvf = data["target_dvf"].to(device)

                fake_dvf = generator(reference_ct, ref_phase, target_phase)
                val_g += float(
                    (
                        dvf_loss.loss(target_dvf, fake_dvf, lung_mask)
                        + 0.5 * img_loss.loss(reference_ct, target_ct, fake_dvf, lung_mask)
                        + 0.1 * smooth_loss.loss(fake_dvf)
                    ).item()
                )

        n_train = max(len(trainset), 1)
        n_val = max(len(valset), 1)
        g_mean = g_sum / n_train
        d_mean = d_sum / n_train
        v_mean = val_g / n_val
        elapsed_h = (time.time() - tic) / 3600.0
        print(
            f"Epoch {epoch:03d}/{args.epochs} | G={g_mean:.4f} D={d_mean:.4f} valG={v_mean:.4f} "
            f"| time={elapsed_h:.2f}h",
            flush=True,
        )

        if v_mean < best_val:
            torch.save(generator.state_dict(), f"weights/{args.out_prefix}_generator.pth")
            torch.save(discriminator.state_dict(), f"weights/{args.out_prefix}_discriminator.pth")
            best_val = v_mean

    print(f"finished best_val={best_val:.6f}", flush=True)


if __name__ == "__main__":
    main()
