"""Leave-one-phase-out LOOCV for UNetCRB warp-D (sequential folds on GPU)."""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from utilities.dataset import PhasePairDataset
from utilities.discriminator import PatchDiscriminator
from utilities.generator_crb import UNetCRB
from utilities.warp import warp
from utilities import losses

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
DATA_ALL = str(PARENT / 'data' / 'spare' / 'all')

lr = 1e-4
lambda_img = 0.5
lambda_smooth = 0.1
lambda_adv = 0.05
im_size = 64
n_phases = 10
epoch_num = 100
d_update_freq = 2
adv_warmup_epochs = 5
patches_per_pair_train = 16
patches_per_pair_val = 8


def run_fold(phase0: int) -> float:
    phase1 = phase0 + 1
    filename = f'dans_crb_loocv_warp_d_hold{phase1:02d}'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    held = [phase0]

    trainset = PhasePairDataset(
        im_dir=DATA_ALL,
        im_size=im_size,
        random_crop=True,
        patches_per_pair=patches_per_pair_train,
        held_out_phases=held,
        holdout_mode='exclude',
    )
    valset = PhasePairDataset(
        im_dir=DATA_ALL,
        im_size=im_size,
        random_crop=False,
        patches_per_pair=patches_per_pair_val,
        held_out_phases=held,
        holdout_mode='include',
    )
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True)
    valloader = DataLoader(valset, batch_size=1, shuffle=False)

    generator = UNetCRB(im_size=im_size, n_phases=n_phases).to(device)
    discriminator = PatchDiscriminator(in_channels=2, base_channels=32).to(device)
    dvf_loss = losses.DVFLoss()
    img_loss = losses.ImageSimilarityLoss()
    smooth_loss = losses.SmoothnessLoss()
    gan_loss = losses.GANLoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

    min_val_loss = float('inf')
    train_losses_g, train_losses_d, val_losses_g = [], [], []
    tic = time.time()
    print(
        f'[{filename}] device={device} held={held} '
        f'train={len(trainset)} val={len(valset)}',
        flush=True,
    )

    for epoch in range(1, epoch_num + 1):
        generator.train()
        discriminator.train()
        train_loss_g, train_loss_d = 0.0, 0.0
        use_adv = epoch > adv_warmup_epochs

        for i, data in enumerate(trainloader):
            reference_ct = data['reference_ct'].to(device)
            target_ct = data['target_ct'].to(device)
            lung_mask = data['lung_mask'].to(device)
            ref_phase = data['ref_phase'].to(device)
            target_phase = data['target_phase'].to(device)
            target_dvf = data['target_dvf'].to(device)
            fake_dvf = generator(reference_ct, ref_phase, target_phase)

            if use_adv and (i % d_update_freq == 0):
                optimizer_d.zero_grad()
                real_a = warp(reference_ct, target_dvf)
                fake_a = warp(reference_ct, fake_dvf.detach())
                loss_d = gan_loss.discriminator_loss(
                    discriminator, real_a, target_ct, fake_a, target_ct
                )
                loss_d.backward()
                optimizer_d.step()
                train_loss_d += loss_d.item()

            optimizer_g.zero_grad()
            loss_g = (
                dvf_loss.loss(target_dvf, fake_dvf, lung_mask)
                + lambda_img * img_loss.loss(reference_ct, target_ct, fake_dvf, lung_mask)
                + lambda_smooth * smooth_loss.loss(fake_dvf)
            )
            if use_adv:
                fake_a = warp(reference_ct, fake_dvf)
                loss_g = loss_g + lambda_adv * gan_loss.generator_loss(
                    discriminator, fake_a, target_ct
                )
            loss_g.backward()
            optimizer_g.step()
            train_loss_g += loss_g.item()

        generator.eval()
        val_loss_g = 0.0
        with torch.no_grad():
            for valdata in valloader:
                reference_ct = valdata['reference_ct'].to(device)
                target_ct = valdata['target_ct'].to(device)
                lung_mask = valdata['lung_mask'].to(device)
                ref_phase = valdata['ref_phase'].to(device)
                target_phase = valdata['target_phase'].to(device)
                target_dvf = valdata['target_dvf'].to(device)
                fake_dvf = generator(reference_ct, ref_phase, target_phase)
                val_loss_g += (
                    dvf_loss.loss(target_dvf, fake_dvf, lung_mask)
                    + lambda_img * img_loss.loss(reference_ct, target_ct, fake_dvf, lung_mask)
                    + lambda_smooth * smooth_loss.loss(fake_dvf)
                ).item()

        n_train = max(len(trainset), 1)
        n_val = max(len(valset), 1)
        train_losses_g.append(train_loss_g / n_train)
        train_losses_d.append(train_loss_d / n_train)
        val_losses_g.append(val_loss_g / n_val)
        elapsed_h = (time.time() - tic) / 3600.0
        hours = int(np.floor(elapsed_h))
        minutes = int((elapsed_h - hours) * 60)
        print(
            'Epoch: %d | G loss: %.4f | D loss: %.4f | val G loss: %.4f | total time: %d hours %d minutes'
            % (epoch, train_losses_g[-1], train_losses_d[-1], val_losses_g[-1], hours, minutes),
            flush=True,
        )

        if val_losses_g[-1] < min_val_loss:
            torch.save(generator.state_dict(), str(ROOT / 'weights' / f'{filename}_generator.pth'))
            torch.save(
                discriminator.state_dict(),
                str(ROOT / 'weights' / f'{filename}_discriminator.pth'),
            )
            min_val_loss = val_losses_g[-1]

        plt.figure()
        plt.plot(np.arange(1, epoch + 1), train_losses_g, 'b', label='Train G')
        plt.plot(np.arange(1, epoch + 1), train_losses_d, 'g', label='Train D')
        plt.plot(np.arange(1, epoch + 1), val_losses_g, 'r', label='Val G')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(filename)
        plt.savefig(str(ROOT / 'plots' / f'{filename}.png'))
        plt.close()

    return min_val_loss


def main():
    os.makedirs(ROOT / 'weights', exist_ok=True)
    os.makedirs(ROOT / 'plots' / 'loocv_crb', exist_ok=True)
    summary_path = ROOT / 'plots' / 'loocv_crb' / 'summary.log'
    with open(summary_path, 'a') as f:
        f.write(f'\n=== LOOCV CRB start {datetime.now().isoformat()} ===\n')

    for phase0 in range(10):
        phase1 = phase0 + 1
        print(f'\n######## LOOCV CRB fold hold-out phase {phase1:02d} ########\n', flush=True)
        best = run_fold(phase0)
        line = f'fold hold{phase1:02d} best_val={best:.6f}\n'
        print(line, flush=True)
        with open(summary_path, 'a') as f:
            f.write(line)

    with open(summary_path, 'a') as f:
        f.write(f'=== LOOCV CRB done {datetime.now().isoformat()} ===\n')


if __name__ == '__main__':
    main()
