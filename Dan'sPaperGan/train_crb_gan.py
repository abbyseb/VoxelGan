"""Train Sang & Ruan UNetCRB + warp-space PatchGAN (standalone Dan'sPaperGan)."""

import os
import time
import warnings
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

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# paths: data lives in parent Voxel_GAN; artifacts stay local
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
DATA_ALL = str(PARENT / 'data' / 'spare' / 'all')

lr = 1e-4
lambda_img = 0.5
lambda_smooth = 0.1
lambda_adv = 0.05

im_size = 64
n_phases = 10
batch_size = 1
epoch_num = 100
d_update_freq = 2
adv_warmup_epochs = 5
patches_per_pair_train = 16
patches_per_pair_val = 8
# leave-out phases 5 & 9 (0-indexed 4 & 8) — match parent baseline
held_out_phases = [4, 8]

filename = 'dans_crb_warp_d'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trainset = PhasePairDataset(
    im_dir=DATA_ALL,
    im_size=im_size,
    random_crop=True,
    patches_per_pair=patches_per_pair_train,
    held_out_phases=held_out_phases,
    holdout_mode='exclude',
)
valset = PhasePairDataset(
    im_dir=DATA_ALL,
    im_size=im_size,
    random_crop=False,
    patches_per_pair=patches_per_pair_val,
    held_out_phases=held_out_phases,
    holdout_mode='include',
)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

generator = UNetCRB(im_size=im_size, n_phases=n_phases)
discriminator = PatchDiscriminator(in_channels=2, base_channels=32)  # warp-D
generator.to(device)
discriminator.to(device)

dvf_loss = losses.DVFLoss()
img_loss = losses.ImageSimilarityLoss()
smooth_loss = losses.SmoothnessLoss()
gan_loss = losses.GANLoss()

optimizer_g = optim.Adam(generator.parameters(), lr=lr)
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

os.makedirs(ROOT / 'weights', exist_ok=True)
os.makedirs(ROOT / 'plots', exist_ok=True)

min_val_loss = float('inf')
train_losses_g, train_losses_d, val_losses_g = [], [], []
tic = time.time()

n_g = sum(p.numel() for p in generator.parameters())
print(
    f'[{filename}] device={device} | UNetCRB params={n_g/1e6:.2f}M | '
    f'D_mode=warp | lambda_adv={lambda_adv} | held_out={held_out_phases}'
)
print(
    f'[{filename}] train {len(trainset)} '
    f'({len(trainset) // patches_per_pair_train} pairs × {patches_per_pair_train}) | '
    f'val {len(valset)} ({len(valset) // patches_per_pair_val} pairs × {patches_per_pair_val})'
)

for epoch in range(1, epoch_num + 1):
    generator.train()
    discriminator.train()
    train_loss_g, train_loss_d = 0.0, 0.0
    use_adv = epoch > adv_warmup_epochs

    for i, data in enumerate(trainloader, 0):
        reference_ct, target_ct, lung_mask, ref_phase, target_phase, target_dvf = (
            data['reference_ct'].to(device),
            data['target_ct'].to(device),
            data['lung_mask'].to(device),
            data['ref_phase'].to(device),
            data['target_phase'].to(device),
            data['target_dvf'].to(device),
        )

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
    discriminator.eval()
    val_loss_g = 0.0
    with torch.no_grad():
        for valdata in valloader:
            reference_ct, target_ct, lung_mask, ref_phase, target_phase, target_dvf = (
                valdata['reference_ct'].to(device),
                valdata['target_ct'].to(device),
                valdata['lung_mask'].to(device),
                valdata['ref_phase'].to(device),
                valdata['target_phase'].to(device),
                valdata['target_dvf'].to(device),
            )
            fake_dvf = generator(reference_ct, ref_phase, target_phase)
            val_loss_g += (
                dvf_loss.loss(target_dvf, fake_dvf, lung_mask)
                + lambda_img * img_loss.loss(reference_ct, target_ct, fake_dvf, lung_mask)
                + lambda_smooth * smooth_loss.loss(fake_dvf)
            ).item()

    toc = time.time()
    elapsed_h = (toc - tic) / 3600.0
    hours = int(np.floor(elapsed_h))
    minutes = int((elapsed_h - hours) * 60)

    n_train = max(len(trainset), 1)
    n_val = max(len(valset), 1)
    train_losses_g.append(train_loss_g / n_train)
    train_losses_d.append(train_loss_d / n_train)
    val_losses_g.append(val_loss_g / n_val)

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

print(f'finished best_val={min_val_loss:.6f}', flush=True)
