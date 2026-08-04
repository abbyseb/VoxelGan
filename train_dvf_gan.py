"""Train phase-conditioned DVF GAN (generator + discriminator, one script)."""

import os
import time
import warnings

import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from utilities.dataset import PhasePairDataset
from utilities.discriminator import PatchDiscriminator
from utilities.generator import UNetFiLM
from utilities import losses

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# constants (module-level, not CLI)
# ---------------------------------------------------------------------------
lr = 1e-4
lambda_img = 0.5
lambda_smooth = 0.1
lambda_adv = 0.05

im_size = 64
n_phases = 10
int_steps = 6
batch_size = 1
epoch_num = 100
d_update_freq = 2          # 1 D step per 2 G steps
adv_warmup_epochs = 5      # supervised-only warm-up before enabling L_adv
patches_per_pair_train = 16   # dense lung-biased crops per phase-pair
patches_per_pair_val = 8      # fixed high-lung grid for stabler val

train_dir = 'data/spare/train'
val_dir = 'data/spare/val'
filename = 'spare_mc_p1_dvf_gan_phase_mlp'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
trainset = PhasePairDataset(
    im_dir=train_dir,
    im_size=im_size,
    random_crop=True,
    patches_per_pair=patches_per_pair_train,
)
valset = PhasePairDataset(
    im_dir=val_dir,
    im_size=im_size,
    random_crop=False,
    patches_per_pair=patches_per_pair_val,
)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
generator = UNetFiLM(im_size=im_size, n_phases=n_phases, int_steps=int_steps)
discriminator = PatchDiscriminator(in_channels=4)
generator.to(device)
discriminator.to(device)

# ---------------------------------------------------------------------------
# losses + optimizers
# ---------------------------------------------------------------------------
dvf_loss = losses.DVFLoss()
img_loss = losses.ImageSimilarityLoss()
smooth_loss = losses.SmoothnessLoss()
gan_loss = losses.GANLoss()

optimizer_g = optim.Adam(generator.parameters(), lr=lr)
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
os.makedirs('weights', exist_ok=True)
os.makedirs('plots', exist_ok=True)

min_val_loss = float('inf')
train_losses_g, train_losses_d, val_losses_g = [], [], []
tic = time.time()

print(
    f'Device: {device} | train {len(trainset)} '
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

        # --- discriminator step ---
        if use_adv and (i % d_update_freq == 0):
            optimizer_d.zero_grad()
            loss_d = gan_loss.discriminator_loss(
                discriminator, reference_ct, target_dvf, fake_dvf.detach()
            )
            loss_d.backward()
            optimizer_d.step()
            train_loss_d += loss_d.item()

        # --- generator step ---
        optimizer_g.zero_grad()
        loss_g = (
            dvf_loss.loss(target_dvf, fake_dvf, lung_mask)
            + lambda_img * img_loss.loss(reference_ct, target_ct, fake_dvf, lung_mask)
            + lambda_smooth * smooth_loss.loss(fake_dvf)
        )
        if use_adv:
            loss_g = loss_g + lambda_adv * gan_loss.generator_loss(
                discriminator, reference_ct, fake_dvf
            )
        loss_g.backward()
        optimizer_g.step()
        train_loss_g += loss_g.item()

    # --- validation (loss_g terms only — no adversarial) ---
    generator.eval()
    discriminator.eval()
    val_loss_g = 0.0
    with torch.no_grad():
        for j, valdata in enumerate(valloader, 0):
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
        % (epoch, train_losses_g[-1], train_losses_d[-1], val_losses_g[-1], hours, minutes)
    )

    if val_losses_g[-1] < min_val_loss:
        torch.save(generator.state_dict(), 'weights/' + filename + '_generator.pth')
        torch.save(discriminator.state_dict(), 'weights/' + filename + '_discriminator.pth')
        min_val_loss = val_losses_g[-1]

    plt.figure()
    plt.plot(np.arange(1, epoch + 1), train_losses_g, 'b', label='Train G')
    plt.plot(np.arange(1, epoch + 1), train_losses_d, 'g', label='Train D')
    plt.plot(np.arange(1, epoch + 1), val_losses_g, 'r', label='Val G')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(filename)
    plt.savefig('plots/' + filename + '.png')
    plt.close()
