"""LeaveOut_3_6: UNetCRB MSE, hold-out SPARE phases 3 & 6 (interp + PE)."""
import os, time, warnings
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from utilities.dataset import PhasePairDataset
from utilities.generator_crb import UNetCRB
from utilities import losses

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
DATA_ALL = str(PARENT / 'data' / 'spare' / 'all')

lr = 1e-4
im_size = 64
n_phases = 10
batch_size = 1
epoch_num = 100
patches_per_pair_train = 16
patches_per_pair_val = 8
# SPARE 1-indexed phases 3 & 6 → 0-indexed 2 & 5
held_out_phases = [2, 5]
filename = 'dans_crb_mse_lo_3_6'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trainset = PhasePairDataset(im_dir=DATA_ALL, im_size=im_size, random_crop=True,
    patches_per_pair=patches_per_pair_train, held_out_phases=held_out_phases, holdout_mode='exclude')
valset = PhasePairDataset(im_dir=DATA_ALL, im_size=im_size, random_crop=False,
    patches_per_pair=patches_per_pair_val, held_out_phases=held_out_phases, holdout_mode='include')
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

generator = UNetCRB(im_size=im_size, n_phases=n_phases).to(device)
mse_loss = losses.DVFMSELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=lr)
os.makedirs(ROOT / 'weights', exist_ok=True)
os.makedirs(ROOT / 'plots', exist_ok=True)

min_val_loss = float('inf')
train_losses, val_losses = [], []
tic = time.time()
n_g = sum(p.numel() for p in generator.parameters())
print(f'[{filename}] device={device} | params={n_g/1e6:.2f}M | held_out_1idx=[3,6] | 0idx={held_out_phases}')
print(f'[{filename}] train {len(trainset)} | val {len(valset)}')

for epoch in range(1, epoch_num + 1):
    generator.train()
    train_loss = 0.0
    for data in trainloader:
        reference_ct = data['reference_ct'].to(device)
        lung_mask = data['lung_mask'].to(device)
        ref_phase = data['ref_phase'].to(device)
        target_phase = data['target_phase'].to(device)
        target_dvf = data['target_dvf'].to(device)
        fake_dvf = generator(reference_ct, ref_phase, target_phase)
        optimizer_g.zero_grad()
        loss = mse_loss.loss(target_dvf, fake_dvf, lung_mask)
        loss.backward(); optimizer_g.step()
        train_loss += loss.item()
    generator.eval()
    val_loss = 0.0
    with torch.no_grad():
        for valdata in valloader:
            reference_ct = valdata['reference_ct'].to(device)
            lung_mask = valdata['lung_mask'].to(device)
            ref_phase = valdata['ref_phase'].to(device)
            target_phase = valdata['target_phase'].to(device)
            target_dvf = valdata['target_dvf'].to(device)
            fake_dvf = generator(reference_ct, ref_phase, target_phase)
            val_loss += mse_loss.loss(target_dvf, fake_dvf, lung_mask).item()
    toc = time.time()
    elapsed_h = (toc - tic) / 3600.0
    hours = int(np.floor(elapsed_h)); minutes = int((elapsed_h - hours) * 60)
    n_train = max(len(trainset), 1); n_val = max(len(valset), 1)
    train_losses.append(train_loss / n_train); val_losses.append(val_loss / n_val)
    print('Epoch: %d | train MSE: %.6f | val MSE: %.6f | total time: %d hours %d minutes'
          % (epoch, train_losses[-1], val_losses[-1], hours, minutes), flush=True)
    if val_losses[-1] < min_val_loss:
        torch.save(generator.state_dict(), str(ROOT / 'weights' / f'{filename}_generator.pth'))
        min_val_loss = val_losses[-1]
    plt.figure()
    plt.plot(np.arange(1, epoch + 1), train_losses, 'b', label='Train MSE')
    plt.plot(np.arange(1, epoch + 1), val_losses, 'r', label='Val MSE')
    plt.legend(); plt.xlabel('Epoch'); plt.ylabel('Lung-masked MSE'); plt.title(filename)
    plt.savefig(str(ROOT / 'plots' / f'{filename}.png')); plt.close()
print(f'finished best_val_mse={min_val_loss:.6f}', flush=True)
