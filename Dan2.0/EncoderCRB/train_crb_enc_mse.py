"""Dan 2.0 Encoder-CRB: UNetCRB + lung-masked MSE only (no D).

CRB on encoder + bottleneck, plain decoder. Same leave-outs as Decoder/Both CRB.

  PYTHONPATH=.. python train_crb_enc_mse.py --held_out 5,9 --run_dir LeaveOut_5_9
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

ENC_ROOT = Path(__file__).resolve().parent
DAN20 = ENC_ROOT.parent
sys.path.insert(0, str(DAN20))

from utilities.dataset import PhasePairDataset
from utilities.generator_crb import UNetCRB
from utilities.losses import DVFMSELoss

warnings.filterwarnings('ignore')


def parse_held_out(s: str):
    ids = [int(x.strip()) for x in s.split(',') if x.strip()]
    if not ids:
        raise ValueError('--held_out must list 1-indexed phases, e.g. 5,9')
    return ids, [p - 1 for p in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--held_out', required=True)
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--lr', type=float, default=1e-4)
    args = ap.parse_args()

    held_1idx, held_0idx = parse_held_out(args.held_out)
    run_dir = ENC_ROOT / args.run_dir
    data_all = str(DAN20.parent / 'data' / 'spare' / 'all')
    tag = '_'.join(f'{p:02d}' for p in held_1idx)
    filename = f'crb_enc_mse_lo_{tag}'

    im_size = 64
    n_phases = 10
    batch_size = 1
    patches_per_pair_train = 16
    patches_per_pair_val = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainset = PhasePairDataset(
        im_dir=data_all, im_size=im_size, random_crop=True,
        patches_per_pair=patches_per_pair_train,
        held_out_phases=held_0idx, holdout_mode='exclude',
    )
    valset = PhasePairDataset(
        im_dir=data_all, im_size=im_size, random_crop=False,
        patches_per_pair=patches_per_pair_val,
        held_out_phases=held_0idx, holdout_mode='include',
    )
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    generator = UNetCRB(im_size=im_size, n_phases=n_phases).to(device)
    mse_loss = DVFMSELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr)

    os.makedirs(run_dir / 'weights', exist_ok=True)
    os.makedirs(run_dir / 'plots', exist_ok=True)

    min_val_loss = float('inf')
    train_losses, val_losses = [], []
    tic = time.time()
    n_g = sum(p.numel() for p in generator.parameters())
    print(
        f'[{filename}] device={device} | UNetCRB params={n_g / 1e6:.2f}M | '
        f'loss=lung-masked MSE | no D | CRB=encoder-only | held_out_1idx={held_1idx}'
    )
    print(
        f'[{filename}] train {len(trainset)} '
        f'({len(trainset) // patches_per_pair_train} pairs × {patches_per_pair_train}) | '
        f'val {len(valset)} ({len(valset) // patches_per_pair_val} pairs × {patches_per_pair_val})'
    )

    for epoch in range(1, args.epochs + 1):
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
            loss.backward()
            optimizer_g.step()
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
        hours = int(np.floor(elapsed_h))
        minutes = int((elapsed_h - hours) * 60)
        n_train = max(len(trainset), 1)
        n_val = max(len(valset), 1)
        train_losses.append(train_loss / n_train)
        val_losses.append(val_loss / n_val)
        print(
            'Epoch: %d | train MSE: %.6f | val MSE: %.6f | total time: %d hours %d minutes'
            % (epoch, train_losses[-1], val_losses[-1], hours, minutes),
            flush=True,
        )
        if val_losses[-1] < min_val_loss:
            torch.save(generator.state_dict(), str(run_dir / 'weights' / f'{filename}_generator.pth'))
            min_val_loss = val_losses[-1]
        plt.figure()
        plt.plot(np.arange(1, epoch + 1), train_losses, 'b', label='Train MSE')
        plt.plot(np.arange(1, epoch + 1), val_losses, 'r', label='Val MSE')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Lung-masked MSE')
        plt.title(filename)
        plt.savefig(str(run_dir / 'plots' / f'{filename}.png'))
        plt.close()

    print(f'finished best_val_mse={min_val_loss:.6f}', flush=True)


if __name__ == '__main__':
    main()
