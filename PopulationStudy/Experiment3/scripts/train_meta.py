"""Experiment 3 episodic train: leave-one-train-patient-out each epoch.

Support = remaining train patients. Query = excluded (zero-shot-this-step)
patient. Loss = L_support + λ L_query (lung-masked MSE). True hold-out P7/P9
is never used here.

  PYTHONPATH=. python scripts/train_meta.py --arch encoder --gpu 0
  PYTHONPATH=. python EncoderCRB/train_crb_enc_mse.py --gpu 0
"""

from __future__ import annotations

import argparse
import itertools
import json
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

E3 = Path(__file__).resolve().parents[1]
if str(E3) not in sys.path:
    sys.path.insert(0, str(E3))

from losses.losses import DVFMSELoss
from networks.generator_crb import UNetCRB
from networks.generator_crb_both import UNetCRBBoth
from networks.generator_crb_dec import UNetCRBDecoder
from utilities.dataset import (
    MultiPatientPhasePairDataset,
    group_pairs_by_patient,
)
from utilities.seed_utils import load_seed_config, set_seeds

warnings.filterwarnings('ignore')

ARCH = {
    'encoder': {
        'cls': UNetCRB,
        'run_dir': E3 / 'EncoderCRB',
        'filename': 'crb_enc_mse_pop_e3',
        'label': 'CRB=encoder-only',
    },
    'decoder': {
        'cls': UNetCRBDecoder,
        'run_dir': E3 / 'DecoderCRB',
        'filename': 'crb_dec_mse_pop_e3',
        'label': 'CRB=decoder-only',
    },
    'both': {
        'cls': UNetCRBBoth,
        'run_dir': E3 / 'BothCRB',
        'filename': 'crb_both_mse_pop_e3',
        'label': 'CRB=encoder+decoder',
    },
}

TAG = {'P1': 'P01', 'P3': 'P03', 'P4': 'P04', 'P5': 'P05'}


def load_manifest():
    path = E3 / 'data' / 'pooled' / 'manifest.json'
    with open(path) as f:
        return json.load(f)


def _batch_mse(mse_loss, generator, data, device):
    reference_ct = data['reference_ct'].to(device)
    lung_mask = data['lung_mask'].to(device)
    ref_phase = data['ref_phase'].to(device)
    target_phase = data['target_phase'].to(device)
    target_dvf = data['target_dvf'].to(device)
    fake_dvf = generator(reference_ct, ref_phase, target_phase)
    return mse_loss.loss(target_dvf, fake_dvf, lung_mask)


def query_patient_for_epoch(epoch, train_pids, schedule, fixed_pid):
    if schedule == 'fixed_p1':
        return fixed_pid
    return train_pids[(epoch - 1) % len(train_pids)]


def make_loader(im_dir, pair_files, im_size, random_crop, patches, seed, shuffle):
    ds = MultiPatientPhasePairDataset(
        im_dir=im_dir,
        pair_files=pair_files,
        im_size=im_size,
        random_crop=random_crop,
        patches_per_pair=patches,
    )
    kwargs = dict(batch_size=1, shuffle=shuffle)
    if shuffle:
        g = torch.Generator()
        g.manual_seed(seed)
        kwargs['generator'] = g
    return ds, DataLoader(ds, **kwargs)


def train_one(args):
    spec = ARCH[args.arch]
    cfg = load_seed_config()
    seed = args.seed if args.seed is not None else int(cfg['torch_seed'])
    set_seeds(seed)
    lam = float(args.lambda_query)
    schedule = args.query_schedule
    fixed_pid = cfg.get('inner_zero_shot_default', 'P1')

    run_dir = spec['run_dir']
    filename = spec['filename']
    man = load_manifest()
    train_dir = man['pooled_train_dir']
    train_pids = list(man['train_patients'])

    im_size = 64
    n_phases = 10
    patches_s = int(args.support_patches)
    patches_q = int(args.query_patches)
    patches_val = 8
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    by_patient = group_pairs_by_patient(man['train_pairs'])
    valset, valloader = make_loader(
        train_dir, man['val_pairs'], im_size, False, patches_val, seed, False
    )

    generator = spec['cls'](im_size=im_size, n_phases=n_phases).to(device)
    mse_loss = DVFMSELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr)

    os.makedirs(run_dir / 'weights', exist_ok=True)
    os.makedirs(run_dir / 'plots', exist_ok=True)

    min_val_loss = float('inf')
    train_losses, val_losses, query_losses = [], [], []
    tic = time.time()
    n_g = sum(p.numel() for p in generator.parameters())
    print(
        f'[{filename}] device={device} | {spec["cls"].__name__} params={n_g / 1e6:.2f}M | '
        f'episodic L_s + {lam} L_q | schedule={schedule} | seed={seed}',
        flush=True,
    )
    print(
        f'[{filename}] train pool={train_pids} | true hold-out={man["holdout_patients"]} '
        f'(QC only) | inner zero-out default={fixed_pid} | val {len(valset)}',
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        q_pid = query_patient_for_epoch(epoch, train_pids, schedule, fixed_pid)
        q_tag = TAG[q_pid]
        support_files, query_files = [], []
        for tag, names in by_patient.items():
            if tag == q_tag:
                query_files.extend(names)
            else:
                support_files.extend(names)
        if not support_files or not query_files:
            raise RuntimeError(f'empty support/query for query={q_pid}')

        _, s_loader = make_loader(
            train_dir, support_files, im_size, True, patches_s, seed + epoch, True
        )
        _, q_loader = make_loader(
            train_dir, query_files, im_size, True, patches_q, seed + epoch + 17, True
        )

        generator.train()
        q_iter = itertools.cycle(q_loader)
        train_loss = 0.0
        q_loss_acc = 0.0
        n_step = 0
        for sdata in s_loader:
            qdata = next(q_iter)
            optimizer_g.zero_grad()
            loss_s = _batch_mse(mse_loss, generator, sdata, device)
            loss_q = _batch_mse(mse_loss, generator, qdata, device)
            loss = loss_s + lam * loss_q
            loss.backward()
            optimizer_g.step()
            train_loss += loss.item()
            q_loss_acc += loss_q.item()
            n_step += 1

        generator.eval()
        val_loss = 0.0
        with torch.no_grad():
            for valdata in valloader:
                val_loss += _batch_mse(mse_loss, generator, valdata, device).item()

        toc = time.time()
        elapsed_h = (toc - tic) / 3600.0
        hours = int(np.floor(elapsed_h))
        minutes = int((elapsed_h - hours) * 60)
        n_val = max(len(valset), 1)
        train_losses.append(train_loss / max(n_step, 1))
        query_losses.append(q_loss_acc / max(n_step, 1))
        val_losses.append(val_loss / n_val)
        print(
            'Epoch: %d | query=%s | train(L_s+λL_q): %.6f | query MSE: %.6f | '
            'val MSE: %.6f | total time: %d hours %d minutes'
            % (
                epoch, q_pid, train_losses[-1], query_losses[-1],
                val_losses[-1], hours, minutes,
            ),
            flush=True,
        )
        if val_losses[-1] < min_val_loss:
            torch.save(
                generator.state_dict(),
                str(run_dir / 'weights' / f'{filename}_generator.pth'),
            )
            min_val_loss = val_losses[-1]
        plt.figure()
        xs = np.arange(1, epoch + 1)
        plt.plot(xs, train_losses, 'b-o', markersize=3, label='Train L_s+λL_q')
        plt.plot(xs, query_losses, 'g-o', markersize=3, label='Query MSE')
        plt.plot(xs, val_losses, 'r-o', markersize=3, label='Val MSE')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Lung-masked MSE')
        plt.title(filename)
        plt.savefig(str(run_dir / 'plots' / f'{filename}.png'))
        plt.close()

    print(f'finished best_val_mse={min_val_loss:.6f}', flush=True)
    return min_val_loss


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', required=True, choices=sorted(ARCH))
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument(
        '--query_schedule',
        default='rotate',
        choices=['rotate', 'fixed_p1'],
        help='rotate train patients as query, or always hold out P1',
    )
    ap.add_argument('--lambda_query', type=float, default=2.5)
    ap.add_argument('--support_patches', type=int, default=16)
    ap.add_argument('--query_patches', type=int, default=8)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    train_one(args)


if __name__ == '__main__':
    main()
