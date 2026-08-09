"""Full-volume warp-D QC panels with locked view config + mm colorbars.

Generates one PNG per phase pair under an output directory.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from utilities.generator import UNetFiLM
from utilities.svf import warp
from utilities import losses
from utilities.view_config import load_view_config, show_ct_slice, show_mag_slice


def _norm_ct(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _to_cdhw(dvf: np.ndarray) -> np.ndarray:
    if dvf.ndim == 4 and dvf.shape[-1] == 3:
        return np.moveaxis(dvf, -1, 0)
    return dvf


def list_pairs(data_dir: Path):
    pairs = []
    for p in sorted(data_dir.glob('*_pair.npy')):
        stem = p.name.replace('_pair.npy', '')
        ref_s, tgt_s = stem.split('_to_')
        pairs.append((int(ref_s), int(tgt_s)))
    return pairs


def save_panel(
    out_path: Path,
    ct_r, ct_t, warped, gt, pred, mask, cfg,
    title: str,
    l1: float,
    cos: float,
):
    r2 = show_ct_slice(ct_r, cfg)
    t2 = show_ct_slice(ct_t, cfg)
    w2 = show_ct_slice(warped, cfg)
    gt_mag = show_mag_slice(gt, cfg)
    pr_mag = show_mag_slice(pred, cfg)
    err_vol = np.linalg.norm(pred - gt, axis=0) * mask
    err_mag = show_ct_slice(err_vol, cfg)

    vmax_mag = float(np.percentile(np.concatenate([gt_mag.ravel(), pr_mag.ravel()]), 99))
    vmax_mag = max(vmax_mag, 1e-3)
    pos = err_mag[err_mag > 0]
    vmax_err = float(np.percentile(pos, 99)) if pos.size else 1.0
    vmax_err = max(vmax_err, 1e-3)

    fig, axs = plt.subplots(2, 3, figsize=(13, 8))
    im0 = axs[0, 0].imshow(r2, cmap='gray', origin='upper', aspect='equal')
    axs[0, 0].set_title('ref CT')
    fig.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04).set_label('norm. HU')

    im1 = axs[0, 1].imshow(t2, cmap='gray', origin='upper', aspect='equal')
    axs[0, 1].set_title('target CT')
    fig.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04).set_label('norm. HU')

    im2 = axs[0, 2].imshow(w2, cmap='gray', origin='upper', aspect='equal')
    axs[0, 2].set_title('warp(ref, pred DVF)')
    fig.colorbar(im2, ax=axs[0, 2], fraction=0.046, pad=0.04).set_label('norm. HU')

    im3 = axs[1, 0].imshow(gt_mag, cmap='magma', origin='upper', aspect='equal', vmin=0, vmax=vmax_mag)
    axs[1, 0].set_title('|Elastix DVF|')
    fig.colorbar(im3, ax=axs[1, 0], fraction=0.046, pad=0.04).set_label('|u| (mm)')

    im4 = axs[1, 1].imshow(pr_mag, cmap='magma', origin='upper', aspect='equal', vmin=0, vmax=vmax_mag)
    axs[1, 1].set_title('|pred DVF|')
    fig.colorbar(im4, ax=axs[1, 1], fraction=0.046, pad=0.04).set_label('|u| (mm)')

    im5 = axs[1, 2].imshow(err_mag, cmap='hot', origin='upper', aspect='equal', vmin=0, vmax=vmax_err)
    axs[1, 2].set_title('|err| · lung mask')
    fig.colorbar(im5, ax=axs[1, 2], fraction=0.046, pad=0.04).set_label('|pred−Elastix| (mm)')

    for ax in axs.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

    fig.suptitle(
        f'{title}  |  View 1 slice {cfg.slice_index} ⊥{cfg.slice_normal}  |  '
        f'L1={l1:.3f} cos={cos:.3f}  |  1 voxel = 1 mm'
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='weights/spare_mc_p1_scenario_warp_d_generator.pth')
    ap.add_argument('--data_dir', default='data/spare/all')
    ap.add_argument('--out_dir', default='plots/qc_test_warp_d')
    ap.add_argument('--view_config', default='configs/dvf_view_config.json')
    ap.add_argument('--summary', default=None, help='optional metrics TSV path')
    args = ap.parse_args()

    cfg = load_view_config(args.view_config)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g = UNetFiLM(im_size=128, n_phases=10, int_steps=6).to(device)
    g.load_state_dict(torch.load(args.ckpt, map_location=device))
    g.eval()
    dvf_loss = losses.DVFLoss()

    pairs = list_pairs(data_dir)
    print(f'ckpt={args.ckpt}  pairs={len(pairs)}  out={out_dir}  device={device}')
    print(cfg.orientation_summary())

    mask = (np.load(data_dir / 'Mask_Lung.npy') > 0).astype(np.float32)
    rows = []
    summary_path = Path(args.summary) if args.summary else out_dir / 'metrics.tsv'

    with torch.no_grad():
        for i, (ref, tgt) in enumerate(pairs, 1):
            ct_r = _norm_ct(np.load(data_dir / f'CT_{ref:02d}.npy'))
            ct_t = _norm_ct(np.load(data_dir / f'CT_{tgt:02d}.npy'))
            if ref == tgt:
                gt = np.zeros((3,) + ct_r.shape, dtype=np.float32)
            else:
                gt = _to_cdhw(np.load(data_dir / f'{ref:02d}_to_{tgt:02d}_pair.npy').astype(np.float32))

            ref_t = torch.from_numpy(ct_r)[None, None].to(device)
            mask_t = torch.from_numpy(mask)[None, None].to(device)
            gt_t = torch.from_numpy(gt)[None].to(device)
            rp = torch.tensor([ref - 1], device=device)
            tp = torch.tensor([tgt - 1], device=device)
            pred_t = g(ref_t, rp, tp)
            warped = warp(ref_t, pred_t)[0, 0].cpu().numpy()
            pred = pred_t[0].cpu().numpy()
            l1 = float(dvf_loss.loss(gt_t, pred_t, mask_t).item())
            zero = float(dvf_loss.loss(gt_t, torch.zeros_like(gt_t), mask_t).item())
            if ref != tgt:
                m = mask > 0.5
                a, b = pred * m[None], gt * m[None]
                cos = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-8))
            else:
                cos = float('nan')

            tag = f'{ref:02d}_to_{tgt:02d}'
            note = 'identity' if ref == tgt else 'pair'
            save_panel(
                out_dir / f'{tag}.png',
                ct_r, ct_t, warped, gt, pred, mask, cfg,
                title=f'{tag} ({note})',
                l1=l1,
                cos=cos if cos == cos else 0.0,
            )
            rows.append((tag, note, l1, zero, cos))
            print(f'[{i:03d}/{len(pairs)}] {tag:10s} L1={l1:.4f} zero={zero:.4f} cos={cos:.4f}', flush=True)

    with open(summary_path, 'w') as f:
        f.write('pair\tnote\tL1_pred\tL1_zero\tcos\n')
        for tag, note, l1, zero, cos in rows:
            f.write(f'{tag}\t{note}\t{l1:.6f}\t{zero:.6f}\t{cos:.6f}\n')
    mean_l1 = float(np.mean([r[2] for r in rows if r[1] != 'identity']))
    mean_cos = float(np.nanmean([r[4] for r in rows if r[1] != 'identity']))
    print(f'done. non-identity mean L1={mean_l1:.4f} mean cos={mean_cos:.4f}')
    print(f'metrics → {summary_path}')


if __name__ == '__main__':
    main()
