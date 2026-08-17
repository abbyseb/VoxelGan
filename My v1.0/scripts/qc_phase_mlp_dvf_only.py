"""DVF-only QC panels (no warped CT) for a generator checkpoint.

Compares |Elastix|, |pred|, |err|·lung under the locked view config.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from utilities.generator import UNetFiLM
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
        a, b = stem.split('_to_')
        pairs.append((int(a), int(b)))
    return pairs


def save_dvf_panel(out_path, gt, pred, mask, cfg, title, l1, cos):
    gt_mag = show_mag_slice(gt, cfg)
    pr_mag = show_mag_slice(pred, cfg)
    err_mag = show_ct_slice(np.linalg.norm(pred - gt, axis=0) * mask, cfg)

    vmax_mag = float(np.percentile(np.concatenate([gt_mag.ravel(), pr_mag.ravel()]), 99))
    vmax_mag = max(vmax_mag, 1e-3)
    pos = err_mag[err_mag > 0]
    vmax_err = float(np.percentile(pos, 99)) if pos.size else 1.0
    vmax_err = max(vmax_err, 1e-3)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
    im0 = axs[0].imshow(gt_mag, cmap='magma', origin='upper', aspect='equal', vmin=0, vmax=vmax_mag)
    axs[0].set_title('|Elastix DVF|')
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04).set_label('|u| (mm)')

    im1 = axs[1].imshow(pr_mag, cmap='magma', origin='upper', aspect='equal', vmin=0, vmax=vmax_mag)
    axs[1].set_title('|pred DVF|')
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04).set_label('|u| (mm)')

    im2 = axs[2].imshow(err_mag, cmap='hot', origin='upper', aspect='equal', vmin=0, vmax=vmax_err)
    axs[2].set_title('|err| · lung mask')
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04).set_label('|pred−Elastix| (mm)')

    for ax in axs:
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
    ap.add_argument('--ckpt', default='weights/spare_mc_p1_dvf_gan_phase_mlp_generator.pth')
    ap.add_argument('--data_dir', default='data/spare/all')
    ap.add_argument('--out_dir', default='plots/qc_test_phase_mlp')
    ap.add_argument('--view_config', default='configs/dvf_view_config.json')
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
    mask = (np.load(data_dir / 'Mask_Lung.npy') > 0).astype(np.float32)
    print(f'ckpt={args.ckpt} pairs={len(pairs)} out={out_dir} device={device}')
    print(cfg.orientation_summary())

    rows = []
    with torch.no_grad():
        for i, (ref, tgt) in enumerate(pairs, 1):
            ct_r = _norm_ct(np.load(data_dir / f'CT_{ref:02d}.npy'))
            if ref == tgt:
                gt = np.zeros((3,) + ct_r.shape, dtype=np.float32)
            else:
                gt = _to_cdhw(
                    np.load(data_dir / f'{ref:02d}_to_{tgt:02d}_pair.npy').astype(np.float32)
                )

            ref_t = torch.from_numpy(ct_r)[None, None].to(device)
            mask_t = torch.from_numpy(mask)[None, None].to(device)
            gt_t = torch.from_numpy(gt)[None].to(device)
            pred_t = g(
                ref_t,
                torch.tensor([ref - 1], device=device),
                torch.tensor([tgt - 1], device=device),
            )
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
            save_dvf_panel(
                out_dir / f'{tag}.png',
                gt, pred, mask, cfg,
                title=f'{tag} ({note})',
                l1=l1,
                cos=0.0 if cos != cos else cos,
            )
            rows.append((tag, note, l1, zero, cos))
            print(f'[{i:03d}/{len(pairs)}] {tag:10s} L1={l1:.4f} zero={zero:.4f} cos={cos:.4f}', flush=True)

    summary = out_dir / 'metrics.tsv'
    with open(summary, 'w') as f:
        f.write('pair\tnote\tL1_pred\tL1_zero\tcos\n')
        for tag, note, l1, zero, cos in rows:
            f.write(f'{tag}\t{note}\t{l1:.6f}\t{zero:.6f}\t{cos:.6f}\n')
    dir_rows = [r for r in rows if r[1] != 'identity']
    mean_l1 = float(np.mean([r[2] for r in dir_rows]))
    mean_cos = float(np.nanmean([r[4] for r in dir_rows]))
    print(f'done. directed mean L1={mean_l1:.4f} mean cos={mean_cos:.4f}')
    print(f'metrics → {summary}')


if __name__ == '__main__':
    main()
