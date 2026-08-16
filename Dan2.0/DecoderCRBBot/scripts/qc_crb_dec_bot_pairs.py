"""Full-volume QC for Decoder-CRB+Bot MSE. Metrics = L1 + cos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

BOT_ROOT = Path(__file__).resolve().parents[1]
DAN20 = BOT_ROOT.parent
REPO = DAN20.parent
sys.path.insert(0, str(DAN20))
sys.path.insert(0, str(BOT_ROOT))

from generator_crb_dec_bot import UNetCRBDecoderBot
from utilities.warp import warp
from utilities import losses
from utilities.view_config import load_view_config, show_ct_slice, show_mag_slice


def _norm_ct(x):
    x = x.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _to_cdhw(dvf):
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


def save_panel(out_path, ct_r, ct_t, warped, gt, pred, mask, cfg, title, l1, cos):
    r2 = show_ct_slice(ct_r, cfg)
    t2 = show_ct_slice(ct_t, cfg)
    w2 = show_ct_slice(warped, cfg)
    gt_mag = show_mag_slice(gt, cfg)
    pr_mag = show_mag_slice(pred, cfg)
    err_mag = show_ct_slice(np.linalg.norm(pred - gt, axis=0) * mask, cfg)
    vmax_mag = float(np.percentile(np.concatenate([gt_mag.ravel(), pr_mag.ravel()]), 99))
    vmax_mag = max(vmax_mag, 1e-3)
    pos = err_mag[err_mag > 0]
    vmax_err = float(np.percentile(pos, 99)) if pos.size else 1.0
    vmax_err = max(vmax_err, 1e-3)
    fig, axs = plt.subplots(2, 3, figsize=(13, 8))
    for ax, im, ttl, cmap, vmin, vmax, cbl in [
        (axs[0, 0], r2, 'ref CT', 'gray', None, None, 'norm. HU'),
        (axs[0, 1], t2, 'target CT', 'gray', None, None, 'norm. HU'),
        (axs[0, 2], w2, 'warp(ref, pred DVF)', 'gray', None, None, 'norm. HU'),
        (axs[1, 0], gt_mag, '|Elastix DVF|', 'magma', 0, vmax_mag, '|u| (mm)'),
        (axs[1, 1], pr_mag, '|pred DVF|', 'magma', 0, vmax_mag, '|u| (mm)'),
        (axs[1, 2], err_mag, '|err| · lung mask', 'hot', 0, vmax_err, '|pred−Elastix| (mm)'),
    ]:
        kw = dict(cmap=cmap, origin='upper', aspect='equal')
        if vmin is not None:
            kw['vmin'] = vmin
            kw['vmax'] = vmax
        h = ax.imshow(im, **kw)
        ax.set_title(ttl)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(h, ax=ax, fraction=0.046, pad=0.04).set_label(cbl)
    fig.suptitle(
        f'{title}  |  View 1 slice {cfg.slice_index}  |  '
        f'L1={l1:.3f} cos={cos:.3f}  |  1 voxel = 1 mm'
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--held_out', default='5,9')
    ap.add_argument('--data_dir', default=str(REPO / 'data' / 'spare' / 'all'))
    ap.add_argument('--view_config', default=str(DAN20 / 'configs' / 'dvf_view_config.json'))
    args = ap.parse_args()

    held = {int(x.strip()) for x in args.held_out.split(',') if x.strip()}
    cfg = load_view_config(args.view_config)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g = UNetCRBDecoderBot(im_size=128, n_phases=10).to(device)
    g.load_state_dict(torch.load(args.ckpt, map_location=device))
    g.eval()
    dvf_l1 = losses.DVFLoss()
    pairs = list_pairs(data_dir)
    mask = (np.load(data_dir / 'Mask_Lung.npy') > 0).astype(np.float32)
    print(f'ckpt={args.ckpt} pairs={len(pairs)} out={out_dir} held_out={sorted(held)}')

    rows = []
    with torch.no_grad():
        for i, (ref, tgt) in enumerate(pairs, 1):
            ct_r = _norm_ct(np.load(data_dir / f'CT_{ref:02d}.npy'))
            ct_t = _norm_ct(np.load(data_dir / f'CT_{tgt:02d}.npy'))
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
            warped = warp(ref_t, pred_t)[0, 0].cpu().numpy()
            pred = pred_t[0].cpu().numpy()
            l1 = float(dvf_l1.loss(gt_t, pred_t, mask_t).item())
            zero = float(dvf_l1.loss(gt_t, torch.zeros_like(gt_t), mask_t).item())
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
                f'{tag} ({note})', l1, 0.0 if cos != cos else cos,
            )
            rows.append((tag, note, l1, zero, cos))
            print(f'[{i:03d}/{len(pairs)}] {tag} L1={l1:.4f} cos={cos:.4f}', flush=True)

    with open(out_dir / 'metrics.tsv', 'w') as f:
        f.write('pair\tnote\tL1_pred\tL1_zero\tcos\n')
        for tag, note, l1, zero, cos in rows:
            f.write(f'{tag}\t{note}\t{l1:.6f}\t{zero:.6f}\t{cos:.6f}\n')
    directed = [r for r in rows if r[1] != 'identity']
    loo = [r for r in directed if int(r[0][:2]) in held or int(r[0][6:8]) in held]
    held_lab = '/'.join(str(p) for p in sorted(held))
    print(
        f'directed mean L1={np.mean([r[2] for r in directed]):.4f} '
        f'cos={np.nanmean([r[4] for r in directed]):.4f}'
    )
    print(
        f'leave-out {held_lab} mean L1={np.mean([r[2] for r in loo]):.4f} '
        f'cos={np.nanmean([r[4] for r in loo]):.4f} n={len(loo)}'
    )


if __name__ == '__main__':
    main()
