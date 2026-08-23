"""Full-volume QC for Experiment 5 CRB models on hold-out patients.

Panel layout (per user request):
  Row 1: original ref CT | original target CT | |target − warp(ref, pred)| · lung
  Row 2: |Elastix DVF|   | |pred DVF|         | |pred − Elastix| · lung

Metrics = lung-masked L1 + cosine vs Elastix (same as Dan 2.0 QC).

  PYTHONPATH=.. python scripts/qc_pairs.py \\
    --arch encoder \\
    --ckpt EncoderCRB/weights/crb_enc_mse_pop_e5_generator.pth \\
    --out_dir EncoderCRB/plots/qc_holdout
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

E5 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E5))

from losses import losses as loss_mod
from networks.generator_crb import UNetCRB
from networks.generator_crb_both import UNetCRBBoth
from networks.generator_crb_dec import UNetCRBDecoder
from utilities.view_config import load_view_config, show_ct_slice, show_mag_slice
from utilities.warp import warp

ARCH = {
    'encoder': UNetCRB,
    'decoder': UNetCRBDecoder,
    'both': UNetCRBBoth,
}


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


def lung_mid_z(mask: np.ndarray) -> int:
    zs = np.where(mask.any(axis=(1, 2)))[0]
    if zs.size == 0:
        return mask.shape[0] // 2
    return int(zs[len(zs) // 2])


def parse_pair(name: str):
    stem = name.replace('_pair.npy', '')
    patient, rest = stem.split('_', 1)
    a, b = rest.split('_to_')
    return patient, int(a), int(b)


def save_panel(out_path, ct_r, ct_t, warped, abs_img_err, gt, pred, mask, cfg, title, l1, cos):
    r2 = show_ct_slice(ct_r, cfg)
    t2 = show_ct_slice(ct_t, cfg)
    w2 = show_ct_slice(warped, cfg)
    e2 = show_ct_slice(abs_img_err, cfg)
    gt_mag = show_mag_slice(gt, cfg)
    pr_mag = show_mag_slice(pred, cfg)
    err_mag = show_ct_slice(np.linalg.norm(pred - gt, axis=0) * mask, cfg)

    vmax_mag = float(np.percentile(np.concatenate([gt_mag.ravel(), pr_mag.ravel()]), 99))
    vmax_mag = max(vmax_mag, 1e-3)
    pos_img = e2[e2 > 0]
    vmax_img = float(np.percentile(pos_img, 99)) if pos_img.size else 1.0
    vmax_img = max(vmax_img, 1e-3)
    pos = err_mag[err_mag > 0]
    vmax_err = float(np.percentile(pos, 99)) if pos.size else 1.0
    vmax_err = max(vmax_err, 1e-3)

    # 2×4: CTs + warp + image residual on top; DVF magnitudes / error on bottom
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    for ax, im, ttl, cmap, vmin, vmax, cbl in [
        (axs[0, 0], r2, 'ref CT (original)', 'gray', None, None, 'norm. HU'),
        (axs[0, 1], t2, 'target CT (original)', 'gray', None, None, 'norm. HU'),
        (axs[0, 2], w2, 'warp(ref, pred DVF)', 'gray', None, None, 'norm. HU'),
        (axs[0, 3], e2, '|target − warp| · lung', 'hot', 0, vmax_img, '|ΔI|'),
        (axs[1, 0], gt_mag, '|Elastix DVF|', 'magma', 0, vmax_mag, '|u| (mm)'),
        (axs[1, 1], pr_mag, '|pred DVF|', 'magma', 0, vmax_mag, '|u| (mm)'),
        (axs[1, 2], err_mag, '|pred − Elastix| · lung', 'hot', 0, vmax_err, '|Δu| (mm)'),
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
    axs[1, 3].axis('off')

    fig.suptitle(
        f'{title}  |  lung mid-Z slice {cfg.slice_index}  |  '
        f'L1={l1:.3f} cos={cos:.3f}  |  1 voxel = 1 mm'
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', required=True, choices=sorted(ARCH))
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--split', default='test', choices=['test', 'val', 'train'])
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument(
        '--view_config',
        default=str(E5 / 'configs' / 'dvf_view_config.json'),
    )
    ap.add_argument('--max_pairs', type=int, default=0, help='0 = all')
    ap.add_argument('--anatomy_dim', type=int, default=32)
    args = ap.parse_args()

    man = json.loads((E5 / 'data' / 'pooled' / 'manifest.json').read_text())
    if args.split == 'test':
        data_dir = Path(man['pooled_test_dir'])
        pair_names = man['test_pairs']
    else:
        data_dir = Path(man['pooled_train_dir'])
        pair_names = man['val_pairs'] if args.split == 'val' else man['train_pairs']

    if args.max_pairs > 0:
        pair_names = pair_names[: args.max_pairs]

    base_cfg = load_view_config(args.view_config)
    out_dir = Path(args.out_dir)
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    if args.arch == 'decoder':
        g = UNetCRBDecoder(im_size=128, n_phases=10)
    else:
        g = ARCH[args.arch](im_size=128, n_phases=10, anatomy_dim=args.anatomy_dim)
    g = g.to(device)
    g.load_state_dict(torch.load(args.ckpt, map_location=device))
    g.eval()
    dvf_l1 = loss_mod.DVFLoss()

    print(
        f'arch={args.arch} ckpt={args.ckpt} split={args.split} '
        f'pairs={len(pair_names)} out={out_dir}',
        flush=True,
    )

    rows = []
    mask_cache = {}
    with torch.no_grad():
        for i, name in enumerate(pair_names, 1):
            patient, ref, tgt = parse_pair(name)
            if patient not in mask_cache:
                mask_cache[patient] = (
                    np.load(data_dir / f'{patient}_Mask_Lung.npy') > 0
                ).astype(np.float32)
            mask = mask_cache[patient]
            cfg = deepcopy(base_cfg)
            cfg.slice_index = lung_mid_z(mask)

            ct_r = _norm_ct(np.load(data_dir / f'{patient}_CT_{ref:02d}.npy'))
            ct_t = _norm_ct(np.load(data_dir / f'{patient}_CT_{tgt:02d}.npy'))
            if ref == tgt:
                gt = np.zeros((3,) + ct_r.shape, dtype=np.float32)
            else:
                gt = _to_cdhw(np.load(data_dir / name).astype(np.float32))

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
            abs_img_err = np.abs(ct_t - warped) * mask

            l1 = float(dvf_l1.loss(gt_t, pred_t, mask_t).item())
            zero = float(dvf_l1.loss(gt_t, torch.zeros_like(gt_t), mask_t).item())
            if ref != tgt:
                m = mask > 0.5
                a, b = pred * m[None], gt * m[None]
                cos = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-8))
            else:
                cos = float('nan')

            tag = f'{patient}_{ref:02d}_to_{tgt:02d}'
            note = 'identity' if ref == tgt else 'pair'
            save_panel(
                out_dir / f'{tag}.png',
                ct_r, ct_t, warped, abs_img_err, gt, pred, mask, cfg,
                f'{tag} ({note})', l1, 0.0 if cos != cos else cos,
            )
            rows.append((tag, note, patient, l1, zero, cos))
            print(f'[{i:03d}/{len(pair_names)}] {tag} L1={l1:.4f} cos={cos:.4f}', flush=True)

    with open(out_dir / 'metrics.tsv', 'w') as f:
        f.write('pair\tnote\tpatient\tL1_pred\tL1_zero\tcos\n')
        for tag, note, patient, l1, zero, cos in rows:
            f.write(f'{tag}\t{note}\t{patient}\t{l1:.6f}\t{zero:.6f}\t{cos:.6f}\n')

    directed = [r for r in rows if r[1] != 'identity']
    print(
        f'directed mean L1={np.mean([r[3] for r in directed]):.4f} '
        f'cos={np.nanmean([r[5] for r in directed]):.4f} n={len(directed)}'
    )
    for patient in sorted({r[2] for r in directed}):
        sub = [r for r in directed if r[2] == patient]
        print(
            f'  {patient} L1={np.mean([r[3] for r in sub]):.4f} '
            f'cos={np.nanmean([r[5] for r in sub]):.4f} n={len(sub)}'
        )


if __name__ == '__main__':
    main()
