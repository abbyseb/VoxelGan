"""Leave-phase-out TEST QC for phase-conditioned DVF generators.

Compares predicted DVF vs Elastix (lung-masked L1, cosine) and vs zero field.
Saves mid-slice panels under an output directory.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from utilities.generator import UNetFiLM
from utilities import losses


PAIRS = [
    # (split, ref, tgt, note) — 1-indexed phases
    ('val', 1, 5, 'held-out target'),
    ('val', 2, 5, 'held-out target'),
    ('val', 5, 1, 'held-out ref'),
    ('val', 5, 9, 'held-out both'),
    ('val', 9, 5, 'held-out both'),
    ('val', 9, 6, 'held-out ref'),
    ('train', 1, 1, 'identity'),
]


def _norm_ct(x):
    x = x.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _to_cdhw(dvf):
    # (D,H,W,3) or (3,D,H,W) → (3,D,H,W)
    if dvf.shape[-1] == 3:
        return np.moveaxis(dvf, -1, 0)
    return dvf


def load_pair(split, ref, tgt, size=64):
    ct_r = _norm_ct(np.load(f'data/spare/{split}/CT_{ref:02d}.npy'))
    ct_t = _norm_ct(np.load(f'data/spare/{split}/CT_{tgt:02d}.npy'))
    mask = (np.load(f'data/spare/{split}/Mask_Lung.npy') > 0).astype(np.float32)
    if ref == tgt:
        dvf = np.zeros((3,) + ct_r.shape, dtype=np.float32)
    else:
        dvf = _to_cdhw(
            np.load(f'data/spare/{split}/{ref:02d}_to_{tgt:02d}_pair.npy').astype(np.float32)
        )

    # center crop
    d, h, w = ct_r.shape
    z0, y0, x0 = (d - size) // 2, (h - size) // 2, (w - size) // 2
    sl = (slice(z0, z0 + size), slice(y0, y0 + size), slice(x0, x0 + size))
    ct_r = ct_r[sl]
    ct_t = ct_t[sl]
    mask = mask[sl]
    dvf = dvf[:, z0:z0 + size, y0:y0 + size, x0:x0 + size]
    return ct_r, ct_t, mask, dvf


def lung_cosine(pred, gt, mask, eps=1e-8):
    m = mask[None]
    a = pred * m
    b = gt * m
    num = (a * b).sum()
    den = np.sqrt((a * a).sum() * (b * b).sum()) + eps
    return float(num / den)


def neg_jacobian_frac(dvf):
    # dvf: (3, D, H, W) voxel displacements; rough finite-diff Jacobian det
    dx = np.gradient(dvf[0], axis=2)
    dy = np.gradient(dvf[1], axis=1)
    dz = np.gradient(dvf[2], axis=0)
    # J = I + grad(u)
    j00, j01, j02 = 1 + dx, np.gradient(dvf[0], axis=1), np.gradient(dvf[0], axis=0)
    j10, j11, j12 = np.gradient(dvf[1], axis=2), 1 + dy, np.gradient(dvf[1], axis=0)
    j20, j21, j22 = np.gradient(dvf[2], axis=2), np.gradient(dvf[2], axis=1), 1 + dz
    det = (
        j00 * (j11 * j22 - j12 * j21)
        - j01 * (j10 * j22 - j12 * j20)
        + j02 * (j10 * j21 - j11 * j20)
    )
    return float((det < 0).mean())


def save_panel(path, ct_r, gt, pred, mask, title):
    mid = ct_r.shape[0] // 2
    mag_gt = np.linalg.norm(gt, axis=0)
    mag_pr = np.linalg.norm(pred, axis=0)
    err = np.linalg.norm(pred - gt, axis=0) * mask
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.5))
    axs[0].imshow(ct_r[mid], cmap='gray')
    axs[0].set_title('ref CT')
    axs[1].imshow(mag_gt[mid], cmap='magma')
    axs[1].set_title('|Elastix|')
    axs[2].imshow(mag_pr[mid], cmap='magma')
    axs[2].set_title('|pred|')
    axs[3].imshow(err[mid], cmap='hot')
    axs[3].set_title('|err| lung')
    for ax in axs:
        ax.axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def eval_checkpoint(ckpt, out_dir, device, size=64):
    os.makedirs(out_dir, exist_ok=True)
    g = UNetFiLM(im_size=size, n_phases=10, int_steps=6).to(device)
    g.load_state_dict(torch.load(ckpt, map_location=device))
    g.eval()
    dvf_loss = losses.DVFLoss()

    rows = []
    print(f'\n=== {ckpt} → {out_dir} ===')
    print(f'{"pair":10s} {"note":18s} {"L1_pred":>8s} {"L1_zero":>8s} {"cos":>7s} {"|p|_lung":>8s} {"negJ%":>7s}')
    with torch.no_grad():
        for split, ref, tgt, note in PAIRS:
            ct_r, ct_t, mask, gt = load_pair(split, ref, tgt, size=size)
            ref_t = torch.from_numpy(ct_r)[None, None].to(device)
            mask_t = torch.from_numpy(mask)[None, None].to(device)
            gt_t = torch.from_numpy(gt)[None].to(device)
            rp = torch.tensor([ref - 1], dtype=torch.long, device=device)
            tp = torch.tensor([tgt - 1], dtype=torch.long, device=device)
            pred_t = g(ref_t, rp, tp)
            l1 = float(dvf_loss.loss(gt_t, pred_t, mask_t).item())
            zero = float(dvf_loss.loss(gt_t, torch.zeros_like(gt_t), mask_t).item())
            pred = pred_t[0].cpu().numpy()
            cos = lung_cosine(pred, gt, mask) if ref != tgt else float('nan')
            m = mask > 0.5
            mean_mag = float(np.linalg.norm(pred, axis=0)[m].mean()) if m.any() else 0.0
            negj = 100.0 * neg_jacobian_frac(pred)
            tag = f'{ref:02d}_to_{tgt:02d}'
            print(
                f'{tag:10s} {note:18s} {l1:8.4f} {zero:8.4f} {cos:7.3f} {mean_mag:8.3f} {negj:6.2f}%'
            )
            save_panel(
                os.path.join(out_dir, f'{tag}.png'),
                ct_r, gt, pred, mask,
                f'{tag} | {note} | L1={l1:.3f} cos={cos:.3f}',
            )
            rows.append({
                'pair': tag, 'note': note, 'L1_pred': l1, 'L1_zero': zero,
                'cos': cos, 'mean_mag': mean_mag, 'negJ_pct': negj,
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', action='append', required=True,
                    help='generator .pth (repeat for multiple models)')
    ap.add_argument('--out', action='append', required=True,
                    help='output dir per ckpt (same order)')
    ap.add_argument('--size', type=int, default=64)
    args = ap.parse_args()
    if len(args.ckpt) != len(args.out):
        raise SystemExit('--ckpt and --out counts must match')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_rows = {}
    for ckpt, out in zip(args.ckpt, args.out):
        all_rows[out] = eval_checkpoint(ckpt, out, device, size=args.size)

    # side-by-side summary if exactly two models
    if len(all_rows) == 2:
        (n0, r0), (n1, r1) = list(all_rows.items())
        print(f'\n=== side-by-side ({n0} vs {n1}) ===')
        print(f'{"pair":10s} {"L1_A":>8s} {"L1_B":>8s} {"dL1":>8s} {"cos_A":>7s} {"cos_B":>7s}')
        for a, b in zip(r0, r1):
            d = b['L1_pred'] - a['L1_pred']
            print(
                f'{a["pair"]:10s} {a["L1_pred"]:8.4f} {b["L1_pred"]:8.4f} {d:+8.4f} '
                f'{a["cos"]:7.3f} {b["cos"]:7.3f}'
            )


if __name__ == '__main__':
    main()
