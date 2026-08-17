"""Green deformation-mesh overlays for a phase pair (View 1, no SI/AP labels)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection

from utilities.generator import UNetFiLM
from utilities.view_config import load_view_config, show_ct_slice

# VoxelMap helper for consistent plane orient
import sys
sys.path.insert(0, '/home/abhishek/Documents/VoxelMap_Clinical')
from ml.volume_view import _orient_plane_2d  # noqa: E402


def _norm(x):
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def _to_cdhw(dvf):
    if dvf.ndim == 4 and dvf.shape[-1] == 3:
        return np.moveaxis(dvf, -1, 0)
    return dvf


def extract_flow_plane(flow_chw, cfg):
    sa, ha, va = int(cfg.slice_axis), int(cfg.h_axis), int(cfg.v_axis)
    si = int(cfg.clamp_slice(flow_chw.shape[1:]))
    u = _orient_plane_2d(np.take(flow_chw[ha], si, axis=sa), sa, ha, va)
    v = _orient_plane_2d(np.take(flow_chw[va], si, axis=sa), sa, ha, va)
    if cfg.flip_h:
        u = np.flip(u, axis=1)
        v = np.flip(v, axis=1)
        u = -u
    if cfg.flip_v:
        u = np.flip(u, axis=0)
        v = np.flip(v, axis=0)
        v = -v
    return u.astype(np.float32), v.astype(np.float32)


def grid_segments(x, y):
    segs = []
    for i in range(x.shape[0]):
        segs.append(np.stack([x[i], y[i]], axis=-1))
    for j in range(x.shape[1]):
        segs.append(np.stack([x[:, j], y[:, j]], axis=-1))
    return segs


def draw_mesh(ax, ct, x0, y0, x1, y1, title):
    H, W = ct.shape
    ax.imshow(ct, cmap='gray', origin='upper', aspect='equal', vmin=0, vmax=1)
    ax.add_collection(
        LineCollection(grid_segments(x0, y0), colors=(0.2, 0.9, 0.3, 0.25), linewidths=0.6)
    )
    ax.add_collection(
        LineCollection(grid_segments(x1, y1), colors=(0.1, 1.0, 0.2, 0.95), linewidths=1.0)
    )
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', type=int, default=1)
    ap.add_argument('--tgt', type=int, default=6)
    ap.add_argument('--data_dir', default='data/spare/all')
    ap.add_argument('--out_dir', default='plots/qc_mesh_01_to_06')
    ap.add_argument('--view_config', default='configs/dvf_view_config.json')
    ap.add_argument('--step', type=int, default=6)
    args = ap.parse_args()

    cfg = load_view_config(args.view_config)
    data = Path(args.data_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ct_r = _norm(np.load(data / f'CT_{args.ref:02d}.npy'))
    gt = _to_cdhw(np.load(data / f'{args.ref:02d}_to_{args.tgt:02d}_pair.npy').astype(np.float32))
    ct2 = show_ct_slice(ct_r, cfg)
    H, W = ct2.shape
    ys = np.arange(args.step // 2, H, args.step)
    xs = np.arange(args.step // 2, W, args.step)
    xx, yy = np.meshgrid(xs, ys)

    def warp_grid(u, v):
        cols = np.clip(xx.astype(int), 0, W - 1)
        rows = np.clip(yy.astype(int), 0, H - 1)
        # display: row~SI component u, col~AP component v (matches extract_flow_plane)
        return xx + v[rows, cols], yy + u[rows, cols]

    x0, y0 = xx.astype(float), yy.astype(float)
    x_gt, y_gt = warp_grid(*extract_flow_plane(gt, cfg))

    g = UNetFiLM(im_size=128, n_phases=10, int_steps=6).to(device)

    def pred_dvf(ckpt):
        g.load_state_dict(torch.load(ckpt, map_location=device))
        g.eval()
        with torch.no_grad():
            return g(
                torch.from_numpy(ct_r)[None, None].to(device),
                torch.tensor([args.ref - 1], device=device),
                torch.tensor([args.tgt - 1], device=device),
            )[0].cpu().numpy()

    pred_mlp = pred_dvf('weights/spare_mc_p1_dvf_gan_phase_mlp_generator.pth')
    pred_wd = pred_dvf('weights/spare_mc_p1_scenario_warp_d_generator.pth')
    x_mlp, y_mlp = warp_grid(*extract_flow_plane(pred_mlp, cfg))
    x_wd, y_wd = warp_grid(*extract_flow_plane(pred_wd, cfg))

    tag = f'{args.ref:02d}_to_{args.tgt:02d}'
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.8))
    draw_mesh(axs[0], ct2, x0, y0, x0, y0, 'undeformed mesh on ref CT')
    draw_mesh(axs[1], ct2, x0, y0, x_gt, y_gt, 'mesh warped by Elastix DVF')
    draw_mesh(axs[2], ct2, x0, y0, x_mlp, y_mlp, 'mesh warped by phase-MLP pred')
    fig.suptitle(f'{tag}  View 1 slice {cfg.slice_index}  |  green mesh = DVF push/pull')
    fig.tight_layout()
    fig.savefig(out / 'mesh_compare_elastix_mlp.png', dpi=150)
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(11, 5.2))
    draw_mesh(axs[0], ct2, x0, y0, x_mlp, y_mlp, 'phase-MLP pred DVF')
    draw_mesh(axs[1], ct2, x0, y0, x_wd, y_wd, 'warp-D pred DVF')
    fig.suptitle(f'{tag} green deformation mesh on ref CT')
    fig.tight_layout()
    fig.savefig(out / 'mesh_mlp_vs_warp_d.png', dpi=150)
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, (x1, y1), title in [
        (axs[0], (x_mlp, y_mlp), 'phase-MLP (zoom)'),
        (axs[1], (x_wd, y_wd), 'warp-D (zoom)'),
    ]:
        draw_mesh(ax, ct2, x0, y0, x1, y1, title)
        ax.set_xlim(20, 110)
        ax.set_ylim(110, 20)
    fig.suptitle(f'{tag} zoomed mesh')
    fig.tight_layout()
    fig.savefig(out / 'mesh_zoom.png', dpi=150)
    plt.close()
    print('wrote', out)


if __name__ == '__main__':
    main()
