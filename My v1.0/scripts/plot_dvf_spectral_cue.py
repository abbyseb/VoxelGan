"""Visualize Elastix vs SVF-model DVF spectral / texture cue.

Shows why a 4-ch [ref, DVF] discriminator can cheat: B-spline Elastix
fields are smoother than SVF-integrated generator fields in frequency.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage

from utilities.generator import UNetFiLM
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


def _highpass_mag(vec_chw: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """|u - GaussianBlur(u)| magnitude — local texture residual."""
    residual = np.empty_like(vec_chw)
    for c in range(3):
        blur = ndimage.gaussian_filter(vec_chw[c], sigma=sigma)
        residual[c] = vec_chw[c] - blur
    return np.linalg.norm(residual, axis=0).astype(np.float32)


def _radial_psd(vec_chw: np.ndarray, mask: np.ndarray, n_bins: int = 48):
    """Mean radial power spectrum of lung-masked DVF channels."""
    m = mask.astype(bool)
    powers = []
    for c in range(3):
        vol = vec_chw[c].astype(np.float64).copy()
        vol[~m] = 0.0
        vol[m] -= vol[m].mean()
        F = np.fft.fftn(vol)
        P = np.abs(F) ** 2
        P = np.fft.fftshift(P)
        powers.append(P)
    Pmean = np.mean(powers, axis=0)

    zz, yy, xx = np.indices(Pmean.shape)
    cz, cy, cx = [s // 2 for s in Pmean.shape]
    r = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = float(min(cz, cy, cx))
    bins = np.linspace(0.0, r_max, n_bins + 1)
    inds = np.digitize(r.ravel(), bins) - 1
    psd = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)
    flat = Pmean.ravel()
    for i, v in zip(inds, flat):
        if 0 <= i < n_bins:
            psd[i] += v
            counts[i] += 1.0
    counts = np.maximum(counts, 1.0)
    psd /= counts
    freq = 0.5 * (bins[:-1] + bins[1:]) / r_max  # normalized [0, 1]
    return freq, psd


def _predict(g, ct_r, ref, tgt, device):
    ref_t = torch.from_numpy(ct_r)[None, None].to(device)
    rp = torch.tensor([ref - 1], device=device)
    tp = torch.tensor([tgt - 1], device=device)
    with torch.no_grad():
        pred = g(ref_t, rp, tp)[0].cpu().numpy()
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', type=int, default=1)
    ap.add_argument('--tgt', type=int, default=6)
    ap.add_argument('--data_dir', default='data/spare/all')
    ap.add_argument('--ckpt_phase_mlp', default='weights/spare_mc_p1_dvf_gan_phase_mlp_generator.pth')
    ap.add_argument('--ckpt_warp_d', default='weights/spare_mc_p1_scenario_warp_d_generator.pth')
    ap.add_argument('--view_config', default='configs/dvf_view_config.json')
    ap.add_argument('--out', default='plots/dvf_spectral_texture_cue.png')
    ap.add_argument('--hp_sigma', type=float, default=1.5)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    cfg = load_view_config(args.view_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ct_r = _norm_ct(np.load(data_dir / f'CT_{args.ref:02d}.npy'))
    gt = _to_cdhw(np.load(data_dir / f'{args.ref:02d}_to_{args.tgt:02d}_pair.npy').astype(np.float32))
    mask = (np.load(data_dir / 'Mask_Lung.npy') > 0).astype(np.float32)

    g_mlp = UNetFiLM(im_size=128, n_phases=10, int_steps=6).to(device)
    g_mlp.load_state_dict(torch.load(args.ckpt_phase_mlp, map_location=device))
    g_mlp.eval()
    pred_mlp = _predict(g_mlp, ct_r, args.ref, args.tgt, device)

    g_wd = UNetFiLM(im_size=128, n_phases=10, int_steps=6).to(device)
    g_wd.load_state_dict(torch.load(args.ckpt_warp_d, map_location=device))
    g_wd.eval()
    pred_wd = _predict(g_wd, ct_r, args.ref, args.tgt, device)

    fields = {
        'Elastix (B-spline)': gt,
        'phase-MLP (SVF)': pred_mlp,
        'warp-D (SVF)': pred_wd,
    }

    mags = {k: show_mag_slice(v, cfg) for k, v in fields.items()}
    hps = {k: show_ct_slice(_highpass_mag(v, sigma=args.hp_sigma) * mask, cfg) for k, v in fields.items()}
    psds = {k: _radial_psd(v, mask) for k, v in fields.items()}

    vmax_mag = float(np.percentile(np.concatenate([m.ravel() for m in mags.values()]), 99))
    vmax_hp = float(np.percentile(np.concatenate([h[h > 0].ravel() for h in hps.values() if (h > 0).any()]), 99))
    vmax_mag = max(vmax_mag, 1e-3)
    vmax_hp = max(vmax_hp, 1e-3)

    fig = plt.figure(figsize=(12.5, 9.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.15], hspace=0.32, wspace=0.22)

    for col, (name, _) in enumerate(fields.items()):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(mags[name], cmap='magma', origin='upper', aspect='equal', vmin=0, vmax=vmax_mag)
        ax.set_title(f'|DVF|  {name}', fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('|u| (mm)')

        ax2 = fig.add_subplot(gs[1, col])
        im2 = ax2.imshow(hps[name], cmap='inferno', origin='upper', aspect='equal', vmin=0, vmax=vmax_hp)
        ax2.set_title(f'high-pass texture  σ={args.hp_sigma:g}', fontsize=11)
        ax2.set_xticks([]); ax2.set_yticks([])
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04).set_label('|u−blur(u)| (mm)')

    ax_psd = fig.add_subplot(gs[2, :])
    styles = {
        'Elastix (B-spline)': ('#1f4e79', '-', 2.4),
        'phase-MLP (SVF)': ('#c0392b', '--', 2.0),
        'warp-D (SVF)': ('#d68910', ':', 2.2),
    }
    for name, (freq, psd) in psds.items():
        color, ls, lw = styles[name]
        # normalize by low-frequency power so shape (not absolute scale) is comparable
        psd_n = psd / (psd[:3].mean() + 1e-12)
        ax_psd.semilogy(freq, psd_n + 1e-16, color=color, ls=ls, lw=lw, label=name)

    ax_psd.set_xlabel('normalized spatial frequency (0 = DC, 1 ≈ Nyquist/2)')
    ax_psd.set_ylabel('radial PSD  (norm. to low-freq)')
    ax_psd.set_title(
        'Lung-masked radial power spectrum of DVF — Elastix is smoother; '
        'SVF models keep more mid/high-frequency energy'
    )
    ax_psd.legend(loc='upper right', frameon=False)
    ax_psd.grid(True, which='both', alpha=0.25)
    ax_psd.set_xlim(0, 1)

    tag = f'{args.ref:02d}_to_{args.tgt:02d}'
    fig.suptitle(
        f'DVF spectral / texture cue  |  pair {tag}  |  '
        f'View 1 slice {cfg.slice_index}  |  why 4-ch [ref, DVF] D can cheat',
        fontsize=13,
        y=0.995,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved → {out}')


if __name__ == '__main__':
    main()
