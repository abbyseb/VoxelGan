"""Jacobian folding QC for all Dan 2.0 generators.

Reports % voxels with det(I + ∇u) < 0 on full volume and inside lung mask.
Same finite-diff formula as scripts/qc_leave_phase_out.py.

Usage (from Dan2.0/):
  python scripts/qc_jacobian_all.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

DAN20 = Path(__file__).resolve().parents[1]
REPO = DAN20.parent


def _import_from(path: Path, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _clear_utilities():
    for k in list(sys.modules):
        if k == 'utilities' or k.startswith('utilities.') or k.startswith('generator_crb'):
            del sys.modules[k]


def load_model(kind: str, device):
    if kind == 'film':
        sys.path = [str(REPO)] + [p for p in sys.path if p != str(REPO)]
        _clear_utilities()
        mod = _import_from(REPO / 'utilities' / 'generator.py', '_jac_unet_film')
        return mod.UNetFiLM(im_size=128, n_phases=10, int_steps=6).to(device)

    prefer = [str(DAN20), str(DAN20 / 'DecoderCRB'), str(DAN20 / 'BothCRB'), str(DAN20 / 'DecoderCRBBot'), str(REPO)]
    sys.path = prefer + [p for p in sys.path if p not in set(prefer)]
    _clear_utilities()

    if kind == 'enc_crb':
        mod = _import_from(DAN20 / 'utilities' / 'generator_crb.py', '_jac_unet_crb')
        return mod.UNetCRB(im_size=128, n_phases=10).to(device)
    if kind == 'dec_crb':
        mod = _import_from(DAN20 / 'DecoderCRB' / 'generator_crb_dec.py', '_jac_unet_crb_dec')
        return mod.UNetCRBDecoder(im_size=128, n_phases=10).to(device)
    if kind == 'both_crb':
        mod = _import_from(DAN20 / 'BothCRB' / 'generator_crb_both.py', '_jac_unet_crb_both')
        return mod.UNetCRBBoth(im_size=128, n_phases=10).to(device)
    if kind == 'dec_bot':
        mod = _import_from(DAN20 / 'DecoderCRBBot' / 'generator_crb_dec_bot.py', '_jac_unet_crb_bot')
        return mod.UNetCRBDecoderBot(im_size=128, n_phases=10).to(device)
    raise ValueError(kind)


def neg_jacobian(dvf: np.ndarray):
    """dvf (3,D,H,W) → det map of I+∇u."""
    j00 = 1 + np.gradient(dvf[0], axis=2)
    j01 = np.gradient(dvf[0], axis=1)
    j02 = np.gradient(dvf[0], axis=0)
    j10 = np.gradient(dvf[1], axis=2)
    j11 = 1 + np.gradient(dvf[1], axis=1)
    j12 = np.gradient(dvf[1], axis=0)
    j20 = np.gradient(dvf[2], axis=2)
    j21 = np.gradient(dvf[2], axis=1)
    j22 = 1 + np.gradient(dvf[2], axis=0)
    return (
        j00 * (j11 * j22 - j12 * j21)
        - j01 * (j10 * j22 - j12 * j20)
        + j02 * (j10 * j21 - j11 * j20)
    )


def _norm_ct(x):
    x = x.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def list_pairs(data_dir: Path):
    pairs = []
    for p in sorted(data_dir.glob('*_pair.npy')):
        a, b = p.name.replace('_pair.npy', '').split('_to_')
        pairs.append((int(a), int(b)))
    return pairs


RUNS = [
    ('EncoderCRB/5_9', 'enc_crb', DAN20 / 'EncoderCRB/LeaveOut_5_9/weights/crb_enc_mse_lo_05_09_generator.pth', {5, 9}),
    ('EncoderCRB/3_6', 'enc_crb', DAN20 / 'EncoderCRB/LeaveOut_3_6/weights/crb_enc_mse_lo_03_06_generator.pth', {3, 6}),
    ('EncoderCRB/3_6_8', 'enc_crb', DAN20 / 'EncoderCRB/LeaveOut_3_6_8/weights/crb_enc_mse_lo_03_06_08_generator.pth', {3, 6, 8}),
    ('DecoderCRB/5_9', 'dec_crb', DAN20 / 'DecoderCRB/LeaveOut_5_9/weights/crb_dec_mse_lo_05_09_generator.pth', {5, 9}),
    ('DecoderCRB/3_6', 'dec_crb', DAN20 / 'DecoderCRB/LeaveOut_3_6/weights/crb_dec_mse_lo_03_06_generator.pth', {3, 6}),
    ('DecoderCRB/3_6_8', 'dec_crb', DAN20 / 'DecoderCRB/LeaveOut_3_6_8/weights/crb_dec_mse_lo_03_06_08_generator.pth', {3, 6, 8}),
    ('BothCRB/5_9', 'both_crb', DAN20 / 'BothCRB/LeaveOut_5_9/weights/crb_both_mse_lo_05_09_generator.pth', {5, 9}),
    ('BothCRB/3_6', 'both_crb', DAN20 / 'BothCRB/LeaveOut_3_6/weights/crb_both_mse_lo_03_06_generator.pth', {3, 6}),
    ('BothCRB/3_6_8', 'both_crb', DAN20 / 'BothCRB/LeaveOut_3_6_8/weights/crb_both_mse_lo_03_06_08_generator.pth', {3, 6, 8}),
    ('DecoderFiLM/5_9', 'film', DAN20 / 'DecoderFiLM/LeaveOut_5_9/weights/film_mse_lo_05_09_generator.pth', {5, 9}),
    ('DecoderFiLM/3_6', 'film', DAN20 / 'DecoderFiLM/LeaveOut_3_6/weights/film_mse_lo_03_06_generator.pth', {3, 6}),
    ('DecoderFiLM/3_6_8', 'film', DAN20 / 'DecoderFiLM/LeaveOut_3_6_8/weights/film_mse_lo_03_06_08_generator.pth', {3, 6, 8}),
    ('DecoderCRBBot/5_9', 'dec_bot', DAN20 / 'DecoderCRBBot/LeaveOut_5_9/weights/crb_dec_bot_mse_lo_05_09_generator.pth', {5, 9}),
    ('DecoderCRBBot/3_6', 'dec_bot', DAN20 / 'DecoderCRBBot/LeaveOut_3_6/weights/crb_dec_bot_mse_lo_03_06_generator.pth', {3, 6}),
    ('DecoderCRBBot/3_6_8', 'dec_bot', DAN20 / 'DecoderCRBBot/LeaveOut_3_6_8/weights/crb_dec_bot_mse_lo_03_06_08_generator.pth', {3, 6, 8}),
]


def eval_run(name, kind, ckpt, held, data_dir, mask, pairs, device):
    g = load_model(kind, device)
    g.load_state_dict(torch.load(ckpt, map_location=device))
    g.eval()

    all_full, all_lung, lo_full, lo_lung = [], [], [], []
    with torch.no_grad():
        for ref, tgt in pairs:
            if ref == tgt:
                continue
            ct = _norm_ct(np.load(data_dir / f'CT_{ref:02d}.npy'))
            ref_t = torch.from_numpy(ct)[None, None].to(device)
            pred = g(
                ref_t,
                torch.tensor([ref - 1], device=device),
                torch.tensor([tgt - 1], device=device),
            )[0].cpu().numpy()
            det = neg_jacobian(pred)
            full = 100.0 * float((det < 0).mean())
            lung = 100.0 * float((det[mask > 0.5] < 0).mean())
            all_full.append(full)
            all_lung.append(lung)
            if ref in held or tgt in held:
                lo_full.append(full)
                lo_lung.append(lung)
    del g
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return {
        'name': name,
        'n': len(all_full),
        'dir_full': float(np.mean(all_full)),
        'dir_lung': float(np.mean(all_lung)),
        'lo_full': float(np.mean(lo_full)),
        'lo_lung': float(np.mean(lo_lung)),
        'max_lung': float(np.max(all_lung)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default=str(DAN20 / 'jacobian_summary.tsv'))
    args = ap.parse_args()
    device = torch.device(args.device if args.device != 'cuda' or torch.cuda.is_available() else 'cpu')
    data_dir = REPO / 'data' / 'spare' / 'all'
    mask = (np.load(data_dir / 'Mask_Lung.npy') > 0).astype(np.float32)
    pairs = list_pairs(data_dir)
    print(f'device={device} pairs={len(pairs)} (directed={sum(1 for a, b in pairs if a != b)})')

    rows = []
    for name, kind, ckpt, held in RUNS:
        ckpt = Path(ckpt)
        if not ckpt.exists():
            print(f'[skip] {name} missing {ckpt}')
            continue
        print(f'[run] {name} ...', flush=True)
        r = eval_run(name, kind, ckpt, held, data_dir, mask, pairs, device)
        rows.append(r)
        print(
            f"  directed negJ% full={r['dir_full']:.3f} lung={r['dir_lung']:.3f} | "
            f"leave-out full={r['lo_full']:.3f} lung={r['lo_lung']:.3f} | max_lung={r['max_lung']:.3f}",
            flush=True,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        f.write('model\tn_directed\tnegJ_dir_full%\tnegJ_dir_lung%\tnegJ_loo_full%\tnegJ_loo_lung%\tmax_lung%\n')
        for r in rows:
            f.write(
                f"{r['name']}\t{r['n']}\t{r['dir_full']:.4f}\t{r['dir_lung']:.4f}\t"
                f"{r['lo_full']:.4f}\t{r['lo_lung']:.4f}\t{r['max_lung']:.4f}\n"
            )
    md = out.with_suffix('.md')
    with open(md, 'w') as f:
        f.write('# Dan 2.0 — negative Jacobian %\n\n')
        f.write('`negJ%` = fraction of voxels with `det(I+∇u) < 0` (finite-diff). Lower is better.\n\n')
        f.write('| Model | Dir full % | Dir lung % | Leave-out lung % | Max lung % |\n')
        f.write('|-------|------------|------------|------------------|------------|\n')
        for r in rows:
            f.write(
                f"| {r['name']} | {r['dir_full']:.3f} | {r['dir_lung']:.3f} | "
                f"{r['lo_lung']:.3f} | {r['max_lung']:.3f} |\n"
            )
    print(f'wrote {out} and {md}')


if __name__ == '__main__':
    main()
