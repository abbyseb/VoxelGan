#!/usr/bin/env python3
"""TCIA3_scaleMap: regional multiplier on a frozen TCIA3 decoder.

The head sees only the CT and |u_pred|. It outputs s(x) in [0.5, 2.5] on an
8³ grid (upsampled), and the corrected field is s·u. Direction is unchanged.

Training pairs are TCIA3's val split: the decoder was not updated on them.
No DIR case, Elastix-at-test, or landmarks are used. Loss is lung-masked L1
between s·u and the TCIA Elastix field (not a magnitude ratio).

The last layer is initialized so s = 1 everywhere.

  cd PopulationStudy/ClinicalExperiments/Grid160/TCIA3_scaleMap
  CUDA_VISIBLE_DEVICES=0 python scripts/train_scale_map.py --gpu 0
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parents[1]
T3 = HERE.parent / "TCIA3"
sys.path.insert(0, str(T3))

_fd_spec = importlib.util.spec_from_file_location(
    "tcia3_fast_dataset", T3 / "scripts" / "fast_dataset.py"
)
_fd = importlib.util.module_from_spec(_fd_spec)
assert _fd_spec.loader is not None
_fd_spec.loader.exec_module(_fd)
FastMmapPhasePairDataset = _fd.FastMmapPhasePairDataset

DECODER_CKPT = T3 / "DecoderCRB" / "checkpoints" / "epoch_100.pt"
OUT = HERE / "checkpoints"
PLOTS = HERE / "plots"
S_LO, S_HI = 0.5, 2.5


class ScaleMapHead(nn.Module):
    """2-channel 64³ → smooth multiplier map. Bottleneck is 8³."""

    def __init__(self, in_ch: int = 2):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(in_ch, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.mid = nn.Sequential(
            nn.Conv3d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.to_scale = nn.Conv3d(32, 1, 1)
        nn.init.zeros_(self.to_scale.weight)
        # sigmoid(-ln 3) = 0.25 → s = 0.5 + 2.0*0.25 = 1
        nn.init.constant_(self.to_scale.bias, -math.log(3.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.to_scale(self.mid(self.down(x)))
        z = F.interpolate(z, size=x.shape[-3:], mode="trilinear", align_corners=False)
        return S_LO + (S_HI - S_LO) * torch.sigmoid(z)


def lung_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask if mask.shape[1] == pred.shape[1] else mask.expand_as(pred)
    err = (pred - target).abs() * m
    return err.sum() / m.sum().clamp_min(1.0)


def lung_mean_s(s: torch.Tensor, mask: torch.Tensor) -> float:
    m = mask[:, :1] > 0.5
    if m.any():
        return float(s[:, :1][m].mean().item())
    return float(s.mean().item())


def load_decoder(device: torch.device) -> nn.Module:
    from networks.generator_crb_dec import UNetCRBDecoder

    g = UNetCRBDecoder(im_size=64, n_phases=10)
    raw = torch.load(str(DECODER_CKPT), map_location=device, weights_only=False)
    state = raw.get("generator", raw.get("state_dict", raw)) if isinstance(raw, dict) else raw
    g.load_state_dict(state, strict=True)
    g.to(device).eval()
    for p in g.parameters():
        p.requires_grad_(False)
    return g


def split_val_pairs(val_pairs: list[str], val_fraction: float = 0.2):
    patients = sorted({p.split("_")[0] for p in val_pairs})
    n_val = max(1, int(round(len(patients) * val_fraction)))
    val_patients = set(patients[-n_val:])
    head_val = [p for p in val_pairs if p.split("_")[0] in val_patients]
    head_train = [p for p in val_pairs if p.split("_")[0] not in val_patients]
    return head_train, head_val, sorted(val_patients)


def make_loader(im_dir, pairs, patches, random_crop, batch, workers, shuffle):
    ds = FastMmapPhasePairDataset(
        im_dir=im_dir,
        pair_files=pairs,
        im_size=64,
        random_crop=random_crop,
        patches_per_pair=patches,
        fov_aug=False,
    )
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def run_epoch(head, decoder, loader, device, opt=None) -> tuple[float, float]:
    train = opt is not None
    head.train(train)
    losses, scales = [], []
    for batch in loader:
        ct = batch["reference_ct"].to(device, non_blocking=True)
        mask = batch["lung_mask"].to(device, non_blocking=True)
        gt = batch["target_dvf"].to(device, non_blocking=True)
        rp = batch["ref_phase"].to(device, non_blocking=True)
        tp = batch["target_phase"].to(device, non_blocking=True)
        with torch.no_grad():
            u = decoder(ct, rp, tp)
        mag = torch.linalg.vector_norm(u, dim=1, keepdim=True)
        s = head(torch.cat([ct, mag], dim=1))
        loss = lung_l1(s * u, gt, mask)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        losses.append(float(loss.item()))
        scales.append(lung_mean_s(s.detach(), mask))
    return float(np.mean(losses)), float(np.mean(scales))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not DECODER_CKPT.is_file():
        raise SystemExit(f"missing frozen decoder: {DECODER_CKPT}")

    man = json.loads((T3 / "data" / "manifest.json").read_text())
    head_train, head_val, val_patients = split_val_pairs(man["val_pairs"])
    print(
        f"[scaleMap] device={device} | frozen {DECODER_CKPT.name} | "
        f"head-train pairs={len(head_train)} head-val pairs={len(head_val)} "
        f"val patients={len(val_patients)} (pair holdout; patients were in TCIA3 train)",
        flush=True,
    )

    train_loader = make_loader(
        man["pooled_train_dir"], head_train, patches=4, random_crop=True,
        batch=args.batch_size, workers=args.workers, shuffle=True,
    )
    val_loader = make_loader(
        man["pooled_train_dir"], head_val, patches=2, random_crop=False,
        batch=args.batch_size, workers=max(1, args.workers // 2), shuffle=False,
    )

    decoder = load_decoder(device)
    head = ScaleMapHead().to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[scaleMap] head params={n_params} | s init should be ~1", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_path = OUT / "scale_map_head_best.pt"
    hist = {"train_l1": [], "val_l1": [], "train_s": [], "val_s": []}
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tr, tr_s = run_epoch(head, decoder, train_loader, device, opt)
        va, va_s = run_epoch(head, decoder, val_loader, device, opt=None)
        hist["train_l1"].append(tr)
        hist["val_l1"].append(va)
        hist["train_s"].append(tr_s)
        hist["val_s"].append(va_s)
        mark = ""
        if va < best_val:
            best_val = va
            mark = " *best*"
            torch.save(
                {
                    "scale_head": head.state_dict(),
                    "epoch": ep,
                    "val_l1": va,
                    "s_lo": S_LO,
                    "s_hi": S_HI,
                    "decoder_ckpt": str(DECODER_CKPT),
                    "kind": "scale_map_8cubed",
                },
                best_path,
            )
        print(
            f"Epoch {ep}/{args.epochs} | train L1 {tr:.4f} | val L1 {va:.4f} | "
            f"mean s train/val {tr_s:.3f}/{va_s:.3f} | {(time.time()-t0)/60:.1f} min{mark}",
            flush=True,
        )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hist["train_l1"], label="train L1")
    ax.plot(hist["val_l1"], label="val L1")
    ax.set_xlabel("epoch")
    ax.set_ylabel("lung-masked L1 of s·u vs Elastix")
    ax.set_title("TCIA3_scaleMap")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "scale_map_loss.png", dpi=120)
    plt.close(fig)
    (OUT / "history.json").write_text(json.dumps(hist, indent=2) + "\n")
    print(f"[scaleMap] best_val_L1={best_val:.4f} → {best_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
