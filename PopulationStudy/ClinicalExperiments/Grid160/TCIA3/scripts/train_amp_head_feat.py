#!/usr/bin/env python3
"""Feature AmpHead: mid-CT + |û| stats → scale a (frozen TCIA3 Decoder).

Network (tiny MLP, ~10k params)
  inputs (8):
    φ_ref/9, φ_tgt/9
    log1p(p95|û|), log1p(mean|û|), log1p(p95|û_y|), log1p(mean|û_y|)
      — û_y = channel 1 ≈ SI in our crops
    lung_frac, mean_μ_in_lung   — mid-CT anatomy cues
  AmpHead: Linear(8→64)→ReLU→Linear(64→64)→ReLU→Linear(64→1)
  a = 0.8 + 1.7·sigmoid(z)

Target a* (train only, from Elastix): clip(p95_lung|El| / p95_lung|û|, 0.8, 2.5)

At inference: same features from mid-CT + one Decoder forward — no other-phase CT,
no Elastix, no landmarks.

  cd PopulationStudy/ClinicalExperiments/Grid160/TCIA3
  LEARN-GUI/.venv/bin/python scripts/train_amp_head_feat.py --gpu 0 --epochs 100 --reuse-cache
"""
from __future__ import annotations

import argparse
import json
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
from torch.utils.data import DataLoader, Dataset

T3 = Path(__file__).resolve().parents[1]
E1 = T3.parent / "Experiment1"
NET = T3.parent.parent.parent / "InitialExperiments" / "Experiment6"
sys.path.insert(0, str(T3))
sys.path.insert(0, str(E1))

import importlib.util

_fd_spec = importlib.util.spec_from_file_location(
    "tcia3_fast_dataset", T3 / "scripts" / "fast_dataset.py"
)
_fd = importlib.util.module_from_spec(_fd_spec)
assert _fd_spec.loader is not None
_fd_spec.loader.exec_module(_fd)
FastMmapPhasePairDataset = _fd.FastMmapPhasePairDataset

A_MIN, A_MAX = 0.8, 2.5
N_FEAT = 8
DEFAULT_CKPT = T3 / "DecoderCRB/checkpoints/epoch_100.pt"
OUT_DIR = T3 / "AmpHeadFeat"


def _ensure_net_path():
    if str(NET) not in sys.path:
        sys.path.append(str(NET))


class FeatAmpHead(nn.Module):
    """MLP on handcrafted mid-CT + |û| features → a ∈ [A_MIN, A_MAX]."""

    def __init__(self, n_in: int = N_FEAT, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        z = self.net(feat).squeeze(-1)
        return A_MIN + (A_MAX - A_MIN) * torch.sigmoid(z)


class FeatAmpDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return {
            "feat": torch.tensor(r["feat"], dtype=torch.float32),
            "a_star": torch.tensor(r["a_star"], dtype=torch.float32),
        }


def p95_mag(dvf: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mag = torch.linalg.vector_norm(dvf, dim=1)
    m = mask[:, 0] > 0.5
    vals = mag[m]
    if vals.numel() < 16:
        vals = mag.reshape(-1)
    return torch.quantile(vals.float(), 0.95)


def masked_stats(vol: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """vol (1,1,D,H,W) or (1,D,H,W); returns (mean, frac)."""
    if vol.ndim == 5:
        vol = vol[:, 0]
    m = mask[:, 0] > 0.5
    frac = m.float().mean()
    if m.any():
        mean = vol[m].float().mean()
    else:
        mean = vol.float().mean()
    return mean, frac


def chan_p95_mean(dvf: torch.Tensor, mask: torch.Tensor, ch: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Absolute component stats on lung."""
    v = dvf[:, ch].abs()
    m = mask[:, 0] > 0.5
    vals = v[m]
    if vals.numel() < 16:
        vals = v.reshape(-1)
    return torch.quantile(vals.float(), 0.95), vals.float().mean()


def feature_vector(
    ref_ct: torch.Tensor,
    mask: torch.Tensor,
    u_hat: torch.Tensor,
    ref_phase: int,
    tgt_phase: int,
) -> list[float]:
    """Build the 8-D inference-time feature vector (CPU floats)."""
    p95 = float(p95_mag(u_hat, mask).item())
    mag = torch.linalg.vector_norm(u_hat, dim=1)
    m = mask[:, 0] > 0.5
    mean_mag = float(mag[m].float().mean().item()) if m.any() else float(mag.mean().item())
    si_p95, si_mean = chan_p95_mean(u_hat, mask, ch=1)
    ct_mean, lung_frac = masked_stats(ref_ct, mask)
    return [
        ref_phase / 9.0,
        tgt_phase / 9.0,
        float(np.log1p(p95)),
        float(np.log1p(mean_mag)),
        float(np.log1p(si_p95.item())),
        float(np.log1p(si_mean.item())),
        float(lung_frac.item()),
        float(ct_mean.item()),
    ]


@torch.no_grad()
def precompute(
    g: nn.Module,
    pair_files: list[str],
    im_dir: str,
    device: torch.device,
    tag: str,
) -> list[dict]:
    ds = FastMmapPhasePairDataset(
        im_dir=im_dir,
        pair_files=pair_files,
        im_size=64,
        random_crop=False,
        patches_per_pair=1,
        fov_aug=False,
    )
    rows = []
    g.eval()
    for i in range(len(ds)):
        batch = ds[i]
        ref = batch["reference_ct"].unsqueeze(0).to(device)
        mask = batch["lung_mask"].unsqueeze(0).to(device)
        gt = batch["target_dvf"].unsqueeze(0).to(device)
        rp = int(batch["ref_phase"])
        tp = int(batch["target_phase"])
        u_hat = g(ref, torch.tensor([rp], device=device), torch.tensor([tp], device=device))
        p_el = p95_mag(gt, mask).clamp_min(1e-4)
        p_hat = p95_mag(u_hat, mask).clamp_min(1e-4)
        a_star = float((p_el / p_hat).clamp(A_MIN, A_MAX).item())
        feat = feature_vector(ref, mask, u_hat, rp, tp)
        rows.append(
            {
                "pair": pair_files[i],
                "ref_phase": rp,
                "target_phase": tp,
                "patient": batch["patient"],
                "a_star": a_star,
                "p95_el": float(p_el.item()),
                "p95_hat": float(p_hat.item()),
                "feat": feat,
            }
        )
        if (i + 1) % 500 == 0 or i == 0:
            print(
                f"  [{tag}] {i+1}/{len(ds)} a*={a_star:.3f} feat0..3={feat[:4]}",
                flush=True,
            )
    return rows


def load_frozen_decoder(ckpt: Path, device: torch.device) -> nn.Module:
    _ensure_net_path()
    from networks.generator_crb_dec import UNetCRBDecoder

    g = UNetCRBDecoder(im_size=64, n_phases=10)
    raw = torch.load(str(ckpt), map_location=device, weights_only=False)
    state = raw.get("generator", raw.get("state_dict", raw)) if isinstance(raw, dict) else raw
    g.load_state_dict(state, strict=True)
    g.to(device).eval()
    for p in g.parameters():
        p.requires_grad_(False)
    return g


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--reuse-cache", action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    media_man = Path(
        "/media/abhishek/3CCA3CADCA3C6574/TCIA_4D-Lung/synth_g160_r3/pooled/manifest.json"
    )
    man = json.loads(media_man.read_text()) if media_man.is_file() else json.loads(
        (T3 / "data" / "manifest.json").read_text()
    )
    im_dir = man["pooled_train_dir"]
    train_pairs, val_pairs = man["train_pairs"], man["val_pairs"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "logs").mkdir(exist_ok=True)
    cache_path = OUT_DIR / "feat_a_star_cache_tcia3_ep100.json"
    ckpt = args.ckpt.resolve()
    if not ckpt.is_file():
        raise SystemExit(f"missing decoder ckpt: {ckpt}")

    print(
        f"[FeatAmp] device={device} | MLP {N_FEAT}→{args.hidden}→{args.hidden}→1 | "
        f"a∈[{A_MIN},{A_MAX}] | ep={args.epochs} lr={args.lr}",
        flush=True,
    )

    if args.reuse_cache and cache_path.is_file():
        cache = json.loads(cache_path.read_text())
        train_rows, val_rows = cache["train"], cache["val"]
        print(f"[FeatAmp] reused {cache_path}", flush=True)
    else:
        g = load_frozen_decoder(ckpt, device)
        print("[FeatAmp] precomputing features + a* …", flush=True)
        t0 = time.time()
        train_rows = precompute(g, train_pairs, im_dir, device, "train")
        val_rows = precompute(g, val_pairs, im_dir, device, "val")
        cache_path.write_text(
            json.dumps(
                {
                    "ckpt": str(ckpt),
                    "n_feat": N_FEAT,
                    "feat_names": [
                        "phi_ref",
                        "phi_tgt",
                        "log1p_p95_hat",
                        "log1p_mean_hat",
                        "log1p_si_p95",
                        "log1p_si_mean",
                        "lung_frac",
                        "lung_mean_mu",
                    ],
                    "train": train_rows,
                    "val": val_rows,
                    "elapsed_s": time.time() - t0,
                }
            )
            + "\n"
        )
        print(f"[FeatAmp] cache → {cache_path} ({time.time()-t0:.0f}s)", flush=True)
        del g
        torch.cuda.empty_cache()

    a_tr = np.array([r["a_star"] for r in train_rows], float)
    print(
        f"[FeatAmp] a* train mean={a_tr.mean():.3f} std={a_tr.std():.3f} "
        f"[{a_tr.min():.2f},{a_tr.max():.2f}]",
        flush=True,
    )

    train_loader = DataLoader(
        FeatAmpDataset(train_rows), batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        FeatAmpDataset(val_rows), batch_size=args.batch_size, shuffle=False
    )

    head = FeatAmpHead(hidden=args.hidden).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    print(f"[FeatAmp] params={sum(p.numel() for p in head.parameters())}", flush=True)

    train_hist, val_hist = [], []
    best_val, best_path = float("inf"), OUT_DIR / "amp_head_feat_best.pt"
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        head.train()
        tr = []
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            a = head(batch["feat"].to(device))
            loss = torch.abs(a - batch["a_star"].to(device)).mean()
            loss.backward()
            opt.step()
            tr.append(float(loss.item()))
        head.eval()
        va = []
        with torch.no_grad():
            for batch in val_loader:
                a = head(batch["feat"].to(device))
                va.append(
                    float(torch.abs(a - batch["a_star"].to(device)).mean().item())
                )
        tr_m, va_m = float(np.mean(tr)), float(np.mean(va))
        train_hist.append(tr_m)
        val_hist.append(va_m)
        mark = ""
        if va_m < best_val:
            best_val = va_m
            mark = " *best*"
            torch.save(
                {
                    "amp_head": head.state_dict(),
                    "hidden": args.hidden,
                    "n_feat": N_FEAT,
                    "a_min": A_MIN,
                    "a_max": A_MAX,
                    "epoch": ep,
                    "val_l1": va_m,
                    "decoder_ckpt": str(ckpt),
                    "kind": "feat_mlp",
                },
                best_path,
            )
        if ep == 1 or ep % 10 == 0 or mark:
            print(
                f"Epoch {ep}/{args.epochs} | train {tr_m:.4f} | val {va_m:.4f} | "
                f"{(time.time()-t0)/60:.1f} min{mark}",
                flush=True,
            )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_hist, label="train")
    ax.plot(val_hist, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("L1(a, a*)")
    ax.set_title("FeatAmpHead mid-CT+|û| (frozen TCIA3 ep100)")
    ax.legend()
    fig.tight_layout()
    plot_path = OUT_DIR / "amp_head_feat_loss.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    (OUT_DIR / "amp_head_feat_history.json").write_text(
        json.dumps(
            {
                "train_l1": train_hist,
                "val_l1": val_hist,
                "best_val_l1": best_val,
                "epochs": args.epochs,
                "lr": args.lr,
                "kind": "feat_mlp",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[FeatAmp] best_val_L1={best_val:.4f} → {best_path}", flush=True)
    print(f"[FeatAmp] plot {plot_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
