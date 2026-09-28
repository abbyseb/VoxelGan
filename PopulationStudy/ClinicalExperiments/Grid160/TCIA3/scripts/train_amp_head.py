#!/usr/bin/env python3
"""Phase-only AmpHead on frozen TCIA3 Decoder.

Architecture
  AmpHead: Linear(2→64)→ReLU→Linear(64→64)→ReLU→Linear(64→1)
  a = 0.8 + 1.7 * sigmoid(z)   ∈ [0.8, 2.5]

Training (TCIA pooled pairs only — not DIR)
  1) Freeze G = TCIA3 epoch_100
  2) Precompute a* = clip(p95_lung(|u_El|) / p95_lung(|û|), 0.8, 2.5) once per pair
  3) Train AmpHead: (φ_ref/9, φ_tgt/9) → a   vs a*   (L1)
  Defaults: epochs=30, lr=1e-3, Adam, batch=256

  cd PopulationStudy/ClinicalExperiments/Grid160/TCIA3
  LEARN-GUI/.venv/bin/python scripts/train_amp_head.py --gpu 0
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
sys.path.insert(0, str(E1))  # FOVAug dataset lives here
# Do NOT put NET on path yet — it shadows utilities.dataset with Experiment6

import importlib.util

_fd_spec = importlib.util.spec_from_file_location(
    "tcia3_fast_dataset", T3 / "scripts" / "fast_dataset.py"
)
_fd = importlib.util.module_from_spec(_fd_spec)
assert _fd_spec.loader is not None
# Ensure Experiment1 utilities resolve while loading fast_dataset
if str(E1) not in sys.path:
    sys.path.insert(0, str(E1))
_fd_spec.loader.exec_module(_fd)
FastMmapPhasePairDataset = _fd.FastMmapPhasePairDataset


def _ensure_net_path():
    if str(NET) not in sys.path:
        sys.path.append(str(NET))  # append so it does not shadow E1 utilities


A_MIN, A_MAX = 0.8, 2.5
DEFAULT_CKPT = T3 / "DecoderCRB/checkpoints/epoch_100.pt"
OUT_DIR = T3 / "AmpHead"


class AmpHead(nn.Module):
    """Tiny MLP: normalized phases → scale a ∈ [A_MIN, A_MAX]."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, ref_phase: torch.Tensor, tgt_phase: torch.Tensor) -> torch.Tensor:
        # phases are long ints 0..9
        x = torch.stack(
            [ref_phase.float() / 9.0, tgt_phase.float() / 9.0], dim=-1
        )
        z = self.net(x).squeeze(-1)
        return A_MIN + (A_MAX - A_MIN) * torch.sigmoid(z)


class PhaseAmpDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return {
            "ref_phase": torch.tensor(r["ref_phase"], dtype=torch.long),
            "target_phase": torch.tensor(r["target_phase"], dtype=torch.long),
            "a_star": torch.tensor(r["a_star"], dtype=torch.float32),
        }


def p95_mag(dvf: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """dvf (1,3,D,H,W), mask (1,1,D,H,W) → scalar p95 of |u| on lung."""
    mag = torch.linalg.vector_norm(dvf, dim=1)  # (1,D,H,W)
    m = mask[:, 0] > 0.5
    vals = mag[m]
    if vals.numel() < 16:
        vals = mag.reshape(-1)
    return torch.quantile(vals.float(), 0.95)


@torch.no_grad()
def precompute_a_star(
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
        rp = batch["ref_phase"].unsqueeze(0).to(device)
        tp = batch["target_phase"].unsqueeze(0).to(device)
        u_hat = g(ref, rp, tp)
        p_el = p95_mag(gt, mask).clamp_min(1e-4)
        p_hat = p95_mag(u_hat, mask).clamp_min(1e-4)
        a_star = float((p_el / p_hat).clamp(A_MIN, A_MAX).item())
        rows.append(
            {
                "pair": pair_files[i],
                "ref_phase": int(batch["ref_phase"]),
                "target_phase": int(batch["target_phase"]),
                "patient": batch["patient"],
                "a_star": a_star,
                "p95_el": float(p_el.item()),
                "p95_hat": float(p_hat.item()),
            }
        )
        if (i + 1) % 500 == 0 or i == 0:
            print(f"  [{tag}] precompute {i+1}/{len(ds)} last a*={a_star:.3f}", flush=True)
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
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--reuse-cache", action="store_true", help="Reuse cached a* JSON if present")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    man = json.loads((T3 / "data" / "manifest.json").read_text())
    # Prefer SSD pooled path from media manifest if linked empty — read media
    media_man = Path("/media/abhishek/3CCA3CADCA3C6574/TCIA_4D-Lung/synth_g160_r3/pooled/manifest.json")
    if media_man.is_file():
        man = json.loads(media_man.read_text())
    im_dir = man["pooled_train_dir"]
    train_pairs = man["train_pairs"]
    val_pairs = man["val_pairs"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OUT_DIR / "a_star_cache_tcia3_ep100.json"
    ckpt = args.ckpt.resolve()
    if not ckpt.is_file():
        raise SystemExit(f"missing decoder ckpt: {ckpt}")

    print(
        f"[AmpHead] device={device} | MLP 2→{args.hidden}→{args.hidden}→1 | "
        f"a∈[{A_MIN},{A_MAX}] | epochs={args.epochs} lr={args.lr} bs={args.batch_size}",
        flush=True,
    )
    print(f"[AmpHead] freeze G ← {ckpt.name} | train pairs={len(train_pairs)} val={len(val_pairs)}", flush=True)

    if args.reuse_cache and cache_path.is_file():
        cache = json.loads(cache_path.read_text())
        train_rows, val_rows = cache["train"], cache["val"]
        print(f"[AmpHead] reused cache {cache_path}", flush=True)
    else:
        g = load_frozen_decoder(ckpt, device)
        print("[AmpHead] precomputing a* (one G forward / pair)...", flush=True)
        t0 = time.time()
        train_rows = precompute_a_star(g, train_pairs, im_dir, device, "train")
        val_rows = precompute_a_star(g, val_pairs, im_dir, device, "val")
        cache = {
            "ckpt": str(ckpt),
            "a_min": A_MIN,
            "a_max": A_MAX,
            "train": train_rows,
            "val": val_rows,
            "elapsed_s": time.time() - t0,
        }
        cache_path.write_text(json.dumps(cache) + "\n")
        print(f"[AmpHead] cached a* → {cache_path} ({time.time()-t0:.0f}s)", flush=True)
        del g
        torch.cuda.empty_cache()

    a_tr = np.array([r["a_star"] for r in train_rows], float)
    print(
        f"[AmpHead] a* train mean={a_tr.mean():.3f} std={a_tr.std():.3f} "
        f"min={a_tr.min():.2f} max={a_tr.max():.2f}",
        flush=True,
    )

    train_loader = DataLoader(
        PhaseAmpDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        PhaseAmpDataset(val_rows),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    head = AmpHead(hidden=args.hidden).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[AmpHead] params={n_params}", flush=True)

    train_hist, val_hist = [], []
    best_val = float("inf")
    best_path = OUT_DIR / "amp_head_best.pt"
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        head.train()
        tr = []
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            a = head(batch["ref_phase"].to(device), batch["target_phase"].to(device))
            loss = torch.abs(a - batch["a_star"].to(device)).mean()
            loss.backward()
            opt.step()
            tr.append(float(loss.item()))

        head.eval()
        va = []
        with torch.no_grad():
            for batch in val_loader:
                a = head(batch["ref_phase"].to(device), batch["target_phase"].to(device))
                va.append(float(torch.abs(a - batch["a_star"].to(device)).mean().item()))

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
                    "a_min": A_MIN,
                    "a_max": A_MAX,
                    "epoch": ep,
                    "val_l1": va_m,
                    "decoder_ckpt": str(ckpt),
                    "lr": args.lr,
                    "epochs": args.epochs,
                },
                best_path,
            )
        print(
            f"Epoch {ep}/{args.epochs} | train L1(a): {tr_m:.4f} | val L1(a): {va_m:.4f} | "
            f"{(time.time()-t0)/60:.1f} min{mark}",
            flush=True,
        )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_hist, label="train")
    ax.plot(val_hist, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("L1(a, a*)")
    ax.set_title("AmpHead phase-only (frozen TCIA3 ep100)")
    ax.legend()
    fig.tight_layout()
    plot_path = OUT_DIR / "amp_head_loss.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)

    hist_path = OUT_DIR / "amp_head_history.json"
    hist_path.write_text(
        json.dumps(
            {
                "train_l1": train_hist,
                "val_l1": val_hist,
                "best_val_l1": best_val,
                "epochs": args.epochs,
                "lr": args.lr,
                "hidden": args.hidden,
                "batch_size": args.batch_size,
                "decoder_ckpt": str(ckpt),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[AmpHead] done best_val_L1={best_val:.4f} → {best_path}", flush=True)
    print(f"[AmpHead] plot {plot_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
