#!/usr/bin/env python3
"""Fine-tune frozen-copy of TCIA3 ep100 with MAE + SI under-move hinge.

Does NOT modify TCIA3 checkpoints. Writes only under TCIA3_magFT/.

  cd PopulationStudy/ClinicalExperiments/Grid160/TCIA3_magFT
  LEARN-GUI/.venv/bin/python scripts/train_finetune_mag.py --gpu 0 --epochs 30 --lr 3e-5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

T = Path(__file__).resolve().parents[1]
E1 = T.parent / "Experiment1"
T3 = T.parent / "TCIA3"

import importlib.util


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_loss_mod = _load("magft_losses", T / "losses" / "losses.py")
DVFMAEUnderMoveLoss = _loss_mod.DVFMAEUnderMoveLoss

if str(E1) not in sys.path:
    sys.path.insert(0, str(E1))
from networks.generator_crb_dec import UNetCRBDecoder  # noqa: E402
from utilities.seed_utils import set_seeds  # noqa: E402

_fd = _load("magft_fast_dataset", T / "scripts" / "fast_dataset.py")
FastMmapPhasePairDataset = _fd.FastMmapPhasePairDataset

warnings.filterwarnings("ignore")

INIT_CKPT = T3 / "DecoderCRB/checkpoints/epoch_100.pt"
CKPT_DIR = T / "DecoderCRB/checkpoints"
WEIGHTS_DIR = T / "DecoderCRB/weights"
PLOTS_DIR = T / "DecoderCRB/plots"
FILENAME = "crb_dec_mae_si_undermove_ft_tcia3ep100"


def load_manifest() -> dict:
    media = Path(
        "/media/abhishek/3CCA3CADCA3C6574/TCIA_4D-Lung/synth_g160_r3/pooled/manifest.json"
    )
    if media.is_file():
        return json.loads(media.read_text())
    p = T / "data" / "manifest.json"
    if p.is_file():
        return json.loads(p.read_text())
    raise FileNotFoundError("manifest.json missing")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init-ckpt", type=Path, default=INIT_CKPT)
    ap.add_argument("--lambda-under", type=float, default=1.0)
    ap.add_argument("--under-eps", type=float, default=0.25)
    ap.add_argument("--under-mode", choices=("si", "vec"), default="si")
    ap.add_argument("--patches-train", type=int, default=8)
    ap.add_argument("--patches-val", type=int, default=4)
    ap.add_argument("--save-every", type=int, default=1)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    set_seeds(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    init_ckpt = args.init_ckpt.resolve()
    if not init_ckpt.is_file():
        raise SystemExit(f"missing init ckpt (will not create): {init_ckpt}")
    # Safety: refuse to write into TCIA3
    if init_ckpt.is_relative_to(T3 / "DecoderCRB"):
        print(f"[magFT] READ-ONLY init ← {init_ckpt}", flush=True)
    else:
        print(f"[magFT] init ← {init_ckpt}", flush=True)

    man = load_manifest()
    im_dir = man["pooled_train_dir"]
    trainset = FastMmapPhasePairDataset(
        im_dir=im_dir,
        pair_files=man["train_pairs"],
        im_size=64,
        random_crop=True,
        patches_per_pair=args.patches_train,
        fov_aug=True,
        aug_seed=args.seed,
    )
    valset = FastMmapPhasePairDataset(
        im_dir=im_dir,
        pair_files=man["val_pairs"],
        im_size=64,
        random_crop=False,
        patches_per_pair=args.patches_val,
        fov_aug=False,
    )
    g_dl = torch.Generator().manual_seed(args.seed)
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, generator=g_dl)
    valloader = DataLoader(valset, batch_size=1, shuffle=False)

    g = UNetCRBDecoder(im_size=64, n_phases=10).to(device)
    raw = torch.load(str(init_ckpt), map_location=device, weights_only=False)
    state = raw.get("generator", raw.get("state_dict", raw)) if isinstance(raw, dict) else raw
    # epoch_*.pt may wrap as {"generator": ...} or bare state_dict
    if isinstance(state, dict) and any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    missing = g.load_state_dict(state, strict=False)
    print(f"[magFT] loaded init | missing={missing.missing_keys[:3]} unexpected={missing.unexpected_keys[:3]}", flush=True)

    crit = DVFMAEUnderMoveLoss(
        lambda_under=args.lambda_under,
        under_eps=args.under_eps,
        mode=args.under_mode,
    )
    opt = optim.Adam(g.parameters(), lr=args.lr)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    (T / "logs").mkdir(exist_ok=True)

    # Never write into the original TCIA3 tree
    # (TCIA3_magFT also starts with "TCIA3" — do not use startswith on the parent path)
    t3_dec = (T3 / "DecoderCRB").resolve()
    ckpt_res = CKPT_DIR.resolve()
    if ckpt_res == t3_dec or t3_dec in ckpt_res.parents:
        raise SystemExit(f"refuse writing into TCIA3: {CKPT_DIR}")

    print(
        f"[magFT] device={device} | MAE+{args.under_mode}-under λ={args.lambda_under} "
        f"eps={args.under_eps} | ep={args.epochs} lr={args.lr} | "
        f"train={len(trainset)} val={len(valset)}",
        flush=True,
    )
    print(f"[magFT] checkpoints → {CKPT_DIR}  (TCIA3 untouched)", flush=True)

    train_hist, val_hist, mae_hist, under_hist = [], [], [], []
    best_val = float("inf")
    best_path = WEIGHTS_DIR / f"{FILENAME}_generator.pth"
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        g.train()
        tr = tr_mae = tr_u = 0.0
        for data in trainloader:
            ref = data["reference_ct"].to(device)
            mask = data["lung_mask"].to(device)
            rp = data["ref_phase"].to(device)
            tp = data["target_phase"].to(device)
            gt = data["target_dvf"].to(device)
            pred = g(ref, rp, tp)
            opt.zero_grad(set_to_none=True)
            total, l_mae, l_u = crit.loss_parts(gt, pred, mask)
            total.backward()
            opt.step()
            tr += float(total.item())
            tr_mae += float(l_mae.item())
            tr_u += float(l_u.item())

        g.eval()
        va = 0.0
        with torch.no_grad():
            for data in valloader:
                ref = data["reference_ct"].to(device)
                mask = data["lung_mask"].to(device)
                rp = data["ref_phase"].to(device)
                tp = data["target_phase"].to(device)
                gt = data["target_dvf"].to(device)
                pred = g(ref, rp, tp)
                va += float(crit.loss(gt, pred, mask).item())

        n_tr, n_va = max(len(trainset), 1), max(len(valset), 1)
        tr_m, va_m = tr / n_tr, va / n_va
        train_hist.append(tr_m)
        val_hist.append(va_m)
        mae_hist.append(tr_mae / n_tr)
        under_hist.append(tr_u / n_tr)

        elapsed = time.time() - t0
        mark = ""
        if va_m < best_val:
            best_val = va_m
            mark = " *best*"
            torch.save(g.state_dict(), str(best_path))

        if ep % args.save_every == 0:
            torch.save(
                {
                    "generator": g.state_dict(),
                    "epoch": ep,
                    "val_loss": va_m,
                    "init_ckpt": str(init_ckpt),
                    "lambda_under": args.lambda_under,
                    "under_mode": args.under_mode,
                    "lr": args.lr,
                },
                str(CKPT_DIR / f"epoch_{ep:03d}.pt"),
            )
            torch.save(
                {"generator": g.state_dict(), "epoch": ep, "val_loss": va_m},
                str(CKPT_DIR / "latest.pt"),
            )

        print(
            f"Epoch {ep}/{args.epochs} | train {tr_m:.4f} (mae {mae_hist[-1]:.4f} "
            f"under {under_hist[-1]:.4f}) | val {va_m:.4f} | "
            f"{elapsed/60:.1f} min{mark}",
            flush=True,
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        xs = np.arange(1, ep + 1)
        ax.plot(xs, train_hist, label="train total")
        ax.plot(xs, val_hist, label="val total")
        ax.plot(xs, mae_hist, "--", label="train mae")
        ax.plot(xs, under_hist, "--", label="train under")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title(FILENAME)
        ax.legend()
        fig.tight_layout()
        fig.savefig(str(PLOTS_DIR / f"{FILENAME}.png"), dpi=120)
        plt.close(fig)

    hist = {
        "train": train_hist,
        "val": val_hist,
        "train_mae": mae_hist,
        "train_under": under_hist,
        "best_val": best_val,
        "init_ckpt": str(init_ckpt),
        "epochs": args.epochs,
        "lr": args.lr,
        "lambda_under": args.lambda_under,
        "under_mode": args.under_mode,
    }
    (PLOTS_DIR / f"{FILENAME}_history.json").write_text(json.dumps(hist, indent=2) + "\n")
    print(f"[magFT] done best_val={best_val:.4f} → {best_path}", flush=True)
    print(f"[magFT] TCIA3 epoch_100.pt unchanged: {init_ckpt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
