#!/usr/bin/env python3
"""Fine-tune G160-A1 Decoder on DIR C1/C5/C8 Elastix pairs (low LR).

  cd "DIR EXPERIMENTS"
  CUDA_VISIBLE_DEVICES=1 LEARN_PY ... scripts/train_g160_dir_ft.py --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

DIR_EXP = Path(__file__).resolve().parents[1]
EXP = DIR_EXP / "experiments" / "G160_DIR_FT_C1C5C8"
G160_E1 = (
    DIR_EXP.parent
    / "PopulationStudy"
    / "ClinicalExperiments"
    / "Grid160"
    / "Experiment1"
)
INIT_CKPT = (
    G160_E1
    / "DecoderCRB"
    / "weights"
    / "crb_dec_mse_iso_g160_fov_full_generator.pth"
)

warnings.filterwarnings("ignore")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", type=int, default=0, help="index within CUDA_VISIBLE_DEVICES")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init-ckpt", type=Path, default=INIT_CKPT)
    ap.add_argument("--manifest", type=Path, default=EXP / "manifest.json")
    args = ap.parse_args()

    if str(G160_E1) not in sys.path:
        sys.path.insert(0, str(G160_E1))

    from losses.losses import DVFMSELoss
    from networks.generator_crb_dec import UNetCRBDecoder
    from utilities.dataset import FOVAugPhasePairDataset
    from utilities.seed_utils import set_seeds

    set_seeds(args.seed)
    man = json.loads(args.manifest.read_text())
    train_dir = Path(man["pooled_train_dir"])
    if not args.init_ckpt.is_file():
        raise SystemExit(f"Missing init ckpt: {args.init_ckpt}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    im_size = 64
    batch_size = 1
    trainset = FOVAugPhasePairDataset(
        im_dir=str(train_dir),
        pair_files=man["train_pairs"],
        im_size=im_size,
        random_crop=True,
        patches_per_pair=16,
        fov_aug=True,
        aug_seed=args.seed,
    )
    valset = FOVAugPhasePairDataset(
        im_dir=str(train_dir),
        pair_files=man["val_pairs"],
        im_size=im_size,
        random_crop=False,
        patches_per_pair=8,
        fov_aug=False,
    )
    g_dl = torch.Generator()
    g_dl.manual_seed(args.seed)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, generator=g_dl)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    generator = UNetCRBDecoder(im_size=im_size, n_phases=10).to(device)
    state = torch.load(str(args.init_ckpt), map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    generator.load_state_dict(state, strict=True)
    print(f"Loaded init {args.init_ckpt.name}", flush=True)

    mse_loss = DVFMSELoss()
    optimizer = optim.Adam(generator.parameters(), lr=args.lr)

    wdir = EXP / "weights"
    pdir = EXP / "plots"
    ldir = EXP / "logs"
    for d in (wdir, pdir, ldir):
        d.mkdir(parents=True, exist_ok=True)

    filename = "g160_dir_ft_c1c5c8_decoder"
    min_val = float("inf")
    train_losses, val_losses = [], []
    tic = time.time()
    print(
        f"[{filename}] device={device} | patients={man['train_patients']} | "
        f"train_pairs={len(man['train_pairs'])} val_pairs={len(man['val_pairs'])} | "
        f"lr={args.lr} epochs={args.epochs}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        generator.train()
        tr = 0.0
        for data in trainloader:
            fake = generator(
                data["reference_ct"].to(device),
                data["ref_phase"].to(device),
                data["target_phase"].to(device),
            )
            loss = mse_loss.loss(
                data["target_dvf"].to(device), fake, data["lung_mask"].to(device)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr += loss.item()

        generator.eval()
        va = 0.0
        with torch.no_grad():
            for data in valloader:
                fake = generator(
                    data["reference_ct"].to(device),
                    data["ref_phase"].to(device),
                    data["target_phase"].to(device),
                )
                va += mse_loss.loss(
                    data["target_dvf"].to(device), fake, data["lung_mask"].to(device)
                ).item()

        ntr = max(len(trainset), 1)
        nva = max(len(valset), 1)
        train_losses.append(tr / ntr)
        val_losses.append(va / nva)
        elapsed = time.time() - tic
        print(
            f"Epoch: {epoch} | train MSE: {train_losses[-1]:.6f} | val MSE: {val_losses[-1]:.6f} | "
            f"{elapsed/60:.1f} min",
            flush=True,
        )
        if val_losses[-1] < min_val:
            min_val = val_losses[-1]
            torch.save(generator.state_dict(), str(wdir / f"{filename}_generator.pth"))
            torch.save(generator.state_dict(), str(wdir / "best.pt"))
        # also save last
        torch.save(generator.state_dict(), str(wdir / f"{filename}_last.pth"))
        plt.figure()
        xs = np.arange(1, epoch + 1)
        plt.plot(xs, train_losses, "b-o", markersize=3, label="Train")
        plt.plot(xs, val_losses, "r-o", markersize=3, label="Val")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Lung-masked MSE")
        plt.title(filename)
        plt.savefig(str(pdir / f"{filename}.png"))
        plt.close()
        (pdir / "loss_history.json").write_text(
            json.dumps(
                {
                    "train_loss": train_losses,
                    "val_loss": val_losses,
                    "best_val": min_val,
                    "init_ckpt": str(args.init_ckpt),
                    "lr": args.lr,
                    "patients": man["train_patients"],
                },
                indent=2,
            )
            + "\n"
        )

    print(f"finished best_val_mse={min_val:.6f} → {wdir / 'best.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
