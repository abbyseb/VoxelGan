"""Train G160-A1: linear CRB + FOV/CBCT aug on data_iso (2 mm 160³).

  cd PopulationStudy/ClinicalExperiments/Grid160/Experiment1
  PYTHONPATH=. python scripts/train_mse.py --arch both --full --gpu 0
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

E1 = Path(__file__).resolve().parents[1]
if str(E1) not in sys.path:
    sys.path.insert(0, str(E1))

from losses.losses import DVFMSELoss
from networks.generator_crb import UNetCRB
from networks.generator_crb_both import UNetCRBBoth
from networks.generator_crb_dec import UNetCRBDecoder
from utilities.dataset import FOVAugPhasePairDataset
from utilities.seed_utils import set_seeds

warnings.filterwarnings("ignore")

ARCH_HOLDOUT = {
    "encoder": {
        "cls": UNetCRB,
        "run_dir": E1 / "EncoderCRB",
        "filename": "crb_enc_mse_iso_g160_fov",
        "label": "G160-A1 encoder · iso 2mm · linear · FOV aug · E1 split",
    },
    "decoder": {
        "cls": UNetCRBDecoder,
        "run_dir": E1 / "DecoderCRB",
        "filename": "crb_dec_mse_iso_g160_fov",
        "label": "G160-A1 decoder · iso 2mm · linear · FOV aug · E1 split",
    },
    "both": {
        "cls": UNetCRBBoth,
        "run_dir": E1 / "BothCRB",
        "filename": "crb_both_mse_iso_g160_fov",
        "label": "G160-A1 both · iso 2mm · linear · FOV aug · E1 split",
    },
}

ARCH_FULL = {
    k: {
        **v,
        "filename": v["filename"] + "_full",
        "label": v["label"].replace("E1 split", "full P1–P9"),
    }
    for k, v in ARCH_HOLDOUT.items()
}


def load_manifest(full: bool = False):
    sub = "pooled_full" if full else "pooled"
    path = E1 / "data" / sub / "manifest.json"
    with open(path) as f:
        return json.load(f)


def load_seed_config(full: bool = False):
    path = E1 / ("seed_full.json" if full else "seed.json")
    with open(path) as f:
        return json.load(f)


def train_one(arch: str, epochs: int, lr: float, gpu: int, seed: int | None, full: bool = False):
    arch_map = ARCH_FULL if full else ARCH_HOLDOUT
    spec = arch_map[arch]
    cfg = load_seed_config(full=full)
    if seed is None:
        seed = int(cfg["torch_seed"])
    set_seeds(seed)

    run_dir = spec["run_dir"]
    filename = spec["filename"]
    man = load_manifest(full=full)
    train_dir = man["pooled_train_dir"]

    im_size = 64
    n_phases = 10
    batch_size = 1
    patches_per_pair_train = 16
    patches_per_pair_val = 8
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    trainset = FOVAugPhasePairDataset(
        im_dir=train_dir,
        pair_files=man["train_pairs"],
        im_size=im_size,
        random_crop=True,
        patches_per_pair=patches_per_pair_train,
        fov_aug=True,
        aug_seed=seed,
    )
    valset = FOVAugPhasePairDataset(
        im_dir=train_dir,
        pair_files=man["val_pairs"],
        im_size=im_size,
        random_crop=False,
        patches_per_pair=patches_per_pair_val,
        fov_aug=False,
    )
    g_dl = torch.Generator()
    g_dl.manual_seed(seed)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, generator=g_dl)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    generator = spec["cls"](im_size=im_size, n_phases=n_phases).to(device)
    mse_loss = DVFMSELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr)

    os.makedirs(run_dir / "weights", exist_ok=True)
    os.makedirs(run_dir / "plots", exist_ok=True)

    min_val_loss = float("inf")
    train_losses, val_losses = [], []
    tic = time.time()
    n_g = sum(p.numel() for p in generator.parameters())
    print(
        f"[{filename}] device={device} | {spec['cls'].__name__} params={n_g / 1e6:.2f}M | "
        f"FOV aug ¼×4 train | val=normal | {spec['label']} | seed={seed}",
        flush=True,
    )
    print(
        f"[{filename}] train {len(trainset)} | val {len(valset)} | "
        f"patients={man['train_patients']} hold-out={man['holdout_patients']}",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        generator.train()
        train_loss = 0.0
        for data in trainloader:
            reference_ct = data["reference_ct"].to(device)
            lung_mask = data["lung_mask"].to(device)
            ref_phase = data["ref_phase"].to(device)
            target_phase = data["target_phase"].to(device)
            target_dvf = data["target_dvf"].to(device)
            fake_dvf = generator(reference_ct, ref_phase, target_phase)
            optimizer_g.zero_grad()
            loss = mse_loss.loss(target_dvf, fake_dvf, lung_mask)
            loss.backward()
            optimizer_g.step()
            train_loss += loss.item()

        generator.eval()
        val_loss = 0.0
        with torch.no_grad():
            for valdata in valloader:
                reference_ct = valdata["reference_ct"].to(device)
                lung_mask = valdata["lung_mask"].to(device)
                ref_phase = valdata["ref_phase"].to(device)
                target_phase = valdata["target_phase"].to(device)
                target_dvf = valdata["target_dvf"].to(device)
                fake_dvf = generator(reference_ct, ref_phase, target_phase)
                val_loss += mse_loss.loss(target_dvf, fake_dvf, lung_mask).item()

        toc = time.time()
        elapsed_h = (toc - tic) / 3600.0
        hours = int(np.floor(elapsed_h))
        minutes = int((elapsed_h - hours) * 60)
        n_train = max(len(trainset), 1)
        n_val = max(len(valset), 1)
        train_losses.append(train_loss / n_train)
        val_losses.append(val_loss / n_val)
        print(
            "Epoch: %d | train MSE: %.6f | val MSE: %.6f | total time: %d hours %d minutes"
            % (epoch, train_losses[-1], val_losses[-1], hours, minutes),
            flush=True,
        )
        if val_losses[-1] < min_val_loss:
            torch.save(
                generator.state_dict(),
                str(run_dir / "weights" / f"{filename}_generator.pth"),
            )
            min_val_loss = val_losses[-1]
        plt.figure()
        xs = np.arange(1, epoch + 1)
        plt.plot(xs, train_losses, "b-o", markersize=3, label="Train MSE")
        plt.plot(xs, val_losses, "r-o", markersize=3, label="Val MSE")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Lung-masked MSE (iso-vox²)")
        plt.title(filename)
        plt.savefig(str(run_dir / "plots" / f"{filename}.png"))
        plt.close()

    print(f"finished best_val_mse={min_val_loss:.6f}", flush=True)
    return min_val_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=sorted(ARCH_HOLDOUT))
    ap.add_argument("--full", action="store_true", help="train on all P1–P9 (G160-A1 default)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    train_one(args.arch, args.epochs, args.lr, args.gpu, args.seed, full=args.full)


if __name__ == "__main__":
    main()
