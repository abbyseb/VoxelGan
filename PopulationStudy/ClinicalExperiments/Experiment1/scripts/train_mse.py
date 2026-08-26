"""Train CRB models for ClinicalExperiments Experiment 1 (full SPARE train).

Reuses InitialExperiments/Experiment1 network/loss/dataset modules to keep
architecture and loss behavior identical to E1/E2.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

E1 = Path(__file__).resolve().parents[1]
IE1 = E1.parents[1] / "InitialExperiments" / "Experiment1"
if str(IE1) not in sys.path:
    sys.path.insert(0, str(IE1))

from losses.losses import DVFMSELoss  # noqa: E402
from networks.generator_crb import UNetCRB  # noqa: E402
from networks.generator_crb_both import UNetCRBBoth  # noqa: E402
from networks.generator_crb_dec import UNetCRBDecoder  # noqa: E402
from utilities.dataset import MultiPatientPhasePairDataset  # noqa: E402
from utilities.seed_utils import set_seeds  # noqa: E402

ARCH = {
    "encoder": {
        "cls": UNetCRB,
        "run_dir": E1 / "EncoderCRB",
        "filename": "crb_enc_mse_full_spare",
        "label": "CRB=encoder-only",
    },
    "decoder": {
        "cls": UNetCRBDecoder,
        "run_dir": E1 / "DecoderCRB",
        "filename": "crb_dec_mse_full_spare",
        "label": "CRB=decoder-only",
    },
    "both": {
        "cls": UNetCRBBoth,
        "run_dir": E1 / "BothCRB",
        "filename": "crb_both_mse_full_spare",
        "label": "CRB=encoder+decoder",
    },
}


def load_manifest():
    path = E1 / "data" / "pooled" / "manifest.json"
    return json.loads(path.read_text())


def load_seed():
    return json.loads((E1 / "seed.json").read_text())


def train_one(arch: str, epochs: int, lr: float, gpu: int, seed: int | None):
    spec = ARCH[arch]
    cfg = load_seed()
    if seed is None:
        seed = int(cfg["torch_seed"])
    set_seeds(seed)
    man = load_manifest()
    train_dir = man["pooled_train_dir"]

    im_size = 64
    n_phases = 10
    patches_per_pair_train = 16
    patches_per_pair_val = 8
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    trainset = MultiPatientPhasePairDataset(
        im_dir=train_dir,
        pair_files=man["train_pairs"],
        im_size=im_size,
        random_crop=True,
        patches_per_pair=patches_per_pair_train,
    )
    valset = MultiPatientPhasePairDataset(
        im_dir=train_dir,
        pair_files=man["val_pairs"],
        im_size=im_size,
        random_crop=False,
        patches_per_pair=patches_per_pair_val,
    )
    g_dl = torch.Generator()
    g_dl.manual_seed(seed)
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, generator=g_dl)
    valloader = DataLoader(valset, batch_size=1, shuffle=False)

    generator = spec["cls"](im_size=im_size, n_phases=n_phases).to(device)
    mse_loss = DVFMSELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr)

    run_dir = spec["run_dir"]
    os.makedirs(run_dir / "weights", exist_ok=True)
    os.makedirs(run_dir / "plots", exist_ok=True)

    min_val_loss = float("inf")
    train_losses, val_losses = [], []
    tic = time.time()
    n_g = sum(p.numel() for p in generator.parameters())
    print(
        f'[{spec["filename"]}] device={device} | {spec["cls"].__name__} params={n_g / 1e6:.2f}M | '
        f'loss=lung-masked MSE | no D | {spec["label"]} | seed={seed}',
        flush=True,
    )
    print(
        f'[{spec["filename"]}] train {len(trainset)} ({len(man["train_pairs"])} pairs × {patches_per_pair_train}) | '
        f'val {len(valset)} ({len(man["val_pairs"])} pairs × {patches_per_pair_val}) | '
        f'train patients={man["train_patients"]}',
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

        elapsed_h = (time.time() - tic) / 3600.0
        hours = int(np.floor(elapsed_h))
        minutes = int((elapsed_h - hours) * 60)
        train_losses.append(train_loss / max(len(trainset), 1))
        val_losses.append(val_loss / max(len(valset), 1))
        print(
            "Epoch: %d | train MSE: %.6f | val MSE: %.6f | total time: %d hours %d minutes"
            % (epoch, train_losses[-1], val_losses[-1], hours, minutes),
            flush=True,
        )
        if val_losses[-1] < min_val_loss:
            torch.save(
                generator.state_dict(),
                str(run_dir / "weights" / f'{spec["filename"]}_generator.pth'),
            )
            min_val_loss = val_losses[-1]
        plt.figure()
        xs = np.arange(1, epoch + 1)
        plt.plot(xs, train_losses, "b-o", markersize=3, label="Train MSE")
        plt.plot(xs, val_losses, "r-o", markersize=3, label="Val MSE")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Lung-masked MSE")
        plt.title(spec["filename"])
        plt.savefig(str(run_dir / "plots" / f'{spec["filename"]}.png'))
        plt.close()

    print(f"finished best_val_mse={min_val_loss:.6f}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=sorted(ARCH))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)
    train_one(args.arch, args.epochs, args.lr, args.gpu, args.seed)


if __name__ == "__main__":
    main()
