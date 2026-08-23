"""Shared MSE train loop for Experiment 4 CRB models (pretrained anatomy).

Default / launched: Decoder-CRB (frozen encoder swap).
Encoder-CRB and Both-CRB side-branch code is here but not started.

  PYTHONPATH=. python DecoderCRB/train_crb_dec_mse.py --gpu 0
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

E4 = Path(__file__).resolve().parents[1]
if str(E4) not in sys.path:
    sys.path.insert(0, str(E4))

from losses.losses import DVFMSELoss
from networks.generator_crb import UNetCRB
from networks.generator_crb_both import UNetCRBBoth
from networks.generator_crb_dec import UNetCRBDecoder
from utilities.dataset import MultiPatientPhasePairDataset
from utilities.seed_utils import load_seed_config, set_seeds

warnings.filterwarnings("ignore")

DEFAULT_ANATOMY = E4 / "AnatomyAE" / "weights" / "anatomy_encoder.pth"

ARCH = {
    "encoder": {
        "cls": UNetCRB,
        "run_dir": E4 / "EncoderCRB",
        "filename": "crb_enc_mse_pop_e4",
        "label": "CRB=encoder + frozen anatomy side-branch",
        "side_branch": True,
    },
    "decoder": {
        "cls": UNetCRBDecoder,
        "run_dir": E4 / "DecoderCRB",
        "filename": "crb_dec_mse_pop_e4",
        "label": "CRB=decoder; frozen pretrained encoder",
        "side_branch": False,
    },
    "both": {
        "cls": UNetCRBBoth,
        "run_dir": E4 / "BothCRB",
        "filename": "crb_both_mse_pop_e4",
        "label": "CRB=encoder+decoder + frozen anatomy side-branch",
        "side_branch": True,
    },
}


def load_manifest():
    path = E4 / "data" / "pooled" / "manifest.json"
    with open(path) as f:
        return json.load(f)


def build_generator(arch: str, n_phases: int, anatomy_dim: int):
    spec = ARCH[arch]
    if spec["side_branch"]:
        return spec["cls"](im_size=64, n_phases=n_phases, anatomy_dim=anatomy_dim)
    return spec["cls"](im_size=64, n_phases=n_phases)


def train_one(
    arch: str,
    epochs: int,
    lr: float,
    gpu: int,
    seed: int | None,
    anatomy_ckpt: Path,
    anatomy_dim: int,
):
    spec = ARCH[arch]
    cfg = load_seed_config()
    if seed is None:
        seed = int(cfg["torch_seed"])
    set_seeds(seed)
    anatomy_ckpt = Path(anatomy_ckpt)
    if not anatomy_ckpt.exists():
        raise FileNotFoundError(
            f"anatomy encoder not found: {anatomy_ckpt} (run scripts/train_ae.py first)"
        )

    run_dir = spec["run_dir"]
    filename = spec["filename"]
    man = load_manifest()
    train_dir = man["pooled_train_dir"]

    im_size = 64
    n_phases = 10
    batch_size = 1
    patches_per_pair_train = 16
    patches_per_pair_val = 8
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

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
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, generator=g_dl
    )
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    generator = build_generator(arch, n_phases, anatomy_dim).to(device)
    generator.load_anatomy_encoder(str(anatomy_ckpt))
    mse_loss = DVFMSELoss()
    trainable = [p for p in generator.parameters() if p.requires_grad]
    optimizer_g = optim.Adam(trainable, lr=lr)

    os.makedirs(run_dir / "weights", exist_ok=True)
    os.makedirs(run_dir / "plots", exist_ok=True)

    n_g = sum(p.numel() for p in generator.parameters())
    n_t = sum(p.numel() for p in trainable)
    min_val_loss = float("inf")
    train_losses, val_losses = [], []
    tic = time.time()
    print(
        f"[{filename}] device={device} | {spec['cls'].__name__} "
        f"params={n_g / 1e6:.2f}M trainable={n_t / 1e6:.2f}M | "
        f"loss=lung-masked MSE | {spec['label']} | seed={seed}",
        flush=True,
    )
    print(
        f"[{filename}] anatomy={anatomy_ckpt} | "
        f"train {len(trainset)} val {len(valset)} | "
        f"train patients={man['train_patients']} hold-out={man['holdout_patients']} (QC only)",
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
        plt.ylabel("Lung-masked MSE")
        plt.title(filename)
        plt.savefig(str(run_dir / "plots" / f"{filename}.png"))
        plt.close()

    print(f"finished best_val_mse={min_val_loss:.6f}", flush=True)
    return min_val_loss


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=sorted(ARCH))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--anatomy_ckpt", type=Path, default=DEFAULT_ANATOMY)
    ap.add_argument("--anatomy_dim", type=int, default=32)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    train_one(
        args.arch,
        args.epochs,
        args.lr,
        args.gpu,
        args.seed,
        args.anatomy_ckpt,
        args.anatomy_dim,
    )


if __name__ == "__main__":
    main()
