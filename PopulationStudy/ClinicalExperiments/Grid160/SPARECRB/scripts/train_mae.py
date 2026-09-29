#!/usr/bin/env python3
"""Original UNetCRBDecoder on SPARE: full 160³, lung-masked L1.

Same net, phase code, and loss as TCIA4. Each sample is the whole 160³ volume.
P2 and P8 are held out. Writes only under SPARECRB/.

  cd …/Grid160/SPARECRB
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/train_mae.py --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

VS = Path(__file__).resolve().parents[1]
if str(VS) not in sys.path:
    sys.path.insert(0, str(VS))

from losses.losses import DVFLoss
from networks.generator_crb_dec import UNetCRBDecoder
from utilities.seed_utils import set_seeds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fast_dataset import FastMmapPhasePairDataset  # noqa: E402

warnings.filterwarnings("ignore")

FILENAME = "crb_dec_mae_full160_spare_p2p8"
LABEL = "UNetCRBDecoder · SPARE · MAE · full 160³ · linear · A1 FOV · hold out P2,P8"
RUN_DIR = VS / "DecoderCRB"
CKPT_DIR = RUN_DIR / "checkpoints"
WEIGHTS_DIR = RUN_DIR / "weights"
PLOTS_DIR = RUN_DIR / "plots"

_STOP_REQUESTED = False


def _request_stop(signum, _frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(
        f"\n[signal {signum}] stop requested — will save interrupt checkpoint "
        "after current batch / at epoch boundary",
        flush=True,
    )


def load_manifest() -> dict:
    local = VS / "data" / "manifest.json"
    if local.is_file():
        return json.loads(local.read_text())
    raise FileNotFoundError(
        "SPARECRB/data/manifest.json missing — run scripts/build_spare_manifest.py"
    )


def load_seed() -> dict:
    return json.loads((VS / "seed.json").read_text())


def resolve_resume(path: str | None) -> Path | None:
    if path is None or path.lower() in {"", "none", "false"}:
        return None
    if path.lower() == "auto":
        latest = CKPT_DIR / "latest.pt"
        interrupt = CKPT_DIR / "interrupt.pt"
        cands = [p for p in (interrupt, latest) if p.is_file()]
        if not cands:
            epochs = sorted(CKPT_DIR.glob("epoch_*.pt"))
            if not epochs:
                return None
            return epochs[-1]
        cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return cands[0]
    p = Path(path)
    if not p.is_absolute():
        p = (VS / p).resolve() if not p.exists() else p.resolve()
        if not p.exists():
            p = (CKPT_DIR / path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    return p


def save_checkpoint(
    path: Path,
    *,
    epoch_done: int,
    generator,
    optimizer,
    train_losses,
    val_losses,
    min_val_loss,
    elapsed_s,
    cfg: dict,
    extra: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch_done": int(epoch_done),
        "next_epoch": int(epoch_done) + 1,
        "generator": generator.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_losses": list(train_losses),
        "val_losses": list(val_losses),
        "min_val_loss": float(min_val_loss),
        "elapsed_s": float(elapsed_s),
        "filename": FILENAME,
        "cfg": cfg,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng": np.random.get_state(),
    }
    if extra:
        payload.update(extra)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def train(
    epochs: int,
    lr: float,
    gpu: int,
    seed: int | None,
    resume_path: Path | None,
    start_epoch: int | None,
    save_every: int,
) -> float:
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    cfg = load_seed()
    if seed is None:
        seed = int(cfg["torch_seed"])
    set_seeds(seed)

    man = load_manifest()
    train_dir = man["pooled_train_dir"]
    im_size = int(cfg.get("im_size", 160))
    n_phases = int(cfg.get("n_phases", 10))
    batch_size = int(cfg.get("batch_size", 1))
    patches_train = int(cfg.get("patches_per_pair_train", 1))
    patches_val = int(cfg.get("patches_per_pair_val", 1))
    random_crop = bool(cfg.get("random_crop", False))

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    num_workers = int(cfg.get("num_workers", 4))
    fov_aug = bool(cfg.get("fov_aug", True))
    trainset = FastMmapPhasePairDataset(
        im_dir=train_dir,
        pair_files=man["train_pairs"],
        im_size=im_size,
        random_crop=random_crop,
        patches_per_pair=patches_train,
        fov_aug=fov_aug,
        aug_seed=seed,
    )
    valset = FastMmapPhasePairDataset(
        im_dir=train_dir,
        pair_files=man["val_pairs"],
        im_size=im_size,
        random_crop=False,
        patches_per_pair=patches_val,
        fov_aug=False,
    )
    g_dl = torch.Generator()
    g_dl.manual_seed(seed)
    dl_common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    if num_workers == 0:
        dl_common.pop("prefetch_factor", None)
        dl_common.pop("persistent_workers", None)
    trainloader = DataLoader(
        trainset, shuffle=True, generator=g_dl, **dl_common
    )
    valloader = DataLoader(valset, shuffle=False, **dl_common)
    print(
        f"[{FILENAME}] full {im_size}³ mmap | num_workers={num_workers} | "
        f"patches_train={patches_train} | random_crop={random_crop}",
        flush=True,
    )

    generator = UNetCRBDecoder(im_size=im_size, n_phases=n_phases).to(device)
    mae_loss = DVFLoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    min_val_loss = float("inf")
    train_losses: list[float] = []
    val_losses: list[float] = []
    start_ep = 1
    elapsed_offset = 0.0

    if resume_path is not None:
        print(f"[resume] loading {resume_path}", flush=True)
        ckpt = torch.load(str(resume_path), map_location=device, weights_only=False)
        generator.load_state_dict(ckpt["generator"])
        if "optimizer" in ckpt:
            optimizer_g.load_state_dict(ckpt["optimizer"])
        train_losses = list(ckpt.get("train_losses", []))
        val_losses = list(ckpt.get("val_losses", []))
        min_val_loss = float(ckpt.get("min_val_loss", float("inf")))
        elapsed_offset = float(ckpt.get("elapsed_s", 0.0))
        start_ep = int(ckpt.get("next_epoch", int(ckpt.get("epoch_done", 0)) + 1))
        if "torch_rng" in ckpt and ckpt["torch_rng"] is not None:
            try:
                st = ckpt["torch_rng"]
                if not torch.is_tensor(st):
                    st = torch.tensor(st, dtype=torch.uint8)
                elif st.dtype != torch.uint8:
                    st = st.to(dtype=torch.uint8)
                torch.set_rng_state(st.cpu())
            except Exception as e:
                print(f"[resume] warn: torch rng restore failed: {e}", flush=True)
        if ckpt.get("cuda_rng") is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
            except Exception as e:
                print(f"[resume] warn: cuda rng restore failed: {e}", flush=True)
        if ckpt.get("numpy_rng") is not None:
            try:
                np.random.set_state(ckpt["numpy_rng"])
            except Exception as e:
                print(f"[resume] warn: numpy rng restore failed: {e}", flush=True)
        print(
            f"[resume] epoch_done={ckpt.get('epoch_done')} → start_ep={start_ep} "
            f"| best_val={min_val_loss:.6f} | elapsed_so_far={elapsed_offset/3600:.2f}h",
            flush=True,
        )

    if start_epoch is not None:
        start_ep = int(start_epoch)
        print(f"[override] --start-epoch → {start_ep}", flush=True)

    n_g = sum(p.numel() for p in generator.parameters())
    print(
        f"[{FILENAME}] device={device} | UNetCRBDecoder params={n_g / 1e6:.2f}M | "
        f"{'FOV aug ¼×4' if fov_aug else 'no FOV'} | {LABEL} | seed={seed}",
        flush=True,
    )
    print(
        f"[{FILENAME}] train {len(trainset)} | val {len(valset)} | "
        f"train_scans={len(man.get('train_scans', []))} | "
        f"holdout={man.get('holdout_scans', [])} | "
        f"pairs train/val={man['n_train_pairs']}/{man['n_val_pairs']} | "
        f"epochs {start_ep}..{epochs}",
        flush=True,
    )

    tic = time.time()
    pid_file = VS / "logs" / "train.pid"
    pid_file.write_text(str(os.getpid()) + "\n")

    interrupted = False
    for epoch in range(start_ep, epochs + 1):
        if _STOP_REQUESTED:
            interrupted = True
            break

        generator.train()
        train_loss = 0.0
        n_batches = 0
        for data in trainloader:
            if _STOP_REQUESTED:
                interrupted = True
                break
            reference_ct = data["reference_ct"].to(device)
            lung_mask = data["lung_mask"].to(device)
            ref_phase = data["ref_phase"].to(device)
            target_phase = data["target_phase"].to(device)
            target_dvf = data["target_dvf"].to(device)
            fake_dvf = generator(reference_ct, ref_phase, target_phase)
            optimizer_g.zero_grad()
            loss = mae_loss.loss(target_dvf, fake_dvf, lung_mask)
            loss.backward()
            optimizer_g.step()
            train_loss += loss.item()
            n_batches += 1

        if interrupted:
            elapsed_s = elapsed_offset + (time.time() - tic)
            save_checkpoint(
                CKPT_DIR / "interrupt.pt",
                epoch_done=epoch - 1,
                generator=generator,
                optimizer=optimizer_g,
                train_losses=train_losses,
                val_losses=val_losses,
                min_val_loss=min_val_loss,
                elapsed_s=elapsed_s,
                cfg=cfg,
                extra={"interrupted_during_epoch": epoch, "partial_train_batches": n_batches},
            )
            print(
                f"[stop] interrupt.pt saved (was in epoch {epoch}, "
                f"{n_batches} train batches). Resume with --resume auto",
                flush=True,
            )
            break

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
                val_loss += mae_loss.loss(target_dvf, fake_dvf, lung_mask).item()

        elapsed_s = elapsed_offset + (time.time() - tic)
        hours = int(np.floor(elapsed_s / 3600.0))
        minutes = int((elapsed_s / 3600.0 - hours) * 60)
        n_train = max(len(trainset), 1)
        n_val = max(len(valset), 1)
        train_losses.append(train_loss / n_train)
        val_losses.append(val_loss / n_val)
        print(
            "Epoch: %d | train MAE: %.6f | val MAE: %.6f | total time: %d hours %d minutes"
            % (epoch, train_losses[-1], val_losses[-1], hours, minutes),
            flush=True,
        )

        if val_losses[-1] < min_val_loss:
            torch.save(
                generator.state_dict(),
                str(WEIGHTS_DIR / f"{FILENAME}_generator.pth"),
            )
            min_val_loss = val_losses[-1]

        plt.figure()
        xs = np.arange(1, len(train_losses) + 1)
        plt.plot(xs, train_losses, "b-o", markersize=3, label="Train MAE")
        plt.plot(xs, val_losses, "r-o", markersize=3, label="Val MAE")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Lung-masked MAE (iso-vox)")
        plt.title(FILENAME)
        plt.savefig(str(PLOTS_DIR / f"{FILENAME}.png"))
        plt.close()

        if epoch % max(1, save_every) == 0 or epoch == epochs:
            ep_path = CKPT_DIR / f"epoch_{epoch:03d}.pt"
            save_checkpoint(
                ep_path,
                epoch_done=epoch,
                generator=generator,
                optimizer=optimizer_g,
                train_losses=train_losses,
                val_losses=val_losses,
                min_val_loss=min_val_loss,
                elapsed_s=elapsed_s,
                cfg=cfg,
            )
            save_checkpoint(
                CKPT_DIR / "latest.pt",
                epoch_done=epoch,
                generator=generator,
                optimizer=optimizer_g,
                train_losses=train_losses,
                val_losses=val_losses,
                min_val_loss=min_val_loss,
                elapsed_s=elapsed_s,
                cfg=cfg,
            )
            interrupt = CKPT_DIR / "interrupt.pt"
            if interrupt.is_file():
                interrupt.unlink()

        if _STOP_REQUESTED:
            interrupted = True
            print(f"[stop] finished epoch {epoch}; latest.pt is resume-ready", flush=True)
            break

    if pid_file.is_file():
        try:
            pid_file.unlink()
        except OSError:
            pass

    status = "interrupted" if interrupted else "finished"
    print(f"{status} best_val_mae={min_val_loss:.6f}", flush=True)
    return min_val_loss


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path, or 'auto' (latest/interrupt), or omit to start fresh",
    )
    ap.add_argument(
        "--start-epoch",
        type=int,
        default=None,
        help="Override next epoch after loading resume (1-based)",
    )
    ap.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Write epoch_XXX.pt every N epochs (default 1)",
    )
    args = ap.parse_args()
    resume = resolve_resume(args.resume)
    train(
        args.epochs,
        args.lr,
        args.gpu,
        args.seed,
        resume,
        args.start_epoch,
        args.save_every,
    )


if __name__ == "__main__":
    main()
