#!/usr/bin/env python3
"""Fine-tune TCIA3 ep100 with MAE + direct |û|↔|El| mag-match (copy, not overwrite).

Uses TCIA3.1 high-motion oversampled train list. Gate: DIR TRE75 at ep5 & ep10;
kill if ep10 gain vs TCIA3 baseline < 0.3 mm.

  cd PopulationStudy/ClinicalExperiments/Grid160/TCIA3_magMatch
  LEARN-GUI/.venv/bin/python scripts/train_finetune_magmatch.py --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
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
DIR_EXP = T.parent.parent.parent.parent / "DIR EXPERIMENTS"
LEARN_PY = Path("/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python")

import importlib.util


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_loss_mod = _load("magmatch_losses", T / "losses" / "losses.py")
DVFMAEMagMatchLoss = _loss_mod.DVFMAEMagMatchLoss

if str(E1) not in sys.path:
    sys.path.insert(0, str(E1))
from networks.generator_crb_dec import UNetCRBDecoder  # noqa: E402
from utilities.seed_utils import set_seeds  # noqa: E402

_fd = _load("magmatch_fast_dataset", T / "scripts" / "fast_dataset.py")
FastMmapPhasePairDataset = _fd.FastMmapPhasePairDataset

warnings.filterwarnings("ignore")

INIT_CKPT = T3 / "DecoderCRB/checkpoints/epoch_100.pt"
CKPT_DIR = T / "DecoderCRB/checkpoints"
WEIGHTS_DIR = T / "DecoderCRB/weights"
PLOTS_DIR = T / "DecoderCRB/plots"
FILENAME = "crb_dec_mae_magmatch_ft_tcia3ep100"
TCIA3_TRE75_BASELINE = 4.938  # known synth-oracle cohort


def load_manifest() -> dict:
    p = T / "data" / "manifest.json"
    if p.is_file() and p.stat().st_size > 0:
        return json.loads(p.read_text())
    raise FileNotFoundError(f"missing {p}")


def dir_tre75(ckpt: Path, gpu: int) -> float:
    """Quick DIR TRE75 cohort mean via amp_oracle a=1."""
    out = T / "logs" / f"gate_tre_{ckpt.stem}.json"
    cmd = [
        str(LEARN_PY),
        "-u",
        str(DIR_EXP / "scripts" / "amp_oracle_a_sweep.py"),
        "--gpu",
        "0",
        "--ckpt",
        str(ckpt),
        "--a-min",
        "1.0",
        "--a-max",
        "1.0",
        "--a-step",
        "1.0",
        "--out",
        str(out),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    subprocess.run(cmd, cwd=str(DIR_EXP), env=env, check=True)
    d = json.loads(out.read_text())
    return float(d["cohort"]["a1"]["mean_tre75_mm"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init-ckpt", type=Path, default=INIT_CKPT)
    ap.add_argument("--lambda-mag", type=float, default=1.0)
    ap.add_argument("--lambda-si", type=float, default=0.5)
    ap.add_argument("--patches-train", type=int, default=4)
    ap.add_argument("--patches-val", type=int, default=2)
    ap.add_argument("--gate-epochs", type=int, nargs="*", default=[5, 10])
    ap.add_argument("--gate-min-gain", type=float, default=0.3)
    ap.add_argument("--baseline-tre75", type=float, default=TCIA3_TRE75_BASELINE)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    set_seeds(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    init_ckpt = args.init_ckpt.resolve()
    if not init_ckpt.is_file():
        raise SystemExit(f"missing init ckpt: {init_ckpt}")
    print(f"[magMatch] READ-ONLY init ← {init_ckpt}", flush=True)

    man = load_manifest()
    im_dir = man["pooled_train_dir"]
    print(
        f"[magMatch] oversample train_pairs={len(man['train_pairs'])} "
        f"val={len(man['val_pairs'])} | criterion={man.get('oversample', {}).get('criterion')}",
        flush=True,
    )

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
    g.load_state_dict(state, strict=True)
    print("[magMatch] loaded init OK", flush=True)

    crit = DVFMAEMagMatchLoss(lambda_mag=args.lambda_mag, lambda_si=args.lambda_si)
    opt = optim.Adam(g.parameters(), lr=args.lr)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    (T / "logs").mkdir(exist_ok=True)

    t3_dec = (T3 / "DecoderCRB").resolve()
    if CKPT_DIR.resolve() == t3_dec or t3_dec in CKPT_DIR.resolve().parents:
        raise SystemExit("refuse writing into TCIA3")

    print(
        f"[magMatch] device={device} | MAE+mag+SI λ_mag={args.lambda_mag} λ_si={args.lambda_si} | "
        f"ep={args.epochs} lr={args.lr} | train={len(trainset)} val={len(valset)}",
        flush=True,
    )
    print(
        f"[magMatch] gate @ {args.gate_epochs}: kill if TRE75 gain < {args.gate_min_gain} mm "
        f"vs baseline {args.baseline_tre75:.3f}",
        flush=True,
    )

    train_hist, val_hist = [], []
    mae_h, mag_h, si_h = [], [], []
    best_val = float("inf")
    best_path = WEIGHTS_DIR / f"{FILENAME}_generator.pth"
    gate_log = []
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        g.train()
        tr = tr_mae = tr_mag = tr_si = 0.0
        for data in trainloader:
            ref = data["reference_ct"].to(device)
            mask = data["lung_mask"].to(device)
            rp = data["ref_phase"].to(device)
            tp = data["target_phase"].to(device)
            gt = data["target_dvf"].to(device)
            pred = g(ref, rp, tp)
            opt.zero_grad(set_to_none=True)
            total, l_mae, l_mag, l_si = crit.loss_parts(gt, pred, mask)
            total.backward()
            opt.step()
            tr += float(total.item())
            tr_mae += float(l_mae.item())
            tr_mag += float(l_mag.item())
            tr_si += float(l_si.item())

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
        mae_h.append(tr_mae / n_tr)
        mag_h.append(tr_mag / n_tr)
        si_h.append(tr_si / n_tr)

        mark = ""
        if va_m < best_val:
            best_val = va_m
            mark = " *best*"
            torch.save(g.state_dict(), str(best_path))

        ep_path = CKPT_DIR / f"epoch_{ep:03d}.pt"
        torch.save(
            {
                "generator": g.state_dict(),
                "epoch": ep,
                "val_loss": va_m,
                "init_ckpt": str(init_ckpt),
                "lambda_mag": args.lambda_mag,
                "lambda_si": args.lambda_si,
                "lr": args.lr,
            },
            str(ep_path),
        )
        torch.save(
            {"generator": g.state_dict(), "epoch": ep, "val_loss": va_m},
            str(CKPT_DIR / "latest.pt"),
        )

        print(
            f"Epoch {ep}/{args.epochs} | train {tr_m:.4f} "
            f"(mae {mae_h[-1]:.4f} mag {mag_h[-1]:.4f} si {si_h[-1]:.4f}) | "
            f"val {va_m:.4f} | {(time.time()-t0)/60:.1f} min{mark}",
            flush=True,
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        xs = np.arange(1, ep + 1)
        ax.plot(xs, train_hist, label="train")
        ax.plot(xs, val_hist, label="val")
        ax.plot(xs, mae_h, "--", label="mae")
        ax.plot(xs, mag_h, "--", label="mag")
        ax.plot(xs, si_h, "--", label="si")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title(FILENAME)
        ax.legend()
        fig.tight_layout()
        fig.savefig(str(PLOTS_DIR / f"{FILENAME}.png"), dpi=120)
        plt.close(fig)

        if ep in args.gate_epochs:
            print(f"[magMatch] GATE ep{ep}: running DIR TRE75…", flush=True)
            # free some GPU mem for TRE
            torch.cuda.empty_cache()
            tre = dir_tre75(ep_path, args.gpu)
            gain = args.baseline_tre75 - tre
            entry = {
                "epoch": ep,
                "tre75": tre,
                "baseline": args.baseline_tre75,
                "gain": gain,
                "ckpt": str(ep_path),
            }
            gate_log.append(entry)
            (T / "logs" / "gate_log.json").write_text(json.dumps(gate_log, indent=2) + "\n")
            print(
                f"[magMatch] GATE ep{ep}: TRE75={tre:.3f} baseline={args.baseline_tre75:.3f} "
                f"gain={gain:+.3f} mm",
                flush=True,
            )
            if ep == max(args.gate_epochs) and gain < args.gate_min_gain:
                print(
                    f"[magMatch] KILL: gain {gain:+.3f} < {args.gate_min_gain} mm — stopping",
                    flush=True,
                )
                break

    hist = {
        "train": train_hist,
        "val": val_hist,
        "mae": mae_h,
        "mag": mag_h,
        "si": si_h,
        "best_val": best_val,
        "gate": gate_log,
        "init_ckpt": str(init_ckpt),
        "lambda_mag": args.lambda_mag,
        "lambda_si": args.lambda_si,
        "oversample": man.get("oversample"),
    }
    (PLOTS_DIR / f"{FILENAME}_history.json").write_text(json.dumps(hist, indent=2) + "\n")
    print(f"[magMatch] done best_val={best_val:.4f} → {best_path}", flush=True)
    print(f"[magMatch] TCIA3 unchanged: {init_ckpt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
