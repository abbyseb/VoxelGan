#!/usr/bin/env python3
"""Oracle check: same scale head, trained on DIR Elastix.

This uses DIR-Lab Elastix as the target. It is a diagnostic, not an A3 score.
The frozen decoder is still TCIA3 epoch_100. The head only sees CT and |u|.

DIR stores DVF_sub with fixed = phase 06. The decoder was trained on the
opposite registration (fixed = target phase). A one-case cosine probe picks
the sign (+field or −field) that matches the decoder, then that sign is saved.

Hold-out patients are C08 and C10. Train is the other eight.

  cd PopulationStudy/ClinicalExperiments/Grid160/TCIA3_scaleMap
  CUDA_VISIBLE_DEVICES=0 python scripts/train_scale_map_dir.py --gpu 0
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
import SimpleITK as sitk
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from train_scale_map import (  # noqa: E402
    S_HI,
    S_LO,
    ScaleMapHead,
    load_decoder,
    lung_l1,
    lung_mean_s,
    make_loader,
    run_epoch,
)

DIR_EXP = HERE.parents[3] / "DIR EXPERIMENTS"
A1 = DIR_EXP / "arms" / "A1_oracle_dirlab" / "runs"
DATA = HERE / "data" / "dir"
OUT = HERE / "checkpoints" / "dir_oracle"
PLOTS = HERE / "plots"
INFER = 160
HOLD_CASES = {8, 10}


def hu_to_mu(hu: np.ndarray) -> np.ndarray:
    return (hu.astype(np.float32) + 1000.0) * (0.02 / 1000.0)


def resize_zyx(vol: np.ndarray, size: int, mode: str) -> np.ndarray:
    t = torch.from_numpy(vol[None, None].astype(np.float32))
    if mode == "nearest":
        t = F.interpolate(t, size=(size, size, size), mode="nearest")
    else:
        t = F.interpolate(t, size=(size, size, size), mode="trilinear", align_corners=True)
    return t[0, 0].numpy()


def upsample_dvf(dvf_zyx3: np.ndarray, size: int) -> np.ndarray:
    d0, h0, w0, _ = dvf_zyx3.shape
    t = torch.from_numpy(np.moveaxis(dvf_zyx3, -1, 0)[None].astype(np.float32))
    out = F.interpolate(t, size=(size, size, size), mode="trilinear", align_corners=True)
    scales = torch.tensor([size / d0, size / h0, size / w0], dtype=out.dtype).view(1, 3, 1, 1, 1)
    out = out * scales
    return np.moveaxis(out[0].numpy(), 0, -1).astype(np.float32)


def mask_to_128(path: Path) -> np.ndarray:
    img = sitk.ReadImage(str(path))
    in_size = img.GetSize()
    in_spacing = img.GetSpacing()
    target = (128, 128, 128)
    out_spacing = [(in_size[i] * in_spacing[i]) / target[i] for i in range(3)]
    rs = sitk.ResampleImageFilter()
    rs.SetInterpolator(sitk.sitkNearestNeighbor)
    rs.SetSize(list(target))
    rs.SetOutputOrigin(img.GetOrigin())
    rs.SetOutputSpacing(out_spacing)
    rs.SetOutputDirection(img.GetDirection())
    rs.SetDefaultPixelValue(0)
    arr = sitk.GetArrayFromImage(rs.Execute(img))
    return (arr > 0).astype(np.float32)


def case_train(case: int) -> Path:
    scan = f"DIR_C{case:02d}"
    return A1 / scan / scan / "train"


def load_case_arrays(case: int) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray]:
    train = case_train(case)
    cts = {}
    for ph in range(1, 11):
        hu = sitk.GetArrayFromImage(sitk.ReadImage(str(train / f"sub_CT_{ph:02d}.mha"))).astype(
            np.float32
        )
        cts[ph] = resize_zyx(hu_to_mu(hu), INFER, "trilinear")
    dvfs = {6: np.zeros((INFER, INFER, INFER, 3), dtype=np.float32)}
    for ph in range(1, 11):
        if ph == 6:
            continue
        raw = sitk.GetArrayFromImage(sitk.ReadImage(str(train / f"DVF_sub_{ph:02d}.mha"))).astype(
            np.float32
        )
        dvfs[ph] = upsample_dvf(raw, INFER)
    mask = resize_zyx(mask_to_128(train / "Mask_Lung.mha"), INFER, "nearest")
    mask = (mask > 0.5).astype(np.float32)
    return cts, dvfs, mask


def _crop_box(mask: np.ndarray, s: int = 64) -> tuple[int, int, int]:
    coords = np.argwhere(mask > 0.5)
    cz, cy, cx = coords.mean(axis=0)
    shape = mask.shape
    return tuple(int(np.clip(c - s // 2, 0, shape[i] - s)) for i, c in enumerate((cz, cy, cx)))


def cosine_u(decoder, device, ct160: np.ndarray, dvf160: np.ndarray, mask160: np.ndarray) -> float:
    z0, y0, x0 = _crop_box(mask160)
    s = 64
    ct = ct160[z0 : z0 + s, y0 : y0 + s, x0 : x0 + s]
    ct = (ct - ct160.min()) / (ct160.max() - ct160.min() + 1e-8)
    m = mask160[z0 : z0 + s, y0 : y0 + s, x0 : x0 + s] > 0.5
    dvf = dvf160[z0 : z0 + s, y0 : y0 + s, x0 : x0 + s]
    gt = torch.from_numpy(np.moveaxis(dvf, -1, 0)).to(device)[None]
    ct_t = torch.from_numpy(ct.astype(np.float32)).to(device)[None, None]
    rp = torch.tensor([5], device=device)
    tp = torch.tensor([0], device=device)
    with torch.no_grad():
        u = decoder(ct_t, rp, tp)
    m_t = torch.from_numpy(m).to(device)
    a = u[0, :, m_t].reshape(-1)
    b = gt[0, :, m_t].reshape(-1)
    return float(F.cosine_similarity(a, b, dim=0).item())


def prepare(decoder, device) -> tuple[list[str], list[str], float]:
    print("[dir-oracle] loading C01 and C08 to pick DVF sign", flush=True)
    c01_ct, c01_dvf, c01_m = load_case_arrays(1)
    c08_ct, c08_dvf, c08_m = load_case_arrays(8)
    cos_pos = 0.5 * (
        cosine_u(decoder, device, c01_ct[6], c01_dvf[1], c01_m)
        + cosine_u(decoder, device, c08_ct[6], c08_dvf[1], c08_m)
    )
    cos_neg = 0.5 * (
        cosine_u(decoder, device, c01_ct[6], -c01_dvf[1], c01_m)
        + cosine_u(decoder, device, c08_ct[6], -c08_dvf[1], c08_m)
    )
    sign = 1.0 if cos_pos >= cos_neg else -1.0
    print(
        f"[dir-oracle] cosine vs DVF_sub {cos_pos:+.3f} | vs −DVF_sub {cos_neg:+.3f} | using sign {sign:+.0f}",
        flush=True,
    )

    DATA.mkdir(parents=True, exist_ok=True)
    train_pairs, val_pairs = [], []
    cache = {1: (c01_ct, c01_dvf, c01_m), 8: (c08_ct, c08_dvf, c08_m)}
    for case in range(1, 11):
        pid = f"D{case:02d}"
        if case in cache:
            cts, dvfs, mask = cache[case]
        else:
            cts, dvfs, mask = load_case_arrays(case)
        np.save(DATA / f"{pid}_Mask_Lung.npy", mask)
        for ph, vol in cts.items():
            np.save(DATA / f"{pid}_CT_{ph:02d}.npy", vol.astype(np.float32))
        for tgt, field in dvfs.items():
            if tgt == 6:
                continue
            name = f"{pid}_06_to_{tgt:02d}_pair.npy"
            np.save(DATA / name, (sign * field).astype(np.float32))
            (val_pairs if case in HOLD_CASES else train_pairs).append(name)
        print(f"[dir-oracle] wrote {pid}", flush=True)
    manifest = {
        "kind": "dir_oracle_diagnostic",
        "not_an_a3_score": True,
        "uses_dir_elastix": True,
        "dvf_sign": sign,
        "cosine_plus": cos_pos,
        "cosine_minus": cos_neg,
        "holdout_cases": sorted(HOLD_CASES),
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "pooled_train_dir": str(DATA),
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return train_pairs, val_pairs, sign


def per_case(head, decoder, pairs, device, batch) -> list[dict]:
    loader = make_loader(
        str(DATA), pairs, patches=2, random_crop=False, batch=batch, workers=0, shuffle=False
    )
    head.eval()
    acc: dict[str, dict[str, float]] = {}
    for batch_i in loader:
        ct = batch_i["reference_ct"].to(device)
        mask = batch_i["lung_mask"].to(device)
        gt = batch_i["target_dvf"].to(device)
        rp = batch_i["ref_phase"].to(device)
        tp = batch_i["target_phase"].to(device)
        patients = batch_i["patient"]
        with torch.no_grad():
            u = decoder(ct, rp, tp)
            mag = torch.linalg.vector_norm(u, dim=1, keepdim=True)
            s = head(torch.cat([ct, mag], dim=1))
        m = mask[:, :1] > 0.5
        for i, pid in enumerate(patients):
            mi = m[i, 0]
            if not mi.any():
                continue
            slot = acc.setdefault(pid, {"s": 0.0, "n": 0.0, "abs_u": 0.0, "abs_gt": 0.0, "l1": 0.0, "l1_1": 0.0})
            n = float(mi.sum().item())
            slot["s"] += float(s[i, 0][mi].sum().item())
            slot["n"] += n
            slot["abs_u"] += float(mag[i, 0][mi].sum().item())
            slot["abs_gt"] += float(torch.linalg.vector_norm(gt[i], dim=0)[mi].sum().item())
            slot["l1"] += float(((s[i] * u[i] - gt[i]).abs().mean(dim=0)[mi]).sum().item())
            slot["l1_1"] += float(((u[i] - gt[i]).abs().mean(dim=0)[mi]).sum().item())
    rows = []
    for pid in sorted(acc):
        a = acc[pid]
        n = max(a["n"], 1.0)
        rows.append(
            {
                "patient": pid,
                "mean_s": a["s"] / n,
                "mag_ratio": a["abs_gt"] / max(a["abs_u"], 1e-6),
                "l1_s": a["l1"] / n,
                "l1_s1": a["l1_1"] / n,
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    decoder = load_decoder(device)
    train_pairs, val_pairs, sign = prepare(decoder, device)
    print(
        f"[dir-oracle] device={device} | train pairs={len(train_pairs)} "
        f"val pairs={len(val_pairs)} (hold out C08,C10) | dvf_sign={sign:+.0f}",
        flush=True,
    )
    print("[dir-oracle] NOT an A3 result — DIR Elastix is the training target", flush=True)

    train_loader = make_loader(
        str(DATA), train_pairs, patches=4, random_crop=True,
        batch=args.batch_size, workers=args.workers, shuffle=True,
    )
    val_loader = make_loader(
        str(DATA), val_pairs, patches=2, random_crop=False,
        batch=args.batch_size, workers=0, shuffle=False,
    )
    head = ScaleMapHead().to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    base_tr, base_trs = run_epoch(head, decoder, train_loader, device, opt=None)
    base_va, base_vas = run_epoch(head, decoder, val_loader, device, opt=None)
    print(
        f"Epoch 0/30 | train L1 {base_tr:.4f} | val L1 {base_va:.4f} | "
        f"mean s train/val {base_trs:.3f}/{base_vas:.3f} | s=1 baseline",
        flush=True,
    )

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
                    "kind": "dir_oracle_scale_map",
                    "dvf_sign": sign,
                    "holdout_cases": sorted(HOLD_CASES),
                },
                best_path,
            )
        print(
            f"Epoch {ep}/{args.epochs} | train L1 {tr:.4f} | val L1 {va:.4f} | "
            f"mean s train/val {tr_s:.3f}/{va_s:.3f} | {(time.time()-t0)/60:.1f} min{mark}",
            flush=True,
        )

    rows = per_case(head, decoder, train_pairs + val_pairs, device, args.batch_size)
    print("[dir-oracle] per case (lung). mag_ratio = |Elastix| / |u|. l1_s1 is the s=1 error.", flush=True)
    for row in rows:
        tag = "VAL" if int(row["patient"][1:]) in HOLD_CASES else "train"
        print(
            f"  {row['patient']} {tag:5s}  mean_s={row['mean_s']:.3f}  "
            f"mag_ratio={row['mag_ratio']:.3f}  L1 {row['l1_s1']:.3f} → {row['l1_s']:.3f}",
            flush=True,
        )
    hist["per_case"] = rows
    hist["baseline"] = {"train_l1": base_tr, "val_l1": base_va, "train_s": base_trs, "val_s": base_vas}
    (OUT / "history.json").write_text(json.dumps(hist, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hist["train_l1"], label="train L1")
    ax.plot(hist["val_l1"], label="val L1 (C08,C10)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("lung-masked L1 of s·u vs DIR Elastix")
    ax.set_title("TCIA3_scaleMap DIR oracle")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "dir_oracle_loss.png", dpi=120)
    plt.close(fig)
    print(f"[dir-oracle] best_val_L1={best_val:.4f} → {best_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
