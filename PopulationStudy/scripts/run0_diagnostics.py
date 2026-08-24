#!/usr/bin/env python3
"""Run 0 / P0-B diagnostics (no training).

1. LOO population atlas (+ oracle rescale)
2. Oracle-scale ablation on existing CRB checkpoints
3. Amplitude ratio ‖pred‖/‖gt‖
4. GT-warp sanity floor

Writes PopulationStudy/Run0_Diagnostics/

  cd PopulationStudy
  PYTHONPATH=InitialExperiments/Experiment1 python scripts/run0_diagnostics.py --gpu 0
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

POP = Path(__file__).resolve().parents[1]
DATA = POP / "data"
OUT = POP / "Run0_Diagnostics"
INIT = POP / "InitialExperiments"
E1 = INIT / "Experiment1"
E2 = INIT / "Experiment2"
E4 = INIT / "Experiment4"

PHASES = list(range(1, 11))


def mag(u_cdhw: np.ndarray) -> np.ndarray:
    return np.linalg.norm(u_cdhw.astype(np.float64), axis=0)


def to_cdhw(dvf: np.ndarray) -> np.ndarray:
    if dvf.ndim == 4 and dvf.shape[-1] == 3:
        return np.moveaxis(dvf, -1, 0)
    return dvf


def load_dvf(pid: str, ref: int, tgt: int) -> np.ndarray:
    if ref == tgt:
        return np.zeros((3, 128, 128, 128), dtype=np.float32)
    return to_cdhw(np.load(DATA / pid / "all" / f"{ref:02d}_to_{tgt:02d}_pair.npy"))


def load_mask(pid: str) -> np.ndarray:
    return (np.load(DATA / pid / "all" / "Mask_Lung.npy") > 0).astype(np.float32)


def lung_l1(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    m = mask > 0.5
    diff = np.abs(a - b)  # (3,D,H,W)
    return float(diff[:, m].mean()) if m.any() else 0.0


def lung_cos(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    m = mask > 0.5
    aa = a[:, m]
    bb = b[:, m]
    num = float((aa * bb).sum())
    den = float(np.sqrt((aa * aa).sum() * (bb * bb).sum()) + 1e-8)
    return num / den


def mean_mag(u: np.ndarray, mask: np.ndarray) -> float:
    return float(mag(u)[mask > 0.5].mean()) if (mask > 0.5).any() else 0.0


def oracle_alpha(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """L2-optimal global scale: argmin_α ‖α pred − gt‖² on lung."""
    m = mask > 0.5
    num = float((pred[:, m] * gt[:, m]).sum())
    den = float((pred[:, m] * pred[:, m]).sum()) + 1e-8
    return num / den


def zero_l1(gt: np.ndarray, mask: np.ndarray) -> float:
    return lung_l1(np.zeros_like(gt), gt, mask)


# ---------------------------------------------------------------------------
# Atlas
# ---------------------------------------------------------------------------


def run_atlas() -> list[dict]:
    patients = [f"P{i}" for i in range(1, 10)]
    masks = {p: load_mask(p) for p in patients}
    rows = []
    for left in patients:
        print(f"[atlas] leave-out {left}", flush=True)
        others = [p for p in patients if p != left]
        mask = masks[left]
        for ref in PHASES:
            for tgt in PHASES:
                if ref == tgt:
                    continue
                stack = np.stack([load_dvf(p, ref, tgt) for p in others], axis=0)
                atlas = stack.mean(axis=0).astype(np.float32)
                gt = load_dvf(left, ref, tgt)
                z = zero_l1(gt, mask)
                l1 = lung_l1(atlas, gt, mask)
                alpha = oracle_alpha(atlas, gt, mask)
                l1_s = lung_l1(alpha * atlas, gt, mask)
                cos = lung_cos(atlas, gt, mask)
                rows.append(
                    {
                        "patient": left,
                        "ref": ref,
                        "tgt": tgt,
                        "L1_atlas": l1,
                        "L1_zero": z,
                        "L1_atlas_oracle": l1_s,
                        "alpha_oracle": alpha,
                        "cos": cos,
                        "ratio_atlas": l1 / z if z > 1e-8 else float("nan"),
                        "ratio_oracle": l1_s / z if z > 1e-8 else float("nan"),
                    }
                )
    return rows


def summarize_atlas(rows: list[dict]) -> list[dict]:
    out = []
    for pid in sorted({r["patient"] for r in rows}):
        sub = [r for r in rows if r["patient"] == pid]
        n = len(sub)
        out.append(
            {
                "patient": pid,
                "n": n,
                "L1_atlas": float(np.mean([r["L1_atlas"] for r in sub])),
                "L1_zero": float(np.mean([r["L1_zero"] for r in sub])),
                "L1_atlas_oracle": float(np.mean([r["L1_atlas_oracle"] for r in sub])),
                "alpha_oracle": float(np.mean([r["alpha_oracle"] for r in sub])),
                "cos": float(np.mean([r["cos"] for r in sub])),
                "ratio_atlas": float(np.nanmean([r["ratio_atlas"] for r in sub])),
                "ratio_oracle": float(np.nanmean([r["ratio_oracle"] for r in sub])),
                "beat_zero_pct": 100.0
                * sum(r["L1_atlas"] < r["L1_zero"] for r in sub)
                / n,
            }
        )
    return out


# ---------------------------------------------------------------------------
# GT-warp sanity
# ---------------------------------------------------------------------------


def _norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def run_gt_warp(device: torch.device) -> list[dict]:
    sys.path.insert(0, str(E1))
    from utilities.warp import warp  # noqa: E402

    patients = [f"P{i}" for i in range(1, 10)]
    pairs = [(1, 6), (1, 3), (5, 10)]
    rows = []
    for pid in patients:
        mask = load_mask(pid)
        for ref, tgt in pairs:
            ct_r = _norm(np.load(DATA / pid / "all" / f"CT_{ref:02d}.npy"))
            ct_t = _norm(np.load(DATA / pid / "all" / f"CT_{tgt:02d}.npy"))
            gt = load_dvf(pid, ref, tgt)
            ref_t = torch.from_numpy(ct_r)[None, None].to(device)
            gt_t = torch.from_numpy(gt)[None].to(device)
            with torch.no_grad():
                warped = warp(ref_t, gt_t)[0, 0].cpu().numpy()
            err = np.abs(ct_t - warped)
            err0 = np.abs(ct_t - ct_r)
            m = mask > 0.5
            rows.append(
                {
                    "patient": pid,
                    "pair": f"{ref:02d}_to_{tgt:02d}",
                    "img_L1_gt_warp": float(err[m].mean()),
                    "img_L1_identity": float(err0[m].mean()),
                    "ratio_gt_vs_id": float(err[m].mean() / (err0[m].mean() + 1e-8)),
                }
            )
            print(
                f"[gt-warp] {pid} {ref:02d}→{tgt:02d} "
                f"gt={rows[-1]['img_L1_gt_warp']:.4f} id={rows[-1]['img_L1_identity']:.4f}",
                flush=True,
            )
    return rows


# ---------------------------------------------------------------------------
# Model oracle-scale
# ---------------------------------------------------------------------------


MODELS = [
    {
        "tag": "E1_Decoder",
        "exp": E1,
        "arch": "decoder",
        "ckpt": E1 / "DecoderCRB" / "weights" / "crb_dec_mse_pop_e1_generator.pth",
        "holdout": ["P3", "P4", "P5"],
        "needs_anatomy": False,
    },
    {
        "tag": "E2_Encoder",
        "exp": E2,
        "arch": "encoder",
        "ckpt": E2 / "EncoderCRB" / "weights" / "crb_enc_mse_pop_e2_generator.pth",
        "holdout": ["P7", "P9"],
        "needs_anatomy": False,
    },
    {
        "tag": "E4_Decoder",
        "exp": E4,
        "arch": "decoder",
        "ckpt": E4 / "DecoderCRB" / "weights" / "crb_dec_mse_pop_e4_generator.pth",
        "holdout": ["P7", "P9"],
        "needs_anatomy": True,
        "anatomy": E4 / "AnatomyAE" / "weights" / "anatomy_encoder.pth",
    },
]


def build_model(spec: dict, device: torch.device):
    exp = spec["exp"]
    sys.path.insert(0, str(exp))
    # clear cached modules from other experiments
    for k in list(sys.modules):
        if k.startswith(("networks", "losses", "utilities")):
            del sys.modules[k]
    from networks.generator_crb import UNetCRB  # noqa: E402
    from networks.generator_crb_both import UNetCRBBoth  # noqa: E402
    from networks.generator_crb_dec import UNetCRBDecoder  # noqa: E402

    arch = {"encoder": UNetCRB, "decoder": UNetCRBDecoder, "both": UNetCRBBoth}[
        spec["arch"]
    ]
    if spec.get("needs_anatomy") and spec["arch"] in ("encoder", "both"):
        g = arch(im_size=128, n_phases=10, anatomy_dim=32).to(device)
        g.load_anatomy_encoder(str(spec["anatomy"]))
    else:
        g = arch(im_size=128, n_phases=10).to(device)
        if spec.get("needs_anatomy") and hasattr(g, "load_anatomy_encoder"):
            # Decoder: anatomy may already be in state_dict; still call if needed before load
            pass
    sd = torch.load(spec["ckpt"], map_location=device)
    g.load_state_dict(sd, strict=False)
    g.eval()
    return g


def run_model_oracle(spec: dict, device: torch.device) -> list[dict]:
    g = build_model(spec, device)
    rows = []
    for pid in spec["holdout"]:
        mask = load_mask(pid)
        print(f"[{spec['tag']}] {pid}", flush=True)
        for ref in PHASES:
            for tgt in PHASES:
                if ref == tgt:
                    continue
                ct = np.load(DATA / pid / "all" / f"CT_{ref:02d}.npy").astype(np.float32)
                lo, hi = float(ct.min()), float(ct.max())
                if hi > lo:
                    ct = (ct - lo) / (hi - lo)
                gt = load_dvf(pid, ref, tgt)
                with torch.no_grad():
                    pred = (
                        g(
                            torch.from_numpy(ct)[None, None].to(device),
                            torch.tensor([ref - 1], device=device),
                            torch.tensor([tgt - 1], device=device),
                        )[0]
                        .cpu()
                        .numpy()
                    )
                z = zero_l1(gt, mask)
                l1 = lung_l1(pred, gt, mask)
                alpha = oracle_alpha(pred, gt, mask)
                l1_s = lung_l1(alpha * pred, gt, mask)
                cos = lung_cos(pred, gt, mask)
                r_amp = mean_mag(pred, mask) / (mean_mag(gt, mask) + 1e-8)
                rows.append(
                    {
                        "model": spec["tag"],
                        "patient": pid,
                        "ref": ref,
                        "tgt": tgt,
                        "L1_pred": l1,
                        "L1_zero": z,
                        "L1_oracle": l1_s,
                        "alpha_oracle": alpha,
                        "cos": cos,
                        "amp_ratio": r_amp,
                        "ratio_pred": l1 / z if z > 1e-8 else float("nan"),
                        "ratio_oracle": l1_s / z if z > 1e-8 else float("nan"),
                    }
                )
    # free GPU
    del g
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return rows


def summarize_model(rows: list[dict]) -> list[dict]:
    out = []
    keys = sorted({(r["model"], r["patient"]) for r in rows})
    for model, pid in keys:
        sub = [r for r in rows if r["model"] == model and r["patient"] == pid]
        n = len(sub)
        out.append(
            {
                "model": model,
                "patient": pid,
                "n": n,
                "L1_pred": float(np.mean([r["L1_pred"] for r in sub])),
                "L1_zero": float(np.mean([r["L1_zero"] for r in sub])),
                "L1_oracle": float(np.mean([r["L1_oracle"] for r in sub])),
                "alpha_oracle": float(np.mean([r["alpha_oracle"] for r in sub])),
                "cos": float(np.mean([r["cos"] for r in sub])),
                "amp_ratio": float(np.mean([r["amp_ratio"] for r in sub])),
                "ratio_pred": float(np.nanmean([r["ratio_pred"] for r in sub])),
                "ratio_oracle": float(np.nanmean([r["ratio_oracle"] for r in sub])),
                "beat_zero_pct": 100.0
                * sum(r["L1_pred"] < r["L1_zero"] for r in sub)
                / n,
                "beat_zero_oracle_pct": 100.0
                * sum(r["L1_oracle"] < r["L1_zero"] for r in sub)
                / n,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Plots + report
# ---------------------------------------------------------------------------


def plot_oracle_bars(model_sum: list[dict], path: Path):
    # one group per (model, patient)
    labels = [f"{r['model']}\n{r['patient']}" for r in model_sum]
    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(labels)), 4.2))
    ax.bar(x - w, [r["ratio_pred"] for r in model_sum], w, label="L1/zero pred", color="#3b6ea5")
    ax.bar(x, [r["ratio_oracle"] for r in model_sum], w, label="L1/zero after oracle α", color="#c45c26")
    ax.bar(x + w, [r["amp_ratio"] for r in model_sum], w, label="‖pred‖/‖gt‖", color="#5a8f5a")
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("ratio")
    ax.set_title("Run 0 — oracle-scale ablation on hold-outs")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_atlas_bars(atlas_sum: list[dict], path: Path):
    order = sorted(atlas_sum, key=lambda r: r["ratio_atlas"])
    labels = [r["patient"] for r in order]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, [r["ratio_atlas"] for r in order], w, label="atlas L1/zero", color="#3b6ea5")
    ax.bar(x + w / 2, [r["ratio_oracle"] for r in order], w, label="atlas+oracle α", color="#c45c26")
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("L1 / zero")
    ax.set_title("LOO population atlas vs identity (all directed pairs)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_report(
    atlas_sum: list[dict],
    model_sum: list[dict],
    gt_sum: list[dict],
    path: Path,
):
    # H1 check: does oracle fix E2-style patients?
    e2 = [r for r in model_sum if r["model"] == "E2_Encoder"]
    e1 = [r for r in model_sum if r["model"] == "E1_Decoder"]
    lines = [
        "# Run 0 diagnostics (P0-B)",
        "",
        "No training. Confirms whether **amplitude** (H1) is the dominant population failure.",
        "",
        "## Verdict checklist",
        "",
    ]
    if e2:
        p7 = next((r for r in e2 if r["patient"] == "P7"), None)
        if p7:
            fixed = p7["ratio_oracle"] <= 0.75
            lines.append(
                f"- **E2 Encoder P7:** L1/zero {p7['ratio_pred']:.2f} → oracle {p7['ratio_oracle']:.2f} "
                f"(α≈{p7['alpha_oracle']:.2f}, amp_ratio={p7['amp_ratio']:.2f}). "
                + ("**H1 supported** (oracle ≤ 0.75)." if fixed else "Oracle did not fully fix; shape may also bind.")
            )
    if e1:
        lines.append(
            "- **E1 Decoder hold-outs:** "
            + ", ".join(
                f"{r['patient']} {r['ratio_pred']:.2f}→{r['ratio_oracle']:.2f}" for r in e1
            )
        )
    lines += [
        "",
        "## 1. LOO population atlas",
        "",
        "Mean DVF of the other 8 patients on the shared 128³ grid (same phase pair). No learning.",
        "",
        "![atlas](plots/atlas_l1_over_zero.png)",
        "",
        "| Patient | L1/zero atlas | L1/zero atlas+α | mean α | cos | beat zero % |",
        "|---------|---------------|-----------------|--------|-----|-------------|",
    ]
    for r in sorted(atlas_sum, key=lambda x: x["ratio_atlas"]):
        lines.append(
            f"| {r['patient']} | {r['ratio_atlas']:.2f} | {r['ratio_oracle']:.2f} | "
            f"{r['alpha_oracle']:.2f} | {r['cos']:.3f} | {r['beat_zero_pct']:.0f}% |"
        )
    lines += [
        "",
        "## 2. Model oracle-scale (hold-outs)",
        "",
        "![oracle](plots/model_oracle_scale.png)",
        "",
        "| Model | Patient | L1/zero | oracle L1/zero | α | ‖pred‖/‖gt‖ | cos | beat% | beat% after α |",
        "|-------|---------|---------|----------------|---|-------------|-----|-------|---------------|",
    ]
    for r in model_sum:
        lines.append(
            f"| {r['model']} | {r['patient']} | {r['ratio_pred']:.2f} | {r['ratio_oracle']:.2f} | "
            f"{r['alpha_oracle']:.2f} | {r['amp_ratio']:.2f} | {r['cos']:.3f} | "
            f"{r['beat_zero_pct']:.0f}% | {r['beat_zero_oracle_pct']:.0f}% |"
        )
    # GT warp per patient mean
    lines += [
        "",
        "## 3. GT-warp sanity",
        "",
        "Lung-masked image L1 of `warp(ref, gt_dvf)` vs target CT (normed). "
        "Should be **much smaller** than identity warp residual.",
        "",
        "| Patient | mean img L1 (GT warp) | mean img L1 (identity) | GT/id |",
        "|---------|-----------------------|------------------------|-------|",
    ]
    for pid in [f"P{i}" for i in range(1, 10)]:
        sub = [r for r in gt_sum if r["patient"] == pid]
        a = float(np.mean([r["img_L1_gt_warp"] for r in sub]))
        b = float(np.mean([r["img_L1_identity"] for r in sub]))
        lines.append(f"| {pid} | {a:.4f} | {b:.4f} | {a / (b + 1e-8):.2f} |")
    lines += [
        "",
        "## Files",
        "",
        "- `atlas_per_pair.tsv` / `atlas_per_patient.tsv`",
        "- `model_oracle_per_pair.tsv` / `model_oracle_per_patient.tsv`",
        "- `gt_warp.tsv`",
        "- `summary.json`",
        "",
        "Next (still Run 0): **P0-A** common-mm regrid · **P0-C** LOPO model selection. "
        "Then **E6** amplitude-decoupled Decoder if H1 holds.",
        "",
    ]
    path.write_text("\n".join(lines))


def write_tsv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--skip_models", action="store_true")
    ap.add_argument("--skip_atlas", action="store_true")
    ap.add_argument("--skip_gt_warp", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(exist_ok=True)
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    print(f"device={device}", flush=True)

    atlas_rows, atlas_sum = [], []
    if not args.skip_atlas:
        atlas_rows = run_atlas()
        atlas_sum = summarize_atlas(atlas_rows)
        write_tsv(OUT / "atlas_per_pair.tsv", atlas_rows)
        write_tsv(OUT / "atlas_per_patient.tsv", atlas_sum)
        plot_atlas_bars(atlas_sum, OUT / "plots" / "atlas_l1_over_zero.png")

    gt_rows = []
    if not args.skip_gt_warp:
        gt_rows = run_gt_warp(device)
        write_tsv(OUT / "gt_warp.tsv", gt_rows)

    model_rows, model_sum = [], []
    if not args.skip_models:
        for spec in MODELS:
            if not spec["ckpt"].exists():
                print(f"skip missing ckpt {spec['ckpt']}", flush=True)
                continue
            model_rows.extend(run_model_oracle(spec, device))
        model_sum = summarize_model(model_rows)
        write_tsv(OUT / "model_oracle_per_pair.tsv", model_rows)
        write_tsv(OUT / "model_oracle_per_patient.tsv", model_sum)
        if model_sum:
            plot_oracle_bars(model_sum, OUT / "plots" / "model_oracle_scale.png")

    write_report(atlas_sum, model_sum, gt_rows, OUT / "Run0.md")
    summary = {
        "atlas": atlas_sum,
        "models": model_sum,
        "gt_warp_mean_ratio": float(
            np.mean([r["ratio_gt_vs_id"] for r in gt_rows]) if gt_rows else float("nan")
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "README.md").write_text(
        "# Run 0 diagnostics\n\nSee **[Run0.md](Run0.md)**.\n\n"
        "```bash\ncd PopulationStudy\n"
        "PYTHONPATH=InitialExperiments/Experiment1 python scripts/run0_diagnostics.py --gpu 0\n"
        "```\n"
    )
    print(f"wrote {OUT / 'Run0.md'}", flush=True)


if __name__ == "__main__":
    main()
