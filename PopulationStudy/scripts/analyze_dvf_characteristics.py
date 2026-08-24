#!/usr/bin/env python3
"""Per-patient DVF motion characteristics + diaphragm landmark trajectories.

Reads PopulationStudy/data/P*/all/ Elastix library and writes:

  PopulationStudy/DVFCharacteristics/
    DVFCharacteristics.md
    metrics_per_patient.tsv
    metrics_per_pair.tsv
    plots/
      motion_ranking.png
      motion_heatmap_all.png
      P0k_diaphragm_trajectory.png
      P0k_diaphragm_qc.png

Usage:
  cd PopulationStudy
  python scripts/analyze_dvf_characteristics.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

POP = Path(__file__).resolve().parents[1]
DATA = POP / "data"
OUT = POP / "DVFCharacteristics"
PLOTS = OUT / "plots"
PHASES = list(range(1, 11))


def mag(u: np.ndarray) -> np.ndarray:
    """‖u‖ from (D,H,W,3) with channels (dx, dy, dz)."""
    return np.linalg.norm(u.astype(np.float64), axis=-1)


def load_pair(patient_dir: Path, ref: int, tgt: int) -> np.ndarray:
    return np.load(patient_dir / f"{ref:02d}_to_{tgt:02d}_pair.npy")


def diaphragm_landmark(mask: np.ndarray) -> tuple[int, int, int]:
    """Pick a lung voxel near the caudal diaphragm dome.

    ZYX convention: take the most inferior lung slice that still has enough
    mask voxels, then the centroid of that slice (near dome apex of that cut).
    Inferior = largest Z index (caudal in this SPARE crop orientation).
    """
    z_has = np.where(mask.sum(axis=(1, 2)) > 50)[0]
    if z_has.size == 0:
        z_has = np.where(mask.any(axis=(1, 2)))[0]
    z = int(z_has.max())
    # Prefer a slice a few voxels above the very edge (more stable dome)
    z = int(max(z_has.min(), z - 2))
    yy, xx = np.where(mask[z] > 0)
    if yy.size == 0:
        z = int(z_has.max())
        yy, xx = np.where(mask[z] > 0)
    y = int(np.round(yy.mean()))
    x = int(np.round(xx.mean()))
    # Snap to nearest lung voxel if centroid fell in a hole
    if mask[z, y, x] == 0:
        d2 = (yy - y) ** 2 + (xx - x) ** 2
        i = int(np.argmin(d2))
        y, x = int(yy[i]), int(xx[i])
    return z, y, x


def sample_u(u: np.ndarray, z: int, y: int, x: int) -> np.ndarray:
    return u[z, y, x].astype(np.float64)  # (dx, dy, dz)


def patient_stats(pid: str) -> dict:
    d = DATA / pid / "all"
    mask = np.load(d / "Mask_Lung.npy").astype(bool)
    ct01 = np.load(d / "CT_01.npy")
    z, y, x = diaphragm_landmark(mask)

    pair_rows = []
    mags_all = []
    for ref in PHASES:
        for tgt in PHASES:
            u = load_pair(d, ref, tgt)
            m = mag(u)[mask]
            mean_m = float(m.mean()) if m.size else 0.0
            p95_m = float(np.percentile(m, 95)) if m.size else 0.0
            max_m = float(m.max()) if m.size else 0.0
            identity = ref == tgt
            pair_rows.append(
                {
                    "patient": pid,
                    "ref": ref,
                    "tgt": tgt,
                    "identity": identity,
                    "L1_mean_lung": mean_m,  # mean magnitude
                    "mag_mean_lung": mean_m,
                    "mag_p95_lung": p95_m,
                    "mag_max_lung": max_m,
                }
            )
            if not identity:
                mags_all.append(mean_m)

    # Trajectories: phase 01 → t (displacement of diaphragm landmark)
    traj = []
    for tgt in PHASES:
        u = load_pair(d, 1, tgt)
        dx, dy, dz = sample_u(u, z, y, x)
        traj.append(
            {
                "phase": tgt,
                "dx": float(dx),
                "dy": float(dy),
                "dz": float(dz),
                "mag": float(np.sqrt(dx * dx + dy * dy + dz * dz)),
            }
        )

    # Max-excursion pair among directed (by lung mean mag)
    directed = [r for r in pair_rows if not r["identity"]]
    best = max(directed, key=lambda r: r["mag_mean_lung"])
    from01 = [r for r in directed if r["ref"] == 1]
    best01 = max(from01, key=lambda r: r["mag_mean_lung"]) if from01 else best

    return {
        "patient": pid,
        "diaphragm_zyx": (z, y, x),
        "n_lung": int(mask.sum()),
        "mag_mean_directed": float(np.mean(mags_all)),
        "mag_p95_mean_directed": float(
            np.mean([r["mag_p95_lung"] for r in directed])
        ),
        "mag_max_pair": best["mag_mean_lung"],
        "max_pair": f"{best['ref']:02d}_to_{best['tgt']:02d}",
        "mag_01_to_max": best01["mag_mean_lung"],
        "pair_01_to_max": f"01_to_{best01['tgt']:02d}",
        "mag_01_to_06": next(
            r["mag_mean_lung"] for r in pair_rows if r["ref"] == 1 and r["tgt"] == 6
        ),
        "traj_peak_mag": max(t["mag"] for t in traj),
        "traj_peak_phase": max(traj, key=lambda t: t["mag"])["phase"],
        "traj": traj,
        "pair_rows": pair_rows,
        "mask": mask,
        "ct01": ct01,
    }


def plot_ranking(summaries: list[dict], path: Path):
    order = sorted(summaries, key=lambda s: s["mag_mean_directed"], reverse=True)
    labels = [s["patient"] for s in order]
    means = [s["mag_mean_directed"] for s in order]
    p01 = [s["mag_01_to_06"] for s in order]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, means, w, label="mean ‖u‖ all directed pairs", color="#3b6ea5")
    ax.bar(x + w / 2, p01, w, label="mean ‖u‖ 01→06", color="#c45c26")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Lung-masked mean ‖DVF‖ (resampled voxels)")
    ax.set_title("SPARE population — per-patient motion ranking")
    ax.legend(frameon=False)
    ax.axhline(np.mean(means), color="gray", ls="--", lw=0.8, label="cohort mean")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_heatmaps(summaries: list[dict], path: Path):
    n = len(summaries)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.2 * rows))
    axes = np.atleast_2d(axes)
    for i, s in enumerate(summaries):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        mat = np.zeros((10, 10), dtype=np.float64)
        for row in s["pair_rows"]:
            mat[row["ref"] - 1, row["tgt"] - 1] = row["mag_mean_lung"]
        im = ax.imshow(mat, cmap="magma", vmin=0, vmax=max(2.5, mat.max()))
        ax.set_title(f"{s['patient']}  mean={s['mag_mean_directed']:.2f}")
        ax.set_xlabel("tgt phase")
        ax.set_ylabel("ref phase")
        ax.set_xticks(range(10))
        ax.set_xticklabels([str(p) for p in PHASES], fontsize=7)
        ax.set_yticks(range(10))
        ax.set_yticklabels([str(p) for p in PHASES], fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")
    fig.suptitle("Lung-masked mean ‖DVF‖ per directed pair", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory(s: dict, path: Path):
    t = s["traj"]
    phases = [r["phase"] for r in t]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax = axes[0]
    ax.plot(phases, [r["dx"] for r in t], "o-", label="dx (LR)", ms=4)
    ax.plot(phases, [r["dy"] for r in t], "o-", label="dy (AP)", ms=4)
    ax.plot(phases, [r["dz"] for r in t], "o-", label="dz (SI)", ms=4)
    ax.plot(phases, [r["mag"] for r in t], "k--", label="‖u‖", lw=1.5)
    ax.set_xlabel("Target phase (ref = 01)")
    ax.set_ylabel("Displacement (resampled voxels)")
    ax.set_title(f"{s['patient']} diaphragm landmark trajectory")
    ax.set_xticks(PHASES)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot([r["dx"] for r in t], [r["dz"] for r in t], "o-", color="#3b6ea5")
    for r in t:
        ax.annotate(str(r["phase"]), (r["dx"], r["dz"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("dx (LR)")
    ax.set_ylabel("dz (SI)")
    ax.set_title("Path in LR–SI plane")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_diaphragm_qc(s: dict, path: Path):
    """CT slice QC: mark diaphragm landmark + 01→06 magnitude overlay."""
    z, y, x = s["diaphragm_zyx"]
    ct = s["ct01"]
    mask = s["mask"]
    d = DATA / s["patient"] / "all"
    u06 = load_pair(d, 1, 6)
    m06 = mag(u06)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    # Coronal (fixed y)
    ax = axes[0]
    ax.imshow(ct[:, y, :].T, cmap="gray", origin="lower", aspect="auto")
    ax.contour(mask[:, y, :].T, levels=[0.5], colors="lime", linewidths=0.6, origin="lower")
    ax.plot(z, x, "r+", ms=12, mew=2)
    ax.set_title(f"{s['patient']} coronal @ y={y}")
    ax.set_xlabel("Z")
    ax.set_ylabel("X")

    # Sagittal (fixed x)
    ax = axes[1]
    ax.imshow(ct[:, :, x].T, cmap="gray", origin="lower", aspect="auto")
    ax.contour(mask[:, :, x].T, levels=[0.5], colors="lime", linewidths=0.6, origin="lower")
    ax.plot(z, y, "r+", ms=12, mew=2)
    ax.set_title(f"sagittal @ x={x}")
    ax.set_xlabel("Z")
    ax.set_ylabel("Y")

    # Axial at landmark Z with ‖u‖ 01→06
    ax = axes[2]
    ax.imshow(ct[z], cmap="gray", origin="lower")
    overlay = np.ma.masked_where(~mask[z], m06[z])
    im = ax.imshow(overlay, cmap="hot", origin="lower", alpha=0.55, vmin=0, vmax=max(3.0, float(overlay.max())))
    ax.plot(x, y, "c+", ms=12, mew=2)
    ax.set_title("axial Z + ‖u‖ 01→06")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"{s['patient']} diaphragm landmark (z,y,x)=({z},{y},{x})  ·  01→06 lung mean ‖u‖={s['mag_01_to_06']:.2f}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_markdown(summaries: list[dict], path: Path):
    ranked = sorted(summaries, key=lambda s: s["mag_mean_directed"], reverse=True)
    farthest = ranked[0]
    lowest = ranked[-1]
    lines = []
    lines.append("# DVF characteristics — SPARE P1–P9")
    lines.append("")
    lines.append("Source: `PopulationStudy/data/P*/all/` Elastix B-spline library (128³, lung-masked).")
    lines.append("")
    lines.append("**Units:** resampled voxels after per-patient lung-bbox → 128³.")
    lines.append("Physical mm/voxel differs by patient (see `ChangesNeeded.md` H2); rankings within this table are still valid in the same units used for PopulationStudy QC.")
    lines.append("")
    lines.append("DVF channels = `(dx, dy, dz)` = (LR, AP, SI) in voxel space.")
    lines.append("")
    lines.append("## Ranking")
    lines.append("")
    lines.append(f"- **Farthest motion:** **{farthest['patient']}** — mean directed ‖u‖ = **{farthest['mag_mean_directed']:.3f}** (01→06 = {farthest['mag_01_to_06']:.3f}).")
    lines.append(f"- **Lowest motion:** **{lowest['patient']}** — mean directed ‖u‖ = **{lowest['mag_mean_directed']:.3f}** (01→06 = {lowest['mag_01_to_06']:.3f}).")
    lines.append(f"- Cohort mean directed ‖u‖ = **{np.mean([s['mag_mean_directed'] for s in summaries]):.3f}**.")
    lines.append("")
    lines.append("![motion ranking](plots/motion_ranking.png)")
    lines.append("")
    lines.append("## Per-patient summary")
    lines.append("")
    lines.append("| Patient | Mean ‖u‖ (all directed) | 01→06 ‖u‖ | Peak pair (by mean ‖u‖) | Peak ‖u‖ | Diaphragm (z,y,x) | Traj peak ‖u‖ @ phase |")
    lines.append("|---------|-------------------------|-----------|-------------------------|----------|-------------------|------------------------|")
    for s in ranked:
        z, y, x = s["diaphragm_zyx"]
        lines.append(
            f"| {s['patient']} | {s['mag_mean_directed']:.3f} | {s['mag_01_to_06']:.3f} | "
            f"`{s['max_pair']}` | {s['mag_max_pair']:.3f} | ({z},{y},{x}) | "
            f"{s['traj_peak_mag']:.2f} @ {s['traj_peak_phase']:02d} |"
        )
    lines.append("")
    lines.append("## Pair-wise heatmaps")
    lines.append("")
    lines.append("Each panel is mean lung-masked ‖DVF‖ for ref→tgt (diagonal = identity ≈ 0).")
    lines.append("")
    lines.append("![heatmaps](plots/motion_heatmap_all.png)")
    lines.append("")
    lines.append("## Diaphragm landmark trajectories")
    lines.append("")
    lines.append("Landmark = centroid of a near-caudal lung-mask slice (diaphragm dome region) on phase **01**.")
    lines.append("Displacement sampled from `01_to_{phase}_pair.npy` at that fixed grid index.")
    lines.append("")
    for s in ranked:
        lines.append(f"### {s['patient']}")
        lines.append("")
        lines.append(f"![traj](plots/{s['patient']}_diaphragm_trajectory.png)")
        lines.append("")
        lines.append(f"![qc](plots/{s['patient']}_diaphragm_qc.png)")
        lines.append("")
        lines.append("| Phase | dx | dy | dz | ‖u‖ |")
        lines.append("|------:|---:|---:|---:|----:|")
        for t in s["traj"]:
            lines.append(
                f"| {t['phase']:02d} | {t['dx']:.3f} | {t['dy']:.3f} | {t['dz']:.3f} | {t['mag']:.3f} |"
            )
        lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("| File | Content |")
    lines.append("|------|---------|")
    lines.append("| `metrics_per_patient.tsv` | Ranking + landmark + summary scalars |")
    lines.append("| `metrics_per_pair.tsv` | All 100 pairs × 9 patients |")
    lines.append("| `summary.json` | Machine-readable copy of the ranking |")
    lines.append("| `plots/` | Ranking, heatmaps, trajectories, QC overlays |")
    lines.append("")
    path.write_text("\n".join(lines))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    patients = [f"P{i}" for i in range(1, 10)]
    summaries = []
    for pid in patients:
        print(f"analyzing {pid} ...", flush=True)
        s = patient_stats(pid)
        summaries.append(s)
        plot_trajectory(s, PLOTS / f"{pid}_diaphragm_trajectory.png")
        plot_diaphragm_qc(s, PLOTS / f"{pid}_diaphragm_qc.png")

    plot_ranking(summaries, PLOTS / "motion_ranking.png")
    plot_heatmaps(summaries, PLOTS / "motion_heatmap_all.png")
    write_markdown(summaries, OUT / "DVFCharacteristics.md")

    ranked = sorted(summaries, key=lambda s: s["mag_mean_directed"], reverse=True)

    # TSVs
    with (OUT / "metrics_per_patient.tsv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "patient",
                "mag_mean_directed",
                "mag_01_to_06",
                "mag_max_pair",
                "max_pair",
                "pair_01_to_max",
                "mag_01_to_max",
                "diaphragm_z",
                "diaphragm_y",
                "diaphragm_x",
                "traj_peak_mag",
                "traj_peak_phase",
                "rank_farthest_1_is_most",
            ],
            delimiter="\t",
        )
        w.writeheader()
        for rank, s in enumerate(ranked, 1):
            z, y, x = s["diaphragm_zyx"]
            w.writerow(
                {
                    "patient": s["patient"],
                    "mag_mean_directed": f"{s['mag_mean_directed']:.6f}",
                    "mag_01_to_06": f"{s['mag_01_to_06']:.6f}",
                    "mag_max_pair": f"{s['mag_max_pair']:.6f}",
                    "max_pair": s["max_pair"],
                    "pair_01_to_max": s["pair_01_to_max"],
                    "mag_01_to_max": f"{s['mag_01_to_max']:.6f}",
                    "diaphragm_z": z,
                    "diaphragm_y": y,
                    "diaphragm_x": x,
                    "traj_peak_mag": f"{s['traj_peak_mag']:.6f}",
                    "traj_peak_phase": s["traj_peak_phase"],
                    "rank_farthest_1_is_most": rank,
                }
            )

    with (OUT / "metrics_per_pair.tsv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "patient",
                "ref",
                "tgt",
                "identity",
                "mag_mean_lung",
                "mag_p95_lung",
                "mag_max_lung",
            ],
            delimiter="\t",
        )
        w.writeheader()
        for s in summaries:
            for row in s["pair_rows"]:
                w.writerow(
                    {
                        "patient": row["patient"],
                        "ref": row["ref"],
                        "tgt": row["tgt"],
                        "identity": int(row["identity"]),
                        "mag_mean_lung": f"{row['mag_mean_lung']:.6f}",
                        "mag_p95_lung": f"{row['mag_p95_lung']:.6f}",
                        "mag_max_lung": f"{row['mag_max_lung']:.6f}",
                    }
                )

    summary_json = {
        "farthest": ranked[0]["patient"],
        "lowest": ranked[-1]["patient"],
        "cohort_mean_directed": float(np.mean([s["mag_mean_directed"] for s in summaries])),
        "patients": [
            {
                "patient": s["patient"],
                "mag_mean_directed": s["mag_mean_directed"],
                "mag_01_to_06": s["mag_01_to_06"],
                "diaphragm_zyx": list(s["diaphragm_zyx"]),
                "traj_peak_mag": s["traj_peak_mag"],
                "traj_peak_phase": s["traj_peak_phase"],
            }
            for s in ranked
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary_json, indent=2))

    # thin README pointer
    (OUT / "README.md").write_text(
        "# DVFCharacteristics\n\n"
        "See **[DVFCharacteristics.md](DVFCharacteristics.md)** for rankings, "
        "diaphragm trajectories, and QC figures.\n\n"
        "Regenerate:\n\n```bash\n"
        "cd PopulationStudy\n"
        "python scripts/analyze_dvf_characteristics.py\n"
        "```\n"
    )
    print(f"wrote {OUT}")
    print(f"farthest={ranked[0]['patient']}  lowest={ranked[-1]['patient']}")


if __name__ == "__main__":
    main()
