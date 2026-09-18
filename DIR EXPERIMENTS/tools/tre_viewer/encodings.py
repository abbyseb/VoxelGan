"""Visual encodings for TRE Viewer (Phase 1)."""

from __future__ import annotations

import numpy as np

# Fixed TRE colour domain (mm) so colours are comparable across cases.
TRE_VMIN_MM = 0.0
TRE_VMAX_MM = 5.0


def tre_to_rgba(tre_mm: np.ndarray, *, vmin: float = TRE_VMIN_MM, vmax: float = TRE_VMAX_MM):
    """Map TRE mm → RGBA via viridis; over-range → magenta."""
    from matplotlib import colormaps

    t = np.asarray(tre_mm, dtype=np.float64)
    cmap = colormaps["viridis"]
    norm = np.clip((t - vmin) / max(vmax - vmin, 1e-8), 0.0, 1.0)
    rgba = cmap(norm)
    over = t > vmax
    rgba[over] = (1.0, 0.0, 1.0, 1.0)  # magenta
    return rgba.astype(np.float32)


def ring_sizes_mm(tre_mm: np.ndarray, *, min_size: float = 1.5, gain: float = 1.0) -> np.ndarray:
    """Point size in world mm ≈ TRE (readable ring metaphor)."""
    t = np.asarray(tre_mm, dtype=np.float64)
    return np.maximum(t * gain, min_size).astype(np.float64)


def error_vectors_zyx(
    truth_pack_xyz: np.ndarray,
    pred_pack_xyz: np.ndarray,
    spacing_xyz_mm: tuple[float, float, float],
) -> np.ndarray:
    """Napari Vectors (N, 2, 3) in (z,y,x) **data** coords: start=truth, vec=pred-truth.

    Length is in voxel units of the pack grid (not mm) so it aligns with the
    image data coordinates; mm length is encoded separately in features.
    """
    truth = np.asarray(truth_pack_xyz, dtype=np.float64)
    pred = np.asarray(pred_pack_xyz, dtype=np.float64)
    start_zyx = np.stack([truth[:, 2], truth[:, 1], truth[:, 0]], axis=1)
    delta_xyz = pred - truth
    delta_zyx = np.stack([delta_xyz[:, 2], delta_xyz[:, 1], delta_xyz[:, 0]], axis=1)
    out = np.zeros((len(truth), 2, 3), dtype=np.float64)
    out[:, 0, :] = start_zyx
    out[:, 1, :] = delta_zyx
    # unused but kept for callers that want mm length
    _ = spacing_xyz_mm
    return out


def summary_line(pl) -> str:
    r, i = pl.registered_stats, pl.identity_stats
    return (
        f"TRE_{pl.which} {pl.pair.replace('_', '→')}  "
        f"mean {r['mean']:.2f} │ p50 {r['p50']:.2f} │ p95 {r['p95']:.2f} │ "
        f"max {r['max']:.2f} mm   identity {i['mean']:.2f}  Δ {pl.improvement_mm:+.2f}"
    )


def worst_table(pl, *, k: int = 12) -> list[dict]:
    order = np.argsort(-pl.tre_mm)
    rows = []
    for rank, idx in enumerate(order[:k]):
        z = float(pl.truth_pack[idx, 2])
        rows.append(
            {
                "rank": rank + 1,
                "id": int(idx),
                "tre_mm": float(pl.tre_mm[idx]),
                "ident_mm": float(pl.identity_mm[idx]),
                "slice_z": int(round(z)),
            }
        )
    return rows
