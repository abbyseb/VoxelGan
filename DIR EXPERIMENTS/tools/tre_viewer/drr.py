"""RTK landmark → detector → 128×128 compress mapping (Phase 3)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# MC/Varian half-fan defaults — matches VoxelMap_Clinical/config/elekta_drr.py
# and DIR EXPERIMENTS prepare_a1_case (mc_varian_drr_opts_for_scan).
DET_ORIGIN = (-200.0, -150.0, 0.0)  # mm
DET_SPACING = (0.388, 0.388, 1.0)  # mm
DET_SIZE_XY = (1024, 768)  # (nx, ny) = (cols, rows)

# Empirically required for A1 R3 C01 landmarks to sit on lung DRRs across angles
# (imshow / ITK row direction). Validated vs SourceProjections/06_Proj_*.
FLIP_ROW_128 = True
FLIP_COL_128 = False


@dataclass(frozen=True)
class GeometryInfo:
    path: Path
    offset_x: float
    offset_y: float
    sid: float
    sdd: float
    matrices: np.ndarray  # (N, 3, 4)
    gantry_deg: np.ndarray  # (N,)

    @property
    def n_views(self) -> int:
        return int(self.matrices.shape[0])


def parse_geometry_xml(path: Path) -> GeometryInfo:
    root = ET.fromstring(Path(path).read_text())

    def _f(tag: str, default: float = 0.0) -> float:
        el = root.find(tag)
        return float(el.text) if el is not None and el.text else default

    mats = []
    angles = []
    for index, proj in enumerate(root.findall("Projection"), start=1):
        m_el = proj.find("Matrix")
        a_el = proj.find("GantryAngle")
        if m_el is None or m_el.text is None:
            raise ValueError(f"Projection {index} has no matrix in {path}")
        vals = [float(x) for x in m_el.text.split()]
        if len(vals) != 12 or not np.isfinite(vals).all():
            raise ValueError(f"Projection {index} needs 12 finite matrix values in {path}")
        mats.append(np.array(vals, dtype=np.float64).reshape(3, 4))
        angles.append(float(a_el.text) if a_el is not None and a_el.text else len(angles))
    if not mats:
        raise ValueError(f"no projection matrices in {path}")
    return GeometryInfo(
        path=Path(path),
        offset_x=_f("ProjectionOffsetX"),
        offset_y=_f("ProjectionOffsetY"),
        sid=_f("SourceToIsocenterDistance", 1000.0),
        sdd=_f("SourceToDetectorDistance", 1500.0),
        matrices=np.stack(mats, axis=0),
        gantry_deg=np.asarray(angles, dtype=np.float64),
    )


def resolve_geometry(run) -> GeometryInfo:
    """Prefer staged per-case Proj/Geometry.xml, then arm Geometry.xml."""
    cands = list(run.geometry_candidates)
    for c in cands:
        p = Path(c)
        if p.is_file() and "staged" in str(p):
            return parse_geometry_xml(p)
    for c in cands:
        p = Path(c)
        if p.is_file():
            return parse_geometry_xml(p)
    raise FileNotFoundError(f"no Geometry.xml for {run.scan_id}")


def itk_index_to_physical(
    xyz_index: np.ndarray,
    origin_xyz: tuple[float, float, float],
    spacing_xyz: tuple[float, float, float],
) -> np.ndarray:
    o = np.asarray(origin_xyz, dtype=np.float64)
    s = np.asarray(spacing_xyz, dtype=np.float64)
    return o + np.asarray(xyz_index, dtype=np.float64) * s


def project_points_rtk(xyz_mm: np.ndarray, matrix_3x4: np.ndarray) -> np.ndarray:
    pts = np.asarray(xyz_mm, dtype=np.float64)
    hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    p = hom @ np.asarray(matrix_3x4, dtype=np.float64).T
    w = p[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return p[:, :2] / w


def detector_mm_to_full_px(
    uv_mm: np.ndarray,
    *,
    origin_xy: tuple[float, float] = DET_ORIGIN[:2],
    spacing_xy: tuple[float, float] = DET_SPACING[:2],
) -> np.ndarray:
    ox, oy = origin_xy
    sx, sy = spacing_xy
    col = (uv_mm[:, 0] - ox) / sx
    row = (uv_mm[:, 1] - oy) / sy
    return np.stack([col, row], axis=1)


def full_px_to_128(
    col_row: np.ndarray,
    *,
    full_hw: tuple[int, int] = (DET_SIZE_XY[1], DET_SIZE_XY[0]),
    flip_row: bool = FLIP_ROW_128,
    flip_col: bool = FLIP_COL_128,
) -> np.ndarray:
    """Full-res (col,row) → 128×128 (col,row) after compress zoom (+ optional flips)."""
    h, w = full_hw
    col = col_row[:, 0] * (128.0 / w)
    row = col_row[:, 1] * (128.0 / h)
    if flip_col:
        col = (128.0 - 1.0) - col
    if flip_row:
        row = (128.0 - 1.0) - row
    return np.stack([col, row], axis=1)


def project_landmarks_to_128(xyz_mm: np.ndarray, matrix_3x4: np.ndarray) -> np.ndarray:
    uv = project_points_rtk(xyz_mm, matrix_3x4)
    return full_px_to_128(detector_mm_to_full_px(uv))


def load_proj_128(path: Path) -> np.ndarray:
    a = np.load(path).astype(np.float32)
    if a.ndim == 2 and a.shape == (128, 128):
        return a
    flat = np.asarray(a).reshape(-1)
    if flat.size == 128 * 128:
        return flat.reshape((128, 128), order="F")
    raise ValueError(f"bad proj shape {a.shape} at {path}")


def phase_to_idx(phase: str) -> int:
    return {
        "T00": 1, "T10": 2, "T20": 3, "T30": 4, "T40": 5,
        "T50": 6, "T60": 7, "T70": 8, "T80": 9, "T90": 10,
    }[phase]


def proj_path(
    mt_dir: Path, phase: str, view_1based: int, *, source: bool = False
) -> Path:
    """Compressed 128² projection path for a respiratory phase.

    VoxelMap ``ModelTraining`` layout: phase **06 (T50)** lives only under
    ``SourceProjections/``; phases 01–05 and 07–10 are under ``TargetProjections/``.
    ``source=True`` always selects the ref (06) folder — legacy VoxelMap pair
    convention. Prefer ``source=False`` with the actual phase string.
    """
    if source:
        idx = 6
        folder = "SourceProjections"
    else:
        idx = phase_to_idx(phase)
        folder = "SourceProjections" if idx == 6 else "TargetProjections"
    return mt_dir / folder / f"{idx:02d}_Proj_{view_1based:03d}_bin.npy"


def r3_ct_physical_landmarks(run, pack_xyz: np.ndarray) -> tuple[np.ndarray, Path]:
    """Pack-frame landmark indices → physical mm in the R3 CT used for DRRs."""
    import SimpleITK as sitk
    from tre_viewer.data import CASE_INFO, train_dir
    from tre_viewer.evaluation import require_evaluator

    (_nx, ny, nz), _ = CASE_INFO[run.case]
    train = train_dir(run.run_root, run.scan_id)
    if train is None:
        raise FileNotFoundError(f"no train folder under {run.run_root}")
    ct_path = train / "CT_06.mha"
    if not ct_path.is_file():
        hits = sorted(train.glob("CT_*.mha"))
        if not hits:
            raise FileNotFoundError(f"no CT_*.mha under {run.run_root}")
        ct_path = hits[0]
    img = sitk.ReadImage(str(ct_path))
    indices = np.asarray(pack_xyz, dtype=np.float64)
    if run.frame == "r3":
        indices = require_evaluator().pack_to_r3(indices, ny, nz)
    elif run.frame != "native":
        raise ValueError("Projection landmarks require a known native/r3 frame")
    # ITK direction may be non-identity; origin + index * spacing alone loses it.
    mm = np.asarray([img.TransformContinuousIndexToPhysicalPoint(tuple(point)) for point in indices])
    return mm, ct_path
