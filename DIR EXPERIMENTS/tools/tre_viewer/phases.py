"""75-point phase evaluation, independent of the historical T00→T50 KPI.

Every score maps T50 landmarks to the requested phase. Raw synth pull fields
are inverted numerically; downstream VoxelMap caches already map T50→phase.
No Elastix/training label is substituted for either model's output.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import tempfile

import numpy as np
from scipy.ndimage import map_coordinates

from . import data

PHASES = tuple(data.PHASE_TO_IDX)
STAGES = {"synth": "Direct synthesizer", "voxelmap": "Downstream VoxelMap"}
REFERENCE = "T50"


@dataclass
class PhaseResult:
    run: data.RunRef
    stage: str
    phase: str
    status: str  # ok / reference / unavailable / error
    reason: str = ""
    landmarks: data.PerLandmarkResult | None = None
    observed_mm: np.ndarray | None = None
    predicted_mm: np.ndarray | None = None

    def metric(self, name: str) -> float:
        if self.landmarks is None:
            return float("nan")
        if name == "improvement":
            return self.landmarks.improvement_mm
        return float(self.landmarks.registered_stats[name])

    def to_dict(self) -> dict:
        return {
            "run": str(self.run.run_root), "case": self.run.case,
            "frame": self.run.frame, "stage": self.stage, "phase": self.phase,
            "reference": REFERENCE, "landmark_set": "75", "status": self.status,
            "reason": self.reason, "n": 75 if self.landmarks else 0,
            "mean_mm": self.metric("mean") if self.landmarks else None,
            "p95_mm": self.metric("p95") if self.landmarks else None,
            "improvement_mm": self.metric("improvement") if self.landmarks else None,
            "identity_mean_mm": self.landmarks.identity_stats["mean"] if self.landmarks else None,
            "observed_xyz_mm": self.observed_mm.tolist() if self.observed_mm is not None else None,
            "predicted_xyz_mm": self.predicted_mm.tolist() if self.predicted_mm is not None else None,
        }


def _points(run, phase):
    points = np.asarray(data.landmarks_75(run.case, phase), dtype=float)
    if points.shape != (75, 3) or not np.isfinite(points).all():
        raise ValueError(f"{phase}: expected 75 finite, corresponding xyz landmarks")
    return points


def phase_cache_path(run, phase):
    idx = data.phase_to_train_idx(phase)
    return run.run_root / "tre" / f"voxelmap_dvf_phase{idx:02d}_mean.npy"


def _real_model_training(run, real_runs_dir=None):
    """A3 trains on synth DRRs but evaluates on real A1 DRRs (eval_a3_tre)."""
    synth_run = (run.run_root / "synth_meta.json").is_file() or run.arm.upper().startswith("A3")
    def checked(root):
        if (root / "synth_meta.json").is_file():
            raise ValueError("Evaluation inputs must be real A1 data, not a synth run")
        summary_path = data._find_tre_summary(root)
        summary = data._read_summary(summary_path) if summary_path else None
        frame = data._read_frame(summary, root)
        if frame in ("native", "r3") and frame != run.frame:
            raise ValueError(f"Real input frame {frame} differs from model evaluation frame {run.frame}")
        return data.model_training_dir(root, run.scan_id)

    if real_runs_dir is not None:
        root = data.normalize_runs_dir(real_runs_dir) / run.scan_id
        mt = checked(root)
        if mt is None:
            raise FileNotFoundError(f"Missing real ModelTraining for {run.scan_id} under {real_runs_dir}")
        return mt
    if run.tre_summary_path and run.tre_summary_path.is_file():
        summary = json.loads(run.tre_summary_path.read_text())
        if summary.get("data_dir"):
            mt = Path(summary["data_dir"])
            if mt.is_dir() and (not synth_run or not mt.is_relative_to(run.run_root)):
                return mt
    if synth_run:
        root = data.ARMS_ROOT / "A1_oracle_dirlab" / "runs" / run.scan_id
        mt = checked(root)
        if mt is None:
            raise FileNotFoundError("A3 inference needs real A1 ModelTraining. Set --real-runs-dir; synthetic training DRRs cannot be used for this evaluation.")
        return mt
    mt = data.model_training_dir(run.run_root, run.scan_id)
    if mt is None:
        raise FileNotFoundError("Missing real ModelTraining")
    return mt


def voxelmap_field(run, phase, *, infer=False, stride=10, device="cuda", real_runs_dir=None):
    path = phase_cache_path(run, phase)
    legacy = run.run_root / "tre_75" / path.name
    for candidate in (path, legacy):
        if candidate.is_file():
            sidecar = candidate.with_suffix(".json")
            if sidecar.is_file():
                metadata = json.loads(sidecar.read_text())
                if any(metadata.get(k) != v for k, v in
                       (("phase", phase), ("reference", REFERENCE), ("frame", run.frame))):
                    raise ValueError(f"Cache provenance mismatch: {sidecar}")
            return data._load_zyx_npy(candidate)
    if not infer:
        raise FileNotFoundError(f"Missing {path.name}; run phase-performance --infer-voxelmap")
    mt = _real_model_training(run, real_runs_dir)
    ckpt = run.run_root / "checkpoints_nofilm" / "best.pt"
    if mt is None or not ckpt.is_file():
        raise FileNotFoundError("VoxelMap inference needs ModelTraining and checkpoints_nofilm/best.pt")
    evaluator = data.require_evaluator()
    dvf, n = evaluator.infer_voxelmap_dvf_phase(
        ckpt, mt, device, stride=stride, phase=data.phase_to_train_idx(phase),
    )
    if dvf.shape != (128, 128, 128, 3) or not np.isfinite(dvf).all():
        raise ValueError("Inference returned an invalid 128³ xyz field")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as tmp:
        temp = Path(tmp.name)
    try:
        np.save(temp, dvf.astype(np.float32))
        temp.replace(path)
        path.with_suffix(".json").write_text(json.dumps({
            "phase": phase, "reference": REFERENCE, "frame": run.frame,
            "checkpoint": str(ckpt.resolve()), "data_dir": str(mt.resolve()),
            "stride": stride, "n_projections": n,
        }, indent=2) + "\n")
    finally:
        temp.unlink(missing_ok=True)
    return dvf


def _synth_inverse(run, phase, src_off):
    """Solve q + u_native(q) = p, matching prepare_a3_dir_case.py's warp.

    Saved channels are xyz on a zyx inference grid. Native resizing uses
    align_corners=True and the preparation script's channel scaling (D,H,W).
    Sample the eight *native* neighbours before interpolation, reproducing
    resize-then-warp even where an inference-grid knot crosses a native cell.
    """
    ev = data.require_evaluator()
    train = data.train_dir(run.run_root, run.scan_id)
    if train is None:
        raise FileNotFoundError("Missing synthesizer train directory")
    idx = data.phase_to_train_idx(phase)
    path = train / f"_synth_dvf_infer_{idx:02d}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path.name}; direct synth requires raw pull fields")
    meta = json.loads((run.run_root / "synth_meta.json").read_text())
    raw = np.load(path, allow_pickle=False)
    if raw.ndim != 4 or raw.shape[0] != 3 or min(raw.shape[1:]) < 2 or not np.isfinite(raw).all():
        raise ValueError(f"Invalid raw synth field {path}: expected finite (3,D,H,W)")
    (nx, ny, nz), spacing = data.CASE_INFO[run.case]
    shape_xyz = np.asarray((nx, nz, ny) if run.r3 else (nx, ny, nz))
    native_zyx = shape_xyz[::-1]
    if list(meta.get("native_shape", [])) != native_zyx.tolist():
        raise ValueError("synth_meta native_shape does not match case/frame")
    src_pack = ev.official_to_pack(src_off, nz)
    src_grid = ev.pack_to_r3(src_pack, ny, nz) if run.r3 else src_pack
    grid_zyx = np.asarray(raw.shape[1:])
    scales = native_zyx / grid_zyx  # intentionally matches synthesis implementation

    def displacement(xyz):
        lower = np.floor(xyz).astype(int)
        weight = xyz - lower
        out = np.zeros_like(xyz)
        for corner in np.ndindex(2, 2, 2):
            corner = np.asarray(corner)
            pos = np.clip(lower + corner, 0, shape_xyz - 1)
            coords = (pos[:, ::-1] * (grid_zyx - 1) / (native_zyx - 1)).T
            sample = np.column_stack([
                map_coordinates(raw[c], coords, order=1, mode="nearest", prefilter=False)
                for c in range(3)
            ]) * scales
            out += sample * np.prod(np.where(corner, weight, 1 - weight), axis=1)[:, None]
        return out

    pred_grid = src_grid.copy()
    grid_spacing = np.asarray(ev.r3_shape_spacing(run.case)[1] if run.r3 else spacing)
    for _ in range(150):
        residual = pred_grid + displacement(pred_grid) - src_grid
        if np.max(np.linalg.norm(residual * grid_spacing, axis=1)) <= 0.01:
            break
        pred_grid -= 0.7 * residual
    residual_mm = np.linalg.norm((pred_grid + displacement(pred_grid) - src_grid) * grid_spacing, axis=1)
    outside = ((pred_grid < 0) | (pred_grid > shape_xyz - 1)).any(axis=1)
    bad = outside | (residual_mm > 0.01) | ~np.isfinite(residual_mm)
    if bad.any():
        raise ValueError(f"Synth inverse invalid for {bad.sum()}/75 landmarks (out of grid or residual >0.01 mm); TRE unavailable")
    pred_pack = ev.r3_to_pack(pred_grid, ny, nz) if run.r3 else pred_grid
    return ev.pack_to_official(pred_pack, nz), pred_pack, outside


def evaluate_phase(run, stage, phase, *, infer=False, stride=10, device="cuda", real_runs_dir=None) -> PhaseResult:
    """Always return a cell, including a reason for missing/invalid data."""
    if stride < 1:
        raise ValueError("Projection stride must be at least 1")
    if stage not in STAGES or phase not in PHASES:
        raise ValueError("Unknown evaluation stage or respiratory phase")
    result = PhaseResult(run, stage, phase, "unavailable")
    try:
        src, truth = _points(run, REFERENCE), _points(run, phase)
        spacing = np.asarray(data.CASE_INFO[run.case][1])
        result.observed_mm = (truth - src) * spacing
        if run.frame not in ("r3", "native"):
            raise ValueError("Unknown frame; provide a summary with native/r3 frame")
        ev = data.require_evaluator()
        nz = data.CASE_INFO[run.case][0][2]
        src_pack, truth_pack = ev.official_to_pack(src, nz), ev.official_to_pack(truth, nz)
        if phase == REFERENCE:
            pred, pred_pack, oob = src.copy(), src_pack.copy(), np.zeros(75, dtype=bool)
        elif stage == "synth":
            pred, pred_pack, oob = _synth_inverse(run, phase, src)
        else:
            dvf = voxelmap_field(run, phase, infer=infer, stride=stride, device=device, real_runs_dir=real_runs_dir)
            pred, pred_pack, oob = data._push_forward(dvf, run.case, src, r3=run.r3, sign=1)
            if oob.any():
                raise ValueError(f"{oob.sum()}/75 landmarks outside VoxelMap grid; TRE unavailable")
        error = (pred - truth) * spacing
        tre = np.linalg.norm(error, axis=1)
        identity = np.linalg.norm(result.observed_mm, axis=1)
        result.landmarks = data.PerLandmarkResult(
            run.case, stage, "75", f"{REFERENCE}_{phase}", run.frame,
            truth, pred, src, truth_pack, pred_pack, src_pack, error, tre, identity,
            oob, data.stats(tre), data.stats(identity),
        )
        result.predicted_mm = (pred - src) * spacing
        result.status = "reference" if phase == REFERENCE else "ok"
        result.reason = "Fixed reference, excluded from phase ranking" if phase == REFERENCE else ""
    except FileNotFoundError as exc:
        result.reason = str(exc)
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
        result.status, result.reason = "error", str(exc)
    return result


def evaluate_cohort(runs, stage, *, cancelled=lambda: False, **kwargs):
    for run in runs:
        for phase in PHASES:
            if cancelled():
                return
            yield evaluate_phase(run, stage, phase, **kwargs)


def empty_landmarks(result):
    """Display-only empty overlays; never exported or counted as a TRE score."""
    xyz, values = np.empty((0, 3)), np.empty(0)
    stats = dict.fromkeys(("mean", "std", "p50", "p95", "max"), float("nan"))
    stats["n"] = 0
    return data.PerLandmarkResult(
        result.run.case, result.stage, "75", f"T50_{result.phase}", result.run.frame,
        xyz, xyz, xyz, xyz, xyz, xyz, xyz, values, values, np.zeros(0, bool), stats, stats,
    )


def synth_image_check(run, phase):
    """Real vs synthesized native CT in HU; independent of landmark coverage."""
    import SimpleITK as sitk
    from .warp_dvf import upsample_to_pack
    truth = data.load_pack_volume(run, phase)
    train = data.train_dir(run.run_root, run.scan_id)
    if train is None or not (run.run_root / "synth_meta.json").is_file():
        raise FileNotFoundError("Image check requires a prepared synth run with synth_meta.json")
    path = train / f"CT_{data.phase_to_train_idx(phase):02d}.mha"
    pred = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
    nx, ny, nz = data.CASE_INFO[run.case][0]
    expected = (ny, nz, nx) if run.r3 else (nz, ny, nx)
    if run.frame not in ("native", "r3") or pred.shape != expected or not np.isfinite(pred).all():
        raise ValueError("Synth CT shape/frame invalid")
    pred = upsample_to_pack(pred, truth.data.shape, frame=run.frame)
    if not np.isfinite(truth.data).all():
        raise ValueError("Real CT contains non-finite HU values")
    diff = pred - truth.data
    return truth, pred, diff, float(np.mean(np.abs(diff)))


def export_results(path: Path, results, *, expected_cells=None):
    records = [r.to_dict() for r in results]
    expected_cells = len(records) if expected_cells is None else expected_cells
    path.write_text(json.dumps({
        "schema_version": 1, "direction": "T50 to target phase", "landmark_set": "75",
        "trajectory_axes": "official xyz (approximately LR/AP/SI), mm relative to T50",
        "evaluated_cells": len(records), "expected_cells": expected_cells,
        "complete": len(records) == expected_cells, "results": records,
    }, indent=2, allow_nan=False) + "\n")
