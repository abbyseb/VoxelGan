"""Data adapter for TRE Viewer.

Thin wrapper over ``scripts/dirlab_tre.py`` and ``scripts/eval_a1_tre.py``.
Does not reimplement the official↔pack↔R3↔sub coordinate chain.

Phase 0 surface:
  - discover_runs()
  - case_info / landmarks in pack or r3 frames
  - load fields (identity / elastix / voxelmap cache)
  - per_landmark() with npz cache
  - verify_against_summary()
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import sys
import warnings
from zipfile import BadZipFile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np

DIR_EXP = Path(__file__).resolve().parents[2]
SCRIPTS = DIR_EXP / "scripts"
ARMS_ROOT = DIR_EXP / "arms"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

os.environ.setdefault("DIRLAB_ROOT", str(DIR_EXP / "data" / "dirlab_packs"))

from dirlab_tre import (  # noqa: E402
    CASE_INFO,
    landmarks_75,
    landmarks_300,
    sample_dvf,
    stats,
    tre_mm,
)
from .evaluation import require_evaluator


FrameName = Literal["pack", "r3", "official", "sub"]
LandmarkSet = Literal["75", "300"]
PhasePair = Literal["T00_T50", "T50_T00"]

_CASE_RE = re.compile(r"(?:DIR_)?C(\d{1,2})$", re.IGNORECASE)
_SCAN_RE = re.compile(r"DIR_C(\d{2})$", re.IGNORECASE)


@dataclass(frozen=True)
class RunRef:
    """One discoverable TRE run under ``arms/*/runs/...``."""

    arm: str
    case: int
    scan_id: str
    run_root: Path
    frame: str  # "r3" | "native" | "unknown"
    fields_available: tuple[str, ...] = ()
    tre_summary_path: Path | None = None
    has_checkpoint: bool = False
    geometry_candidates: tuple[str, ...] = ()

    @property
    def r3(self) -> bool:
        return self.frame == "r3"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["run_root"] = str(self.run_root)
        d["tre_summary_path"] = (
            str(self.tre_summary_path) if self.tre_summary_path else None
        )
        return d


@dataclass
class PerLandmarkResult:
    """Per-landmark TRE for one field / pair / set."""

    case: int
    field: str
    which: LandmarkSet
    pair: PhasePair
    frame: str  # display/compute frame tag from run ("r3"|"native")
    truth_official: np.ndarray  # (N,3) destination landmarks, official
    pred_official: np.ndarray  # (N,3)
    src_official: np.ndarray  # (N,3) source landmarks, official
    truth_pack: np.ndarray
    pred_pack: np.ndarray
    src_pack: np.ndarray
    err_vec_mm: np.ndarray  # pred - truth in official mm (dx,dy,dz)
    tre_mm: np.ndarray  # (N,)
    identity_mm: np.ndarray  # (N,)
    oob_mask: np.ndarray  # (N,) bool — source sample clamped
    registered_stats: dict[str, float] = field(default_factory=dict)
    identity_stats: dict[str, float] = field(default_factory=dict)

    @property
    def improvement_mm(self) -> float:
        return float(self.identity_stats["mean"] - self.registered_stats["mean"])


# ---------------------------------------------------------------------------
# Paths / discovery
# ---------------------------------------------------------------------------
def arm_dir(arm: str) -> Path:
    """Map short arm id (A1) or folder name to arms/<folder>."""
    arm = arm.strip()
    if (ARMS_ROOT / arm).is_dir():
        return ARMS_ROOT / arm
    aliases = {
        "A0": "A0_identity",
        "A1": "A1_oracle_dirlab",
        "A2": "A2_generic_spare",
        "A3": "A3_synth_conditioned",
        "A4": "A4_mismatched_conditioning",
        "A5": "A5_synth_cbct_finetune",
    }
    key = arm.upper()
    if key in aliases:
        return ARMS_ROOT / aliases[key]
    raise FileNotFoundError(f"Unknown arm {arm!r} under {ARMS_ROOT}")


def case_info(case: int) -> dict[str, Any]:
    (nx, ny, nz), (dx, dy, dz) = CASE_INFO[case]
    r3_shape, r3_sp = require_evaluator().r3_shape_spacing(case)
    return {
        "case": case,
        "pack_shape_xyz": (nx, ny, nz),
        "pack_spacing_mm": (dx, dy, dz),
        "r3_shape_xyz": r3_shape,
        "r3_spacing_mm": r3_sp,
        "sub_shape": (128, 128, 128),
        # sub-voxel → mm on pack axes (native) and on R3 axes
        "sub_to_mm_pack": (nx / 128.0 * dx, ny / 128.0 * dy, nz / 128.0 * dz),
        "sub_to_mm_r3": (
            r3_shape[0] / 128.0 * r3_sp[0],
            r3_shape[1] / 128.0 * r3_sp[1],
            r3_shape[2] / 128.0 * r3_sp[2],
        ),
    }


def _parse_case_from_name(name: str) -> int | None:
    m = _SCAN_RE.fullmatch(name) or _CASE_RE.fullmatch(name)
    if not m:
        return None
    case = int(m.group(1))
    return case if case in CASE_INFO else None


def _read_summary(path: Path) -> dict[str, Any] | None:
    try:
        summary = json.loads(path.read_text())
        if not isinstance(summary, dict):
            raise ValueError("expected a JSON object")
        return summary
    except (OSError, ValueError) as exc:
        warnings.warn(f"Cannot read TRE summary {path}: {exc}", stacklevel=2)
        return None


def _find_tre_summary(run_root: Path) -> Path | None:
    for rel in (
        "tre/tre_summary.json",
        "tre_75/tre_75_summary.json",
        "tre_75/tre_summary.json",
        "results/tre_summary.json",
    ):
        p = run_root / rel
        if p.is_file():
            return p
    # last resort: any *tre*summary*.json under run
    hits = sorted(run_root.glob("**/tre*summary*.json"))
    return hits[0] if hits else None


def _read_frame(summary: dict[str, Any] | None, run_root: Path) -> str:
    if summary and summary.get("frame") in ("r3", "native"):
        return str(summary["frame"])
    # Heuristic: R3 CT is (nx, nz, ny) e.g. 256x94x256 for case 1
    train = _train_dir(run_root, run_root.name)
    ct = (train or run_root / "train") / "CT_01.mha"
    if not ct.is_file():
        # some layouts nest scan_id twice
        for cand in run_root.glob("DIR_C*/train/CT_01.mha"):
            ct = cand
            break
    if ct.is_file():
        try:
            import SimpleITK as sitk

            reader = sitk.ImageFileReader()
            reader.SetFileName(str(ct))
            reader.ReadImageInformation()
            size = tuple(int(x) for x in reader.GetSize())  # ITK (x,y,z)
            # pack native: (nx,ny,nz) with ny≈nx; R3: (nx,nz,ny) with middle = nz small
            if len(size) == 3 and size[1] < size[0] and size[1] < size[2]:
                return "r3"
            return "native"
        except Exception:
            pass
    return "unknown"


def model_training_dir(run_root: Path, scan_id: str) -> Path | None:
    return _model_training_dir(run_root, scan_id)


def train_dir(run_root: Path, scan_id: str) -> Path | None:
    return _train_dir(run_root, scan_id)


def _model_training_dir(run_root: Path, scan_id: str) -> Path | None:
    for cand in (
        run_root / "ModelTraining" / "train" / scan_id,
        run_root / "ModelTraining" / scan_id,
    ):
        if cand.is_dir():
            return cand
    return None


def _train_dir(run_root: Path, scan_id: str) -> Path | None:
    for cand in (
        run_root / scan_id / "train",
        run_root / "train",
    ):
        if cand.is_dir():
            return cand
    return None


def _fields_for_run(run_root: Path, scan_id: str) -> tuple[str, ...]:
    found: list[str] = ["identity"]
    train = _train_dir(run_root, scan_id)
    mt = _model_training_dir(run_root, scan_id)
    if train and (train / "DVF_sub_01.mha").is_file():
        found.append("elastix_mha")
    if mt and (mt / "DVFs" / "DVF_01_mha.npy").is_file():
        found.append("elastix_npy")
    for p in (
        run_root / "tre" / "voxelmap_dvf_phase01_mean.npy",
        run_root / "tre_75" / "voxelmap_dvf_phase01_mean.npy",
    ):
        if p.is_file():
            found.append("voxelmap")
            break
    ckpt = run_root / "checkpoints_nofilm" / "best.pt"
    if "voxelmap" not in found and ckpt.is_file():
        found.append("voxelmap_ckpt")  # needs inference later
    return tuple(found)


def _geometry_candidates(run_root: Path, arm: str, scan_id: str) -> tuple[str, ...]:
    cands: list[Path] = []
    staged = ARMS_ROOT / arm / "data" / "staged" / scan_id / "Proj" / "Geometry.xml"
    cands.extend(
        [
            run_root / "Geometry.xml",
            ARMS_ROOT / arm / "Geometry.xml",
            staged,
            Path(
                "/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/"
                "modules/drr_generation/Geometry_SPARE.xml"
            ),
        ]
    )
    return tuple(str(p) for p in cands if p.is_file())


def _make_run_ref(
    run_root: Path,
    arm_name: str,
    *,
    require_summary: bool = False,
) -> RunRef | None:
    """Build a RunRef from a single case folder (``DIR_Cxx`` / ``Cxx``)."""
    case = _parse_case_from_name(run_root.name)
    if case is None:
        return None
    scan_id = f"DIR_C{case:02d}"
    summary_path = _find_tre_summary(run_root)
    if require_summary and summary_path is None:
        return None
    summary = _read_summary(summary_path) if summary_path else None
    if summary is None:
        summary_path = None
        if require_summary:
            return None
    if summary and "case" in summary:
        if str(summary["case"]) != str(case):
            warnings.warn(f"Skipping {run_root}: summary case does not match folder")
            return None
    frame = _read_frame(summary, run_root)
    fields = _fields_for_run(run_root, scan_id)
    ckpt = (run_root / "checkpoints_nofilm" / "best.pt").is_file()
    return RunRef(
        arm=arm_name,
        case=case,
        scan_id=scan_id,
        run_root=run_root,
        frame=frame,
        fields_available=fields,
        tre_summary_path=summary_path,
        has_checkpoint=ckpt,
        geometry_candidates=_geometry_candidates(run_root, arm_name, scan_id),
    )


def normalize_runs_dir(path: Path | str) -> Path:
    """Accept arm root, ``…/runs``, or a single ``DIR_Cxx`` folder → runs dir."""
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise NotADirectoryError(path)
    if (path / "runs").is_dir():
        return (path / "runs").resolve()
    if _parse_case_from_name(path.name) is not None:
        return path.parent.resolve()
    return path


def arm_label_from_runs_dir(runs_dir: Path) -> str:
    runs_dir = Path(runs_dir)
    if runs_dir.name == "runs":
        return runs_dir.parent.name
    return runs_dir.name


def discover_runs_in_folder(
    path: Path | str,
    *,
    require_summary: bool = False,
) -> list[RunRef]:
    """Discover ``DIR_Cxx`` runs under an arbitrary folder (viewer browse)."""
    runs_root = normalize_runs_dir(path)
    arm_name = arm_label_from_runs_dir(runs_root)
    out: list[RunRef] = []
    for run_root in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        ref = _make_run_ref(run_root, arm_name, require_summary=require_summary)
        if ref is not None:
            out.append(ref)
    return out


def discover_runs(
    arms: Iterable[str] | None = None,
    *,
    require_summary: bool = False,
) -> list[RunRef]:
    """Glob ``arms/*/runs/*`` and return RunRefs (tolerant of empty arms)."""
    if arms is None:
        if not ARMS_ROOT.is_dir():
            return []
        arm_folders = sorted(
            p.name for p in ARMS_ROOT.iterdir() if p.is_dir() and not p.name.startswith(".")
        )
    else:
        arm_folders = [arm_dir(a).name for a in arms]

    out: list[RunRef] = []
    for arm_name in arm_folders:
        runs_root = ARMS_ROOT / arm_name / "runs"
        if not runs_root.is_dir():
            continue
        out.extend(
            discover_runs_in_folder(runs_root, require_summary=require_summary)
        )
    return out


def load_tre_summary(run: RunRef) -> dict[str, Any] | None:
    if run.tre_summary_path is None or not run.tre_summary_path.is_file():
        return None
    return _read_summary(run.tre_summary_path)


# ---------------------------------------------------------------------------
# Volumes (pack-native GTVol — Phase 1 display frame)
# ---------------------------------------------------------------------------
# DIR phase index: T00→01 … T50→06 … T90→10
PHASE_TO_IDX = {f"T{10 * i:02d}": i + 1 for i in range(10)}


def phase_to_train_idx(phase: str) -> int:
    if phase not in PHASE_TO_IDX:
        raise ValueError(f"Unknown phase {phase!r}")
    return PHASE_TO_IDX[phase]


def pair_phases(pair: PhasePair) -> tuple[str, str]:
    if pair == "T00_T50":
        return "T00", "T50"
    if pair == "T50_T00":
        return "T50", "T00"
    raise ValueError(pair)


@dataclass(frozen=True)
class PackVolume:
    """Pack-frame CT as napari-ready (z,y,x) with mm scale."""

    data: np.ndarray  # (nz, ny, nx) float32
    spacing_zyx_mm: tuple[float, float, float]  # (dz, dy, dx)
    path: Path
    phase: str

    @property
    def scale(self) -> tuple[float, float, float]:
        return self.spacing_zyx_mm


def load_pack_volume(run: RunRef, phase: str) -> PackVolume:
    """Load ``GTVol_XX.mha`` (pack native).

    A2/A3 runs often only stage DRRs/DVFs; fall back to the matching A1 pack
    ``GTVol`` (same case) so the viewer can still display anatomy + TRE.
    """
    idx = phase_to_train_idx(phase)
    candidates: list[Path] = []
    train = _train_dir(run.run_root, run.scan_id)
    if train is not None:
        candidates.append(train / f"GTVol_{idx:02d}.mha")
    # Same-case A1 oracle pack (canonical GTVol for DIR EXPERIMENTS)
    a1 = ARMS_ROOT / "A1_oracle_dirlab" / "runs" / run.scan_id
    candidates.extend(
        [
            a1 / run.scan_id / "train" / f"GTVol_{idx:02d}.mha",
            a1 / "train" / f"GTVol_{idx:02d}.mha",
        ]
    )

    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            f"No GTVol for {run.scan_id} phase={phase}; tried: "
            + ", ".join(str(p) for p in candidates)
        )
    import SimpleITK as sitk

    img = sitk.ReadImage(str(path))
    # ITK size (x,y,z); array is (z,y,x)
    data = sitk.GetArrayFromImage(img).astype(np.float32)
    sp = img.GetSpacing()  # (dx, dy, dz)
    spacing_zyx = (float(sp[2]), float(sp[1]), float(sp[0]))
    # Sanity vs CASE_INFO
    (nx, ny, nz), (dx, dy, dz) = CASE_INFO[run.case]
    if data.shape != (nz, ny, nx):
        raise ValueError(
            f"{path}: array shape {data.shape} != expected (nz,ny,nx)={(nz, ny, nx)}"
        )
    # Prefer CASE_INFO spacing when shapes match pack
    spacing_zyx = (float(dz), float(dy), float(dx))
    return PackVolume(data=data, spacing_zyx_mm=spacing_zyx, path=path, phase=phase)


def xyz_to_zyx(pts_xyz: np.ndarray) -> np.ndarray:
    """(N,3) x,y,z voxel indices → napari (z,y,x)."""
    p = np.asarray(pts_xyz, dtype=np.float64)
    return np.stack([p[:, 2], p[:, 1], p[:, 0]], axis=1)


def zyx_to_xyz(pts_zyx: np.ndarray) -> np.ndarray:
    p = np.asarray(pts_zyx, dtype=np.float64)
    return np.stack([p[:, 2], p[:, 1], p[:, 0]], axis=1)


def resolve_run(
    arm: str | None,
    case: int,
    *,
    runs_dir: Path | str | None = None,
) -> RunRef:
    if runs_dir is not None:
        runs = discover_runs_in_folder(runs_dir)
    else:
        runs = discover_runs([arm] if arm else None)
    hits = [r for r in runs if r.case == case]
    if not hits:
        where = f"runs_dir={runs_dir!r}" if runs_dir else f"arm={arm!r}"
        raise FileNotFoundError(f"No run for {where} case={case}")
    if len(hits) > 1:
        raise ValueError(
            f"Multiple runs match case {case}; select --arm or --runs-dir: "
            + ", ".join(str(r.run_root) for r in hits)
        )
    return hits[0]


def default_field(run: RunRef) -> str:
    for pref in ("voxelmap", "elastix_mha", "elastix_npy", "identity"):
        if pref in run.fields_available:
            return pref
    return run.fields_available[0] if run.fields_available else "identity"


# ---------------------------------------------------------------------------
# Landmarks
# ---------------------------------------------------------------------------
def _load_pair(case: int, which: LandmarkSet, pair: PhasePair):
    if which not in ("75", "300"):
        raise ValueError(f"Unknown landmark set {which!r}; choose 75 or 300")
    load = landmarks_75 if which == "75" else landmarks_300
    if pair == "T00_T50":
        return load(case, "T00"), load(case, "T50")
    if pair == "T50_T00":
        return load(case, "T50"), load(case, "T00")
    raise ValueError(pair)


def landmarks(
    case: int,
    phase: str,
    which: LandmarkSet = "75",
    frame: FrameName = "pack",
) -> np.ndarray:
    """Return (N,3) landmarks in the requested frame (voxel indices)."""
    load = landmarks_75 if which == "75" else landmarks_300
    xyz = load(case, phase)  # official
    (nx, ny, nz), _ = CASE_INFO[case]
    if frame == "official":
        return xyz
    pack = require_evaluator().official_to_pack(xyz, nz)
    if frame == "pack":
        return pack
    if frame == "r3":
        return require_evaluator().pack_to_r3(pack, ny, nz)
    if frame == "sub":
        # sub of pack (native) — for r3-sub use frame='r3' then pack_to_sub yourself
        return require_evaluator().pack_to_sub(pack, (nx, ny, nz))
    raise ValueError(frame)


# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------
def load_field_zyx3(run: RunRef, field: str, *, infer: bool = True) -> np.ndarray:
    """Load a dense DVF as (z,y,x,3) in **sub-voxels** on the 128³ grid.

    ``identity`` → zeros.
    ``elastix_mha`` / ``elastix_npy`` / ``voxelmap`` from disk.
    ``voxelmap`` / ``voxelmap_ckpt``: if cache missing and ``infer``, run best.pt once.
    """
    if field == "identity":
        return np.zeros((128, 128, 128, 3), dtype=np.float64)

    scan_id = run.scan_id
    train = _train_dir(run.run_root, scan_id)
    mt = _model_training_dir(run.run_root, scan_id)

    if field == "elastix_mha":
        if train is None:
            raise FileNotFoundError(f"no train/ under {run.run_root}")
        return require_evaluator().load_dvf_zyx3(train / "DVF_sub_01.mha")

    if field == "elastix_npy":
        if mt is None:
            raise FileNotFoundError(f"no ModelTraining under {run.run_root}")
        return require_evaluator().load_dvf_zyx3(mt / "DVFs" / "DVF_01_mha.npy")

    if field in ("voxelmap", "voxelmap_ckpt"):
        return ensure_voxelmap_dvf(run, infer=infer)

    # ad-hoc path
    path = Path(field)
    if path.is_file():
        if path.suffix == ".mha":
            return require_evaluator().load_dvf_zyx3(path)
        return _load_zyx_npy(path) if _npy_is_zyx_cache(path) else require_evaluator().load_dvf_zyx3(path)

    raise KeyError(
        f"Unknown field {field!r}. Available: {run.fields_available}"
    )


def voxelmap_cache_path(run: RunRef) -> Path:
    return run.run_root / "tre" / "voxelmap_dvf_phase01_mean.npy"


def ensure_voxelmap_dvf(
    run: RunRef,
    *,
    infer: bool = True,
    stride: int = 10,
) -> np.ndarray:
    """Return VoxelMap DVF (z,y,x,3); infer from ``best.pt`` once if needed."""
    for p in (
        voxelmap_cache_path(run),
        run.run_root / "tre_75" / "voxelmap_dvf_phase01_mean.npy",
    ):
        if p.is_file():
            return _load_zyx_npy(p)
    if not infer:
        raise FileNotFoundError(
            f"no voxelmap DVF cache under {run.run_root}/tre"
        )

    ckpt = run.run_root / "checkpoints_nofilm" / "best.pt"
    mt = _model_training_dir(run.run_root, run.scan_id)
    if not ckpt.is_file():
        raise FileNotFoundError(f"missing checkpoint {ckpt}")
    if mt is None:
        raise FileNotFoundError(f"missing ModelTraining under {run.run_root}")

    out = voxelmap_cache_path(run)
    out.parent.mkdir(parents=True, exist_ok=True)
    dvf = _infer_voxelmap_dvf(ckpt, mt, stride=stride)
    np.save(out, dvf.astype(np.float32))
    legacy = run.run_root / "tre_75" / "voxelmap_dvf_phase01_mean.npy"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    np.save(legacy, dvf.astype(np.float32))
    return dvf


def _infer_voxelmap_dvf(ckpt: Path, mt: Path, *, stride: int = 10) -> np.ndarray:
    """Run ``infer_voxelmap_dvf_phase01``; fall back to LEARN-GUI venv subprocess."""
    try:
        import torch  # noqa: F401

        from eval_a1_tre import infer_voxelmap_dvf_phase01

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dvf, n = infer_voxelmap_dvf_phase01(ckpt, mt, device, stride=stride)
        print(f"[tre_viewer] inferred VoxelMap DVF over {n} projs → cache")
        return np.asarray(dvf, dtype=np.float64)
    except ImportError as exc:
        import_error = exc

    learn_py = Path(os.environ.get(
        "TRE_VIEWER_INFERENCE_PYTHON",
        "/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python",
    )).expanduser()
    if not learn_py.is_file():
        raise ImportError(
            f"VoxelMap inference dependency unavailable: {import_error}. "
            "Install torch and set VOXELMAP_CLINICAL_ROOT, set "
            "TRE_VIEWER_INFERENCE_PYTHON to a prepared Python environment, "
            "or run eval_a1_tre.py there to create the DVF cache."
        ) from import_error
    import subprocess
    import tempfile

    helper = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    try:
        helper.write(
            "import sys\n"
            "from pathlib import Path\n"
            "import numpy as np\n"
            "import torch\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "from eval_a1_tre import infer_voxelmap_dvf_phase01\n"
            "ckpt, mt, out = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])\n"
            "stride = int(sys.argv[5])\n"
            "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
            "dvf, n = infer_voxelmap_dvf_phase01(ckpt, mt, device, stride=stride)\n"
            "np.save(out, dvf.astype(np.float32))\n"
            "print(f'WROTE {out} n={n}')\n"
        )
        helper.close()
        fd, tmp_name = tempfile.mkstemp(suffix=".npy")
        os.close(fd)
        tmp_out = Path(tmp_name)
        cmd = [
            str(learn_py),
            helper.name,
            str(SCRIPTS),
            str(ckpt),
            str(mt),
            str(tmp_out),
            str(stride),
        ]
        print(f"[tre_viewer] inferring via LEARN python: {learn_py}")
        subprocess.run(cmd, check=True)
        return _load_zyx_npy(tmp_out)
    finally:
        Path(helper.name).unlink(missing_ok=True)
        if "tmp_out" in locals() and tmp_out.is_file():
            tmp_out.unlink(missing_ok=True)


def _load_hwd_volume(path: Path) -> np.ndarray:
    """Validate once; repeatedly squeezing a non-singleton axis never terminates."""
    a = np.load(path, allow_pickle=False).squeeze()
    if a.shape != (128, 128, 128):
        raise ValueError(f"Expected a 128³ HWD volume, got {a.shape} from {path}")
    if not np.isfinite(a).all():
        raise ValueError(f"Non-finite values in volume {path}")
    return np.transpose(a, (2, 0, 1))


def load_sub_volume_zyx(run: RunRef, phase: str) -> np.ndarray:
    """Load 128³ sub_CT as (z,y,x) float32 (from npy HWD or mha)."""
    idx = phase_to_train_idx(phase)
    mt = _model_training_dir(run.run_root, run.scan_id)
    if mt is not None:
        npy = mt / "SourceVolumes" / f"sub_CT_{idx:02d}_mha.npy"
        if npy.is_file():
            return _load_hwd_volume(npy).astype(np.float32)
    train = _train_dir(run.run_root, run.scan_id)
    if train is None:
        raise FileNotFoundError(f"no sub_CT for {run.scan_id} phase {phase}")
    path = train / f"sub_CT_{idx:02d}.mha"
    if not path.is_file():
        raise FileNotFoundError(path)
    import SimpleITK as sitk

    # sitk array is already zyx for these volumes
    a = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
    if a.shape != (128, 128, 128) or not np.isfinite(a).all():
        raise ValueError(f"Expected a finite 128³ volume, got {a.shape} from {path}")
    return a


def load_lung_mask_zyx(run: RunRef) -> np.ndarray | None:
    mt = _model_training_dir(run.run_root, run.scan_id)
    if mt is None:
        return None
    p = mt / "Masks" / "Mask_Lung_mha.npy"
    if not p.is_file():
        return None
    return _load_hwd_volume(p).astype(bool)


def sub_to_mm_xyz(run: RunRef) -> tuple[float, float, float]:
    info = case_info(run.case)
    if run.frame == "r3":
        return tuple(info["sub_to_mm_r3"])  # type: ignore[return-value]
    return tuple(info["sub_to_mm_pack"])  # type: ignore[return-value]


def pack_shape_zyx(case: int) -> tuple[int, int, int]:
    (nx, ny, nz), _ = CASE_INFO[case]
    return (nz, ny, nx)


def field_warp_bundle(
    run: RunRef,
    field: str,
    pair: PhasePair = "T00_T50",
) -> dict[str, Any]:
    """Source/target/warped/diffs + DVF mm maps on the sub grid (+ pack upsamples)."""
    if pair == "T50_T00" and field != "identity":
        raise ValueError(
            "Reverse image warping requires an inverse DVF. Select T00_T50 for "
            "image overlays; reverse landmark TRE is still available."
        )
    if run.frame not in ("native", "r3"):
        raise ValueError("Cannot display DVF overlays without a known native/r3 frame")
    from .warp_dvf import (
        dvf_components_mm,
        label_components,
        lung_mae,
        upsample_to_pack,
        warp_pull,
    )

    src_ph, dst_ph = pair_phases(pair)
    src = load_sub_volume_zyx(run, src_ph)
    tgt = load_sub_volume_zyx(run, dst_ph)
    mask = load_lung_mask_zyx(run)
    dvf = load_field_zyx3(run, field, infer=True)
    # Stored field is defined on fixed=T50 and pulls the moving=T00 image.
    warped = warp_pull(src, dvf, sign=1.0)
    diff = (tgt - warped).astype(np.float32)
    ident_diff = (tgt - src).astype(np.float32)
    sx_yz = sub_to_mm_xyz(run)
    comps = dvf_components_mm(dvf, sx_yz)
    labels = label_components(run.frame)
    # Remap friendly SI/AP for UI
    ui_comps = {
        "mag": comps["mag"],
        "LR": comps["LR"],
        "SI": comps[labels["SI"]],
        "AP": comps[labels["AP"]],
    }
    if run.r3:
        ui_comps["SI"] = -ui_comps["SI"]
        ui_comps["AP"] = -ui_comps["AP"]
    pshape = pack_shape_zyx(run.case)
    return {
        "src": src,
        "tgt": tgt,
        "warped": warped,
        "diff": diff,
        "ident_diff": ident_diff,
        "dvf": dvf,
        "comps": ui_comps,
        "mask": mask,
        "mae_region": "lung" if mask is not None else "whole volume (no lung mask)",
        "mae_warped_lung": lung_mae(warped, tgt, mask),
        "mae_ident_lung": lung_mae(src, tgt, mask),
        "pack": {
            "warped": upsample_to_pack(warped, pshape, frame=run.frame),
            "diff": upsample_to_pack(diff, pshape, frame=run.frame),
            "ident_diff": upsample_to_pack(ident_diff, pshape, frame=run.frame),
            "mag": upsample_to_pack(ui_comps["mag"], pshape, frame=run.frame),
            "SI": upsample_to_pack(ui_comps["SI"], pshape, frame=run.frame),
            "AP": upsample_to_pack(ui_comps["AP"], pshape, frame=run.frame),
            "LR": upsample_to_pack(ui_comps["LR"], pshape, frame=run.frame),
            "dvf": None,  # filled lazily if arrows need pack
        },
        "sub_to_mm_xyz": sx_yz,
        "src_phase": src_ph,
        "dst_phase": dst_ph,
    }


def _npy_is_zyx_cache(path: Path) -> bool:
    name = path.name.lower()
    return "voxelmap_dvf" in name or path.parent.name in ("tre", "tre_75")


def _load_zyx_npy(path: Path) -> np.ndarray:
    a = np.load(path, allow_pickle=False).astype(np.float64)
    if a.shape != (128, 128, 128, 3):
        raise ValueError(f"Expected (128,128,128,3) zyx DVF, got {a.shape} from {path}")
    if not np.isfinite(a).all():
        raise ValueError(f"Non-finite values in DVF {path}")
    return a


# ---------------------------------------------------------------------------
# Per-landmark TRE (mirrors eval_a1_tre.tre_t00_t50 / tre_t50_t00)
# ---------------------------------------------------------------------------
def _push_forward(
    dvf_zyx3: np.ndarray,
    case: int,
    src_official: np.ndarray,
    *,
    r3: bool,
    sign: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pred_official, pred_pack, oob_mask).

    sign=-1 for T00→T50 (ITK pull-back inverse on landmarks);
    sign=+1 for T50→T00.
    """
    (nx, ny, nz), _ = CASE_INFO[case]
    src_pack = require_evaluator().official_to_pack(src_official, nz)
    if r3:
        shape, _ = require_evaluator().r3_shape_spacing(case)
        src_g = require_evaluator().pack_to_r3(src_pack, ny, nz)
        src_sub = require_evaluator().pack_to_sub(src_g, shape)
        disp = sample_dvf(dvf_zyx3, src_sub)
        pred_g = src_g + sign * require_evaluator().sub_disp_to_pack(disp, shape)
        pred_pack = require_evaluator().r3_to_pack(pred_g, ny, nz)
        # oob in the grid where we sample (r3 continuous → sub)
    else:
        shape = (nx, ny, nz)
        src_sub = require_evaluator().pack_to_sub(src_pack, shape)
        disp = sample_dvf(dvf_zyx3, src_sub)
        pred_pack = src_pack + sign * require_evaluator().sub_disp_to_pack(disp, shape)
    # Report clamping on the actual sampled grid, including its upper edge.
    hi = np.asarray(dvf_zyx3.shape[:3][::-1]) - 1
    oob = ((src_sub < 0) | (src_sub > hi)).any(axis=1)
    pred_official = require_evaluator().pack_to_official(pred_pack, nz)
    return pred_official, pred_pack, oob


def per_landmark_cache_path(run: RunRef, field: str, which: str, pair: str) -> Path:
    """External field paths must never become directories inside the cache name."""
    key = field
    if field not in ("identity", "elastix_mha", "elastix_npy", "voxelmap", "voxelmap_ckpt"):
        key = "custom_" + hashlib.sha256(str(Path(field).resolve()).encode()).hexdigest()[:16]
    return run.run_root / "tre" / f"per_landmark_{which}_{pair}_{key}.npz"


def _cache_signature(run: RunRef, field: str, src: np.ndarray, dst: np.ndarray) -> str:
    paths = [Path(require_evaluator().__file__)]
    train = _train_dir(run.run_root, run.scan_id)
    mt = _model_training_dir(run.run_root, run.scan_id)
    if field == "elastix_mha" and train:
        paths.append(train / "DVF_sub_01.mha")
    elif field == "elastix_npy" and mt:
        paths.append(mt / "DVFs" / "DVF_01_mha.npy")
    elif field in ("voxelmap", "voxelmap_ckpt"):
        paths.extend([
            voxelmap_cache_path(run),
            run.run_root / "tre_75" / "voxelmap_dvf_phase01_mean.npy",
        ])
    elif field != "identity":
        paths.append(Path(field))
    metadata = []
    for path in paths:
        if path.is_file():
            st = path.stat()
            metadata.append((str(path.resolve()), st.st_size, st.st_mtime_ns))
        else:
            metadata.append((str(path.resolve()), None, None))
    digest = hashlib.sha256(json.dumps([2, run.case, run.frame, field, metadata]).encode())
    digest.update(np.asarray(src, dtype=np.float64).tobytes())
    digest.update(np.asarray(dst, dtype=np.float64).tobytes())
    return digest.hexdigest()


def per_landmark(
    run: RunRef,
    field: str = "elastix_mha",
    *,
    which: LandmarkSet = "75",
    pair: PhasePair = "T00_T50",
    use_cache: bool = True,
    write_cache: bool = True,
) -> PerLandmarkResult:
    """Compute (or load) per-landmark TRE for a run/field."""
    src_off, dst_off = _load_pair(run.case, which, pair)
    if (src_off.shape != dst_off.shape or src_off.shape != (int(which), 3)
            or not np.isfinite(src_off).all() or not np.isfinite(dst_off).all()):
        raise ValueError(f"Expected two finite ({which}, 3) landmark sets for C{run.case:02d}")
    if field != "identity" and run.frame not in ("native", "r3"):
        raise ValueError("Unknown DVF frame: provide a tre_summary.json with frame 'native' or 'r3'.")
    cache_path = per_landmark_cache_path(run, field, which, pair)
    signature = _cache_signature(run, field, src_off, dst_off)
    if use_cache and cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                if str(cached.get("signature", "")) == signature:
                    return _load_per_landmark_npz(cache_path)
        except (OSError, ValueError, KeyError, EOFError, BadZipFile):
            warnings.warn(f"Rebuilding unreadable landmark cache {cache_path}", stacklevel=2)

    (nx, ny, nz), spacing = CASE_INFO[run.case]
    src_pack = require_evaluator().official_to_pack(src_off, nz)
    dst_pack = require_evaluator().official_to_pack(dst_off, nz)

    if field == "identity":
        pred_off = src_off.copy()
        pred_pack = src_pack.copy()
        oob = np.zeros(len(src_off), dtype=bool)
    else:
        dvf = load_field_zyx3(run, field)
        sign = -1 if pair == "T00_T50" else +1
        r3 = run.frame == "r3"
        pred_off, pred_pack, oob = _push_forward(
            dvf, run.case, src_off, r3=r3, sign=sign
        )

    err = (pred_off - dst_off) * np.asarray(spacing)
    tre = np.linalg.norm(err, axis=-1)
    ident = tre_mm(src_off, dst_off, spacing)

    result = PerLandmarkResult(
        case=run.case,
        field=field,
        which=which,
        pair=pair,
        frame=run.frame,
        truth_official=dst_off,
        pred_official=pred_off,
        src_official=src_off,
        truth_pack=dst_pack,
        pred_pack=pred_pack,
        src_pack=src_pack,
        err_vec_mm=err,
        tre_mm=tre,
        identity_mm=ident,
        oob_mask=oob,
        registered_stats=stats(tre),
        identity_stats=stats(ident),
    )
    if write_cache:
        # Inference may have created the DVF file since the initial signature.
        signature = _cache_signature(run, field, src_off, dst_off)
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            _save_per_landmark_npz(cache_path, result, signature=signature)
        except OSError as exc:
            warnings.warn(f"TRE computed, but could not save cache {cache_path}: {exc}", stacklevel=2)
    return result


def _save_per_landmark_npz(path: Path, r: PerLandmarkResult, *, signature: str = "") -> None:
    import tempfile

    # A cancelled save must not replace a usable cache with a partial zip file.
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        _write_per_landmark_npz(temporary, r, signature=signature)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_per_landmark_npz(path: Path, r: PerLandmarkResult, *, signature: str) -> None:
    np.savez_compressed(
        path,
        signature=signature,
        case=r.case,
        field=r.field,
        which=r.which,
        pair=r.pair,
        frame=r.frame,
        truth_official=r.truth_official,
        pred_official=r.pred_official,
        src_official=r.src_official,
        truth_pack=r.truth_pack,
        pred_pack=r.pred_pack,
        src_pack=r.src_pack,
        err_vec_mm=r.err_vec_mm,
        tre_mm=r.tre_mm,
        identity_mm=r.identity_mm,
        oob_mask=r.oob_mask,
        registered_stats_json=np.asarray(json.dumps(r.registered_stats)),
        identity_stats_json=np.asarray(json.dumps(r.identity_stats)),
    )


def _load_per_landmark_npz(path: Path) -> PerLandmarkResult:
    with np.load(path, allow_pickle=False) as cache:
        z = {key: cache[key] for key in cache.files}
    return PerLandmarkResult(
        case=int(z["case"]),
        field=str(z["field"]),
        which=str(z["which"]),  # type: ignore[arg-type]
        pair=str(z["pair"]),  # type: ignore[arg-type]
        frame=str(z["frame"]),
        truth_official=z["truth_official"],
        pred_official=z["pred_official"],
        src_official=z["src_official"],
        truth_pack=z["truth_pack"],
        pred_pack=z["pred_pack"],
        src_pack=z["src_pack"],
        err_vec_mm=z["err_vec_mm"],
        tre_mm=z["tre_mm"],
        identity_mm=z["identity_mm"],
        oob_mask=z["oob_mask"].astype(bool),
        registered_stats=json.loads(str(z["registered_stats_json"])),
        identity_stats=json.loads(str(z["identity_stats_json"])),
    )


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
def verify_against_summary(
    run: RunRef,
    *,
    field: str | None = None,
    which: LandmarkSet = "75",
    pair: PhasePair = "T00_T50",
    atol: float = 1e-6,
) -> list[dict[str, Any]]:
    """Compare adapter per-landmark means to ``tre_summary.json`` / A0 identity.

    Returns a list of check dicts with ok/expected/got.
    """
    checks: list[dict[str, Any]] = []
    summary = load_tre_summary(run)

    # Identity always — compare to A0 and/or summary identity block
    pl_id = per_landmark(run, "identity", which=which, pair=pair, use_cache=False)
    a0_path = ARMS_ROOT / "A0_identity" / "results" / "summary.json"
    if a0_path.is_file():
        a0 = json.loads(a0_path.read_text())
        src_ph, dst_ph = pair_phases(pair)
        a0_case = next((c for c in a0["cases"]
                        if int(c["case"]) == run.case
                        and str(c.get("set")) == which
                        and {c.get("src"), c.get("dst")} == {src_ph, dst_ph}), None)
        if a0_case is not None:
            exp = float(a0_case["mean_mm"])
            got = float(pl_id.identity_stats["mean"])
            checks.append(
                {
                    "name": f"identity_vs_A0[C{run.case:02d}]",
                    "ok": abs(exp - got) <= atol,
                    "expected": exp,
                    "got": got,
                    "delta": got - exp,
                }
            )

    fields = [field] if field and field != "identity" else [
        f for f in run.fields_available if f not in ("identity", "voxelmap_ckpt")
    ]
    if field == "identity":
        fields = []
    if summary is None:
        for f in fields:
            pl = per_landmark(run, f, which=which, pair=pair, use_cache=False)
            checks.append(
                {
                    "name": f"{f}/{which}/{pair} (no summary)",
                    "ok": None,
                    "expected": None,
                    "got": pl.registered_stats["mean"],
                    "delta": None,
                    "note": "no tre_summary.json to compare",
                }
            )
        if not checks:
            checks.append({"name": "reference", "ok": None, "note": "No matching reference to verify"})
        return checks

    arms = summary.get("arms") or {}
    for f in fields:
        arm = arms.get(f)
        if not arm:
            checks.append(
                {
                    "name": f"{f}/{which}/{pair}",
                    "ok": False,
                    "expected": None,
                    "got": None,
                    "delta": None,
                    "note": f"field {f} missing from tre_summary.json",
                }
            )
            continue
        # Legacy top-level pairs contain the 75-point result only.
        block = (arm.get(which) or {}).get(pair)
        if not block and which == "75":
            block = arm.get(pair)
        if not block:
            checks.append(
                {
                    "name": f"{f}/{which}/{pair}",
                    "ok": False,
                    "expected": None,
                    "got": None,
                    "note": "pair missing in summary",
                }
            )
            continue
        exp = float(block["registered"]["mean"])
        pl = per_landmark(run, f, which=which, pair=pair, use_cache=False)
        got = float(pl.registered_stats["mean"])
        checks.append(
            {
                "name": f"{f}/{which}/{pair}",
                "ok": abs(exp - got) <= atol,
                "expected": exp,
                "got": got,
                "delta": got - exp,
                "oob": int(pl.oob_mask.sum()),
            }
        )
        # also identity inside the block
        exp_i = float(block["identity"]["mean"])
        got_i = float(pl.identity_stats["mean"])
        checks.append(
            {
                "name": f"{f}/{which}/{pair}/identity",
                "ok": abs(exp_i - got_i) <= atol,
                "expected": exp_i,
                "got": got_i,
                "delta": got_i - exp_i,
            }
        )
    if not checks:
        checks.append({"name": "reference", "ok": None, "note": "No matching reference to verify"})
    return checks
