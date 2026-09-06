#!/usr/bin/env python3
"""Stage a VoxelMap_Experiments arm as a LEARN-GUI-compatible run folder.

LEARN-GUI DVF Visualisation (gui/ui/dvf_visualisation_dialog.py) expects a patient
test folder with:

  ModelTraining/test/<patient_id>/
    TestProjections/          Proj_*_bin.npy + RespBin.csv
    SourceTestProjections/    06_Proj_*_bin.npy
    SourceProjections/        (train split; optional for test viz)
    SourceVolumes/            sub_CT_06_mha.npy
    DVFs/                     DVF_XX_mha.npy
    Masks/                    sub_Abdomen_mha.npy, Mask_PTV_mha.npy, ...
    Angles.csv
    RespBin.csv

Opening from the GUI (tabs._open_dvf_visualisation) auto-fills:
  patient  -> <run>/ModelTraining/test/<first subdir>
  model    -> <run>/Models/*.pth  (latest)

Usage:
  python scripts/setup_learn_gui_run.py --arm elastix
  python scripts/setup_learn_gui_run.py --arm synth --link-learn-gui
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
LEARN = Path(os.environ.get("LEARN_GUI_ROOT", "/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python"))
DEFAULT_SCAN = "CV_P3_V_01"


def export_learn_gui_weights(ckpt: Path, out_pth: Path) -> None:
    """LEARN-GUI DVF dialog loads with strict=True; include transformer.grid buffers."""
    import sys
    import torch

    learn = Path(os.environ.get("LEARN_GUI_ROOT", "/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python"))
    if str(learn) not in sys.path:
        sys.path.insert(0, str(learn))
    from ml.utilities import networksFiLM

    raw = torch.load(ckpt, map_location="cpu")
    if isinstance(raw, dict) and "model_state" in raw:
        sd = raw["model_state"]
        cfg = raw.get("config") or {}
    elif isinstance(raw, dict) and "state_dict" in raw:
        sd = raw["state_dict"]
        cfg = {}
    else:
        sd = raw
        cfg = {}

    arch = cfg.get("architecture", "concatenated")
    use_film = bool(cfg.get("use_film", False))
    im_size = int(cfg.get("im_size", 128))

    model = networksFiLM.Model(
        architecture=arch,
        im_size=im_size,
        in_channels=int(cfg.get("in_channels", 1)),
        use_film=use_film,
    )
    model.load_state_dict(sd, strict=False)
    out_pth.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_pth)


def setup_arm(arm: str, scan_id: str, link_learn_gui: bool, run_name: str | None) -> Path:
    from run_paths import resolve_arm, run_root as rr

    arm = resolve_arm(arm)
    run_root = rr(scan_id, arm)
    if not (run_root / "ModelTraining" / "test" / scan_id).is_dir():
        raise SystemExit(f"Missing test patient dir: {run_root / 'ModelTraining' / 'test' / scan_id}")

    ckpt = run_root / "checkpoints_nofilm" / "best.pt"
    if not ckpt.is_file():
        raise SystemExit(f"Missing checkpoint: {ckpt}")

    # LEARN-GUI tabs.py looks for run/Models/*.pth
    models_dir = run_root / "Models"
    models_dir.mkdir(exist_ok=True)
    pth_name = f"weights_concatenated_nofilm_{arm}_best.pth"
    pth_path = models_dir / pth_name
    export_learn_gui_weights(ckpt, pth_path)

    # Optional external log dir (matches run_folder_manager layout)
    log_run_name = run_name or f"{scan_id}_{arm}_VoxelMap"
    logs_dir = run_root / "logs" / log_run_name
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"run_{stamp}.log"
    if not log_file.exists():
        log_file.write_text(
            f"VoxelMap_Experiments {arm} arm — LEARN-GUI compatible run\n"
            f"scan={scan_id}\n"
            f"checkpoint={ckpt}\n"
            f"test_patient={run_root / 'ModelTraining' / 'test' / scan_id}\n",
            encoding="utf-8",
        )

    readme = run_root / "LEARN_GUI_README.txt"
    readme.write_text(
        "\n".join(
            [
                "LEARN-GUI DVF Visualisation",
                "============================",
                "",
                f"Run folder: {run_root}",
                "",
                "1. Open LEARN-GUI-Python and load this folder as the active run, OR",
                "   Menu → DVF Visualisation and set:",
                f"   Patient folder: ModelTraining/test/{scan_id}",
                f"   Model weights:  Models/{pth_name}",
                "",
                "2. Click Index, then Load model. Architecture: concatenated, FiLM: off.",
                "",
                "Required test layout (already present):",
                "  TestProjections/, SourceTestProjections/, DVFs/, Masks/, SourceVolumes/",
                "",
                f"Raw acquisition (prep_train source): {run_root / scan_id / 'train'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    if link_learn_gui:
        learn_runs = LEARN / "Runs"
        learn_runs.mkdir(parents=True, exist_ok=True)
        link_name = run_name or f"{scan_id}_{arm}_VoxelMap"
        link_path = learn_runs / link_name
        if link_path.is_symlink():
            link_path.unlink()
        elif link_path.exists():
            raise SystemExit(f"Refusing to overwrite existing: {link_path}")
        link_path.symlink_to(run_root.resolve())
        print(f"Linked LEARN-GUI Runs/{link_name} -> {run_root}")

    print(f"Exported Models/{pth_name} (flat state_dict from {ckpt.name})")
    print(f"DVF patient: {run_root / 'ModelTraining' / 'test' / scan_id}")
    return run_root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm",
        required=True,
        choices=["elastix", "synth", "synth_g160_a1_dec", "synth_e3_both"],
    )
    ap.add_argument("--scan-id", default=DEFAULT_SCAN)
    ap.add_argument("--link-learn-gui", action="store_true", help="Symlink into LEARN-GUI-Python/Runs/")
    ap.add_argument("--run-name", default=None, help="Folder name under LEARN-GUI Runs/")
    args = ap.parse_args()
    setup_arm(args.arm, args.scan_id, args.link_learn_gui, args.run_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
