"""Experiment 4 Both-CRB — side-branch code ready, **not launched**.

  PYTHONPATH=.. PYTHONUNBUFFERED=1 python train_crb_both_mse.py --gpu 0
"""
from __future__ import annotations

import sys
from pathlib import Path

BOTH_ROOT = Path(__file__).resolve().parent
E4 = BOTH_ROOT.parent
sys.path.insert(0, str(E4))
sys.path.insert(0, str(E4 / "scripts"))

from train_mse import main as _main  # noqa: E402


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--arch" not in argv:
        argv = ["--arch", "both"] + argv
    _main(argv)
