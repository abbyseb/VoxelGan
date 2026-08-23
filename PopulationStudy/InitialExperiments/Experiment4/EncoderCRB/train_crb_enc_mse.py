"""Experiment 4 Encoder-CRB — side-branch code ready, **not launched**.

Frozen AnatomyEncoder → GAP → concat onto phase cond. Do not start unless asked.

  PYTHONPATH=.. PYTHONUNBUFFERED=1 python train_crb_enc_mse.py --gpu 0
"""
from __future__ import annotations

import sys
from pathlib import Path

ENC_ROOT = Path(__file__).resolve().parent
E4 = ENC_ROOT.parent
sys.path.insert(0, str(E4))
sys.path.insert(0, str(E4 / "scripts"))

from train_mse import main as _main  # noqa: E402


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--arch" not in argv:
        argv = ["--arch", "encoder"] + argv
    _main(argv)
