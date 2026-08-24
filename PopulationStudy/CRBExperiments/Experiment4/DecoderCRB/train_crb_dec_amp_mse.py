"""CRBExperiments Experiment 4 — cyclic + δ_tgt + oracle amp.

  PYTHONPATH=. PYTHONUNBUFFERED=1 python train_crb_dec_amp_mse.py --gpu 0
"""
from __future__ import annotations

import sys
from pathlib import Path

DEC_ROOT = Path(__file__).resolve().parent
E4 = DEC_ROOT.parent
sys.path.insert(0, str(E4))
sys.path.insert(0, str(E4 / "scripts"))

from train_amp import main as _main  # noqa: E402


if __name__ == "__main__":
    _main()
