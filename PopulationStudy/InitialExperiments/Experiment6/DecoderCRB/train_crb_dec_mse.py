"""Experiment 6 Decoder-CRB: A0 baseline on data_iso (same split as E1).

  PYTHONPATH=. PYTHONUNBUFFERED=1 python train_crb_dec_mse.py --gpu 1
"""

from __future__ import annotations

import sys
from pathlib import Path

DEC_ROOT = Path(__file__).resolve().parent
E6 = DEC_ROOT.parent
sys.path.insert(0, str(E6))
sys.path.insert(0, str(E6 / 'scripts'))

from train_mse import main as _main  # noqa: E402


if __name__ == '__main__':
    argv = sys.argv[1:]
    if '--arch' not in argv:
        argv = ['--arch', 'decoder'] + argv
    _main(argv)
