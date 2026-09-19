"""IsoExperiments E2 — Decoder-CRB on data_iso (2 mm, 160³).

  PYTHONPATH=.. PYTHONUNBUFFERED=1 python train_crb_dec_mse.py --gpu 0
"""

from __future__ import annotations

import sys
from pathlib import Path

DEC_ROOT = Path(__file__).resolve().parent
E2 = DEC_ROOT.parent
sys.path.insert(0, str(E2))
sys.path.insert(0, str(E2 / 'scripts'))

from train_mse import main as _main  # noqa: E402


if __name__ == '__main__':
    argv = sys.argv[1:]
    if '--arch' not in argv:
        argv = ['--arch', 'decoder'] + argv
    _main(argv)
