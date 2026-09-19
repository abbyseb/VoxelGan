"""IsoExperiments E2 — Encoder-CRB on data_iso (2 mm, 160³).

  PYTHONPATH=.. PYTHONUNBUFFERED=1 python train_crb_enc_mse.py --gpu 0
"""

from __future__ import annotations

import sys
from pathlib import Path

ENC_ROOT = Path(__file__).resolve().parent
E2 = ENC_ROOT.parent
sys.path.insert(0, str(E2))
sys.path.insert(0, str(E2 / 'scripts'))

from train_mse import main as _main  # noqa: E402


if __name__ == '__main__':
    argv = sys.argv[1:]
    if '--arch' not in argv:
        argv = ['--arch', 'encoder'] + argv
    _main(argv)
