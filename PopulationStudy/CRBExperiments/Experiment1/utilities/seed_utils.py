"""Reproduce CRBExperiments Experiment 1 RNGs from seed.json."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

E1_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED_PATH = E1_ROOT / "seed.json"


def load_seed_config(path=None) -> dict:
    path = Path(path) if path else DEFAULT_SEED_PATH
    with open(path) as f:
        return json.load(f)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
