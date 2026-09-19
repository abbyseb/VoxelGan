"""Load the experiment's coordinate adapter only when it is needed."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path


def require_evaluator():
    """Keep discovery/help usable when the unbundled evaluator is missing."""
    try:
        return import_module("eval_a1_tre")
    except ModuleNotFoundError as exc:
        if exc.name != "eval_a1_tre":
            raise
        path = Path(__file__).resolve().parents[2] / "scripts" / "eval_a1_tre.py"
        raise FileNotFoundError(
            f"Missing coordinate adapter: {path}. Restore eval_a1_tre.py from "
            "the experiment environment before computing TRE or viewing cases. "
            "Run 'python -m tre_viewer doctor' to check the setup."
        ) from exc
