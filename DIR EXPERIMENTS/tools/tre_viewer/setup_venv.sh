#!/usr/bin/env bash
# Create dedicated tre_viewer venv (Phase 0 deps).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -r "$ROOT/requirements.txt"
echo "OK: $VENV"
echo "Try: $VENV/bin/python -m tre_viewer list-runs"
echo "  (run from: $ROOT/..  OR  PYTHONPATH=$ROOT/.. )"
