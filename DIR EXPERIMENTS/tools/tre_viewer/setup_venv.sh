#!/usr/bin/env bash
# Create the viewer's dedicated environment.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv"
PYTHON_BIN="${PYTHON:-python3}"
if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
    echo "Python 3.10+ is required. Try: PYTHON=python3.11 bash setup_venv.sh" >&2
    exit 1
fi
"$PYTHON_BIN" -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -r "$ROOT/requirements.txt"
echo "OK: $VENV"
echo "Try: \"$VENV/bin/python\" -m tre_viewer doctor"
echo "  (run from: $ROOT/..  OR  PYTHONPATH=$ROOT/.. )"
