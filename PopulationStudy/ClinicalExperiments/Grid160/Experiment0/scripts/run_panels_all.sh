#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python}"
GPU="${GPU:-0}"
mkdir -p logs
echo "[$(date -Is)] A0 panels-only (100 pairs) ..."
PYTHONPATH=. "$PY" scripts/qc_clinical.py \
  --arms a0_full,a0h --norm both --panels-only --panel-pairs all --gpu "$GPU" \
  2>&1 | tee logs/qc_clinical_a0_panels_all.log
echo "[$(date -Is)] DONE"
