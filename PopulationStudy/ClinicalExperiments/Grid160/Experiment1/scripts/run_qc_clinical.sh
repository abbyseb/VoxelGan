#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python}"
GPU="${GPU:-0}"
mkdir -p logs
echo "[$(date -Is)] A1 clinical QC (metrics + 100-pair panels) ..."
PYTHONPATH=. "$PY" scripts/qc_clinical.py \
  --arms a1_full --norm both --panel-pairs all --gpu "$GPU" \
  2>&1 | tee logs/qc_clinical_a1_full.log
echo "[$(date -Is)] DONE"
