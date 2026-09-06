#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python}"
GPU="${GPU:-0}"
mkdir -p logs
echo "[$(date -Is)] G160-E2 norm sweep ..."
PYTHONPATH=. "$PY" scripts/qc_norm_sweep.py \
  --arms a0_full,a0h,a1_full --arches encoder,decoder,both --gpu "$GPU" \
  2>&1 | tee logs/qc_norm_sweep.log
echo "[$(date -Is)] DONE"
