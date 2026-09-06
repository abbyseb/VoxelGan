#!/usr/bin/env bash
# Finish metrics (if needed) then write all 100 phase-pair PNGs per scan.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python}"
GPU="${GPU:-0}"
mkdir -p logs

echo "[$(date -Is)] metrics QC (skip existing summaries) ..."
PYTHONPATH=. "$PY" scripts/qc_clinical.py \
  --arms a0_full,a0h --norm both --no-panels --gpu "$GPU" \
  2>&1 | tee -a logs/qc_clinical_a0_metrics.log

echo "[$(date -Is)] panels QC (all 100 pairs) ..."
PYTHONPATH=. "$PY" scripts/qc_clinical.py \
  --arms a0_full,a0h --norm both --panels-only --panel-pairs all --gpu "$GPU" \
  2>&1 | tee logs/qc_clinical_a0_panels_all.log

echo "[$(date -Is)] DONE"
