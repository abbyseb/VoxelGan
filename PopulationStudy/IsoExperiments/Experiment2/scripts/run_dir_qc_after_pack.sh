#!/usr/bin/env bash
set -euo pipefail
E2="$(cd "$(dirname "$0")/.." && pwd)"
DIR_E2="$(cd "$E2/../../DIR-Experiments/Experiment2" && pwd)"
PYTHON="${PYTHON:-/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python}"
GPU="${GPU:-0}"
LOG="$E2/logs/qc_dir_pipeline.log"
mkdir -p "$E2/logs" "$DIR_E2/logs"

exec >>"$LOG" 2>&1
echo "=== wait pack $(date -Is) ==="
while [[ ! -f "$DIR_E2/packed_iso/P10_DIR/pack_meta.json" ]]; do
  sleep 60
done
echo "=== prepare_dir_dvf_library_iso $(date -Is) ==="
cd "$DIR_E2"
$PYTHON scripts/prepare_dir_dvf_library_iso.py \
  --patients P1_DIR,P2_DIR,P3_DIR,P4_DIR,P5_DIR,P6_DIR,P7_DIR,P8_DIR,P9_DIR,P10_DIR \
  --skip_existing
echo "=== qc_dirlab $(date -Is) ==="
cd "$E2"
PYTHONPATH=. PYTHONUNBUFFERED=1 $PYTHON scripts/qc_dirlab.py \
  --arches encoder,decoder,both --no-panels --gpu "$GPU" --force \
  2>&1 | tee -a "$E2/logs/qc_dirlab_full.log"
echo "=== DIR QC done $(date -Is) ==="
