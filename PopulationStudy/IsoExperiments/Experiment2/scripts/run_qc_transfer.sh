#!/usr/bin/env bash
# Prep DIR iso + run Iso-E2 full QC on Clinical and DIR.
set -euo pipefail
E2="$(cd "$(dirname "$0")/.." && pwd)"
DIR_E2="$E2/../../DIR-Experiments/Experiment2"
PYTHON="${PYTHON:-/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python}"
GPU="${GPU:-0}"
LOG="$E2/logs/qc_transfer_run.log"
mkdir -p "$E2/logs"

exec > >(tee -a "$LOG") 2>&1
echo "=== Iso-E2 transfer QC $(date -Is) ==="

# 1) Clinical QC (128³ µ, metrics only)
cd "$E2"
PYTHONPATH=. $PYTHON scripts/qc_clinical.py \
  --arches encoder,decoder,both --norm both --no-panels --gpu "$GPU" --force

# 2) DIR iso pack (if missing)
if [[ ! -f "$DIR_E2/packed_iso/P1_DIR/all/CT_01.npy" ]]; then
  echo "=== pack_dirlab_iso ==="
  cd "$DIR_E2"
  $PYTHON scripts/pack_dirlab_iso.py
fi

# 3) DIR iso Elastix library (if pairs missing)
if [[ ! -f "$DIR_E2/packed_iso/P1_DIR/all/01_to_02_pair.npy" ]]; then
  echo "=== prepare_dir_dvf_library_iso (long) ==="
  cd "$DIR_E2"
  $PYTHON scripts/prepare_dir_dvf_library_iso.py \
    --patients P1_DIR,P2_DIR,P3_DIR,P4_DIR,P5_DIR,P6_DIR,P7_DIR,P8_DIR,P9_DIR,P10_DIR \
    --skip_existing
fi

# 4) DIR QC
cd "$E2"
PYTHONPATH=. $PYTHON scripts/qc_dirlab.py \
  --arches encoder,decoder,both --no-panels --gpu "$GPU" --force

echo "=== DONE $(date -Is) ==="
