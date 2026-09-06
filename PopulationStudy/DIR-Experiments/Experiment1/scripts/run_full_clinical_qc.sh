#!/bin/bash
# Build Elastix GT (100 pairs × 10 patients) then clinical-style QC with panels.
set -euo pipefail
E1=/home/abhishek/Voxel_GAN/PopulationStudy/DIR-Experiments/Experiment1
VENV=/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python
PY=python3
PATS=$(printf "P%d_DIR," {1..10} | sed 's/,$//')

cd "$E1"
echo "=== DIR Elastix DVF library (all patients) ==="
$VENV scripts/prepare_dir_dvf_library.py \
  --patients "$PATS" \
  --skip_existing \
  2>&1 | tee logs/prepare_dir_dvf_library.log

echo "=== DIR clinical QC (100 pairs + panels) ==="
PYTHONPATH=../../InitialExperiments/Experiment1 PYTHONUNBUFFERED=1 \
  $PY scripts/qc_e6_dirlab.py \
  --arches encoder,decoder,both \
  --norm both \
  --force \
  --gpu 0 \
  2>&1 | tee logs/qc_clinical_full.log

echo "ALL DONE"
