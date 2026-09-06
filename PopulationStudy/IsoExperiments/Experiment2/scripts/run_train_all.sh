#!/usr/bin/env bash
# Train Encoder, Decoder, Both sequentially (or set ARCH env to one arch).
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-100}"

$PYTHON scripts/build_pooled_dataset.py

for arch in encoder decoder both; do
  if [[ -n "${ARCH:-}" && "${ARCH}" != "$arch" ]]; then
    continue
  fi
  log="logs/train_${arch}.log"
  mkdir -p logs
  echo "=== $arch gpu=$GPU ===" | tee "$log"
  PYTHONPATH=. PYTHONUNBUFFERED=1 $PYTHON scripts/train_mse.py \
    --arch "$arch" --gpu "$GPU" --epochs "$EPOCHS" 2>&1 | tee -a "$log"
done
