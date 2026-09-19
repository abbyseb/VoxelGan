#!/usr/bin/env bash
# G160-A1: train Enc / Dec / Both with FOV aug on full P1–P9 (3 runs).
set -euo pipefail
E1="$(cd "$(dirname "$0")/.." && pwd)"
cd "$E1"
PY="${PY:-/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-100}"
for arch in encoder decoder both; do
  echo "======== G160-A1 $arch ========"
  PYTHONPATH=. PYTHONUNBUFFERED=1 "$PY" scripts/train_mse.py \
    --arch "$arch" --full --gpu "$GPU" --epochs "$EPOCHS" \
    2>&1 | tee "logs/train_${arch}_g160_fov_full.log"
done
echo "DONE G160-A1 all architectures"
