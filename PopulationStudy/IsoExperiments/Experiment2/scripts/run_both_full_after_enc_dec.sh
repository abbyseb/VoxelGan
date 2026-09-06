#!/usr/bin/env bash
# Run Both-CRB full P1–P9 after encoder/decoder full jobs finish.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python}"
GPU="${GPU:-1}"
LOG=logs/train_both_full.log
mkdir -p logs

while pgrep -f "train_mse.py --arch (encoder|decoder) --full" >/dev/null 2>&1; do
  echo "waiting for encoder/decoder full …" | tee -a "$LOG"
  sleep 120
done

echo "=== both full P1–P9 gpu=$GPU ===" | tee "$LOG"
PYTHONPATH=. PYTHONUNBUFFERED=1 $PYTHON scripts/train_mse.py \
  --arch both --full --gpu "$GPU" --epochs 100 2>&1 | tee -a "$LOG"
