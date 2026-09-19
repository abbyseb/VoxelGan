#!/usr/bin/env bash
# G160-A1: train Enc / Dec / Both in parallel on one GPU.
set -euo pipefail
E1="$(cd "$(dirname "$0")/.." && pwd)"
cd "$E1"
PY="${PY:-/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-100}"
mkdir -p logs

for arch in "$@"; do
  log="logs/train_${arch}_g160_fov_full.log"
  if pgrep -f "train_mse.py --arch ${arch} --full --gpu ${GPU}" >/dev/null 2>&1; then
    echo "skip $arch (already running)"
    continue
  fi
  echo "starting $arch on cuda:$GPU → $log"
  nohup env PYTHONPATH=. PYTHONUNBUFFERED=1 "$PY" scripts/train_mse.py \
    --arch "$arch" --full --gpu "$GPU" --epochs "$EPOCHS" \
    >> "$log" 2>&1 &
  echo "  pid=$!"
done
