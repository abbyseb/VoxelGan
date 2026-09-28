#!/usr/bin/env bash
# Start (or resume) G160-TCIA3 Decoder MAE training on a free GPU.
# Usage:
#   bash scripts/start_train.sh              # fresh if no ckpt, else --resume auto
#   bash scripts/start_train.sh fresh        # ignore checkpoints, start epoch 1
#   bash scripts/start_train.sh resume       # force --resume auto
#   GPU=0 bash scripts/start_train.sh
set -euo pipefail
T3="$(cd "$(dirname "$0")/.." && pwd)"
cd "$T3"
PY="${PY:-/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python}"
GPU="${GPU:-0}"
MODE="${1:-auto}"
LOGDIR="$T3/logs"
mkdir -p "$LOGDIR" DecoderCRB/checkpoints DecoderCRB/weights DecoderCRB/plots

PIDFILE="$LOGDIR/train.pid"
if [[ -f "$PIDFILE" ]]; then
  old=$(cat "$PIDFILE" || true)
  if [[ -n "${old:-}" ]] && kill -0 "$old" 2>/dev/null; then
    echo "already running pid=$old (stop with scripts/stop_train.sh)"
    exit 1
  fi
  rm -f "$PIDFILE"
fi

if [[ ! -f data/manifest.json ]]; then
  echo "building pooled dataset…"
  "$PY" scripts/build_pooled_dataset.py
fi

RESUME_ARGS=()
case "$MODE" in
  fresh)
    echo "starting FRESH (no resume)"
    ;;
  resume)
    RESUME_ARGS=(--resume auto)
    echo "RESUMING from latest/interrupt checkpoint"
    ;;
  auto|*)
    if [[ -f DecoderCRB/checkpoints/latest.pt || -f DecoderCRB/checkpoints/interrupt.pt ]]; then
      RESUME_ARGS=(--resume auto)
      echo "auto: found checkpoint → resume"
    else
      echo "auto: no checkpoint → fresh start"
    fi
    ;;
esac

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOGDIR/train_decoder_${STAMP}.log"
ln -sfn "$(basename "$LOG")" "$LOGDIR/train_decoder_latest.log"

export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

echo "GPU(physical)=$GPU  log=$LOG  loss=MAE (TCIA3)"
nohup "$PY" -u scripts/train_mae.py --gpu 0 --epochs 100 --lr 1e-4 \
  "${RESUME_ARGS[@]}" \
  >"$LOG" 2>&1 &
echo $! >"$PIDFILE"
echo "started pid=$(cat "$PIDFILE")"
echo "  tail -f $LOG"
echo "  bash scripts/stop_train.sh"
echo "  bash scripts/start_train.sh resume"
echo "  weights → DecoderCRB/weights/crb_dec_mae_iso_g160_tcia_r3_a1_generator.pth"
