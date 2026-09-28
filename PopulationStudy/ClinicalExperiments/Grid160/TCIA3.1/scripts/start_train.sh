#!/usr/bin/env bash
# TCIA3.1 — MAE + high 01→06 scan oversampling
#   bash scripts/start_train.sh
#   GPU=0 bash scripts/start_train.sh fresh
set -euo pipefail
T31="$(cd "$(dirname "$0")/.." && pwd)"
cd "$T31"
PY="${PY:-/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python}"
GPU="${GPU:-0}"
MODE="${1:-auto}"
LOGDIR="$T31/logs"
mkdir -p "$LOGDIR" DecoderCRB/checkpoints DecoderCRB/weights DecoderCRB/plots data

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
  if [[ ! -f /media/abhishek/3CCA3CADCA3C6574/TCIA_4D-Lung/synth_g160_r3/pooled/manifest.json ]]; then
    echo "building pooled symlinks (once)…"
    "$PY" scripts/build_pooled_dataset.py
  fi
  echo "building oversampled train manifest…"
  "$PY" scripts/build_oversampled_manifest.py
fi

RESUME_ARGS=()
case "$MODE" in
  fresh) echo "starting FRESH (no resume)" ;;
  resume) RESUME_ARGS=(--resume auto); echo "RESUMING" ;;
  auto|*)
    if [[ -f DecoderCRB/checkpoints/latest.pt || -f DecoderCRB/checkpoints/interrupt.pt ]]; then
      RESUME_ARGS=(--resume auto)
      echo "auto: resume"
    else
      echo "auto: fresh start"
    fi
    ;;
esac

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOGDIR/train_decoder_${STAMP}.log"
ln -sfn "$(basename "$LOG")" "$LOGDIR/train_decoder_latest.log"

export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

echo "GPU(physical)=$GPU  log=$LOG  TCIA3.1 MAE+oversample"
nohup "$PY" -u scripts/train_mae_oversample.py --gpu 0 --epochs 100 --lr 1e-4 \
  "${RESUME_ARGS[@]}" \
  >"$LOG" 2>&1 &
echo $! >"$PIDFILE"
echo "started pid=$(cat "$PIDFILE")"
echo "  tail -f $LOG"
echo "  weights → DecoderCRB/weights/crb_dec_mae_oversample_iso_g160_tcia_r3_a1_generator.pth"
