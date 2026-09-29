#!/usr/bin/env bash
# Graceful stop: SIGTERM → train saves interrupt.pt (or finishes epoch → latest.pt).
# Usage: bash scripts/stop_train.sh
set -euo pipefail
VS="$(cd "$(dirname "$0")/.." && pwd)"
PIDFILE="$VS/logs/train.pid"

if [[ ! -f "$PIDFILE" ]]; then
  echo "no pid file at $PIDFILE — is training running?"
  pgrep -af "scripts/train_mae.py" | grep -F VoxelSynth || true
  exit 1
fi
PID=$(cat "$PIDFILE")
if ! kill -0 "$PID" 2>/dev/null; then
  echo "pid $PID not running; removing stale pidfile"
  rm -f "$PIDFILE"
  exit 1
fi

echo "sending SIGTERM to $PID (graceful checkpoint)…"
kill -TERM "$PID"
for i in $(seq 1 1800); do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "stopped after ${i}s"
    rm -f "$PIDFILE"
    ls -lt "$VS/DecoderCRB/checkpoints/" 2>/dev/null | head -8
    exit 0
  fi
  sleep 1
  if (( i % 30 == 0 )); then
    echo "  still waiting for graceful exit… ${i}s"
  fi
done

echo "still alive after 1800s — sending SIGKILL"
kill -KILL "$PID" 2>/dev/null || true
rm -f "$PIDFILE"
echo "killed hard (may need --resume from last epoch_XXX.pt)"
