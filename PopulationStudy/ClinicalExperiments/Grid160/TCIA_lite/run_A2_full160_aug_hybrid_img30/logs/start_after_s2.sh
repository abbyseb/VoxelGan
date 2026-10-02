#!/bin/bash
# Waits for hybrid seed 2 (PID 2688728, GPU1) to exit, then starts the img30 run on GPU1 only if s2 finished cleanly.
cd "$(dirname "$0")/.."
while kill -0 2688728 2>/dev/null; do sleep 60; done
if grep -q "^finished" ../run_A2_full160_aug_hybrid_s2/logs/train.log; then
  echo "$(date) s2 finished, starting img30" >> logs/launcher.log
  /home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python -u scripts/train_aug_hybrid_img30.py --gpu 1 > logs/train.log 2>&1
  echo "$(date) img30 exited code $?" >> logs/launcher.log
else
  echo "$(date) s2 did NOT finish cleanly, img30 not started" >> logs/launcher.log
fi
