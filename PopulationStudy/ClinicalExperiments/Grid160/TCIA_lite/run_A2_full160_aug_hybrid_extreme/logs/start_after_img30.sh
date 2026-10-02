#!/bin/bash
# Waits for the img30 run (PID 2819310, GPU1) to exit, then starts the extreme run on GPU1 if img30 finished cleanly.
cd "$(dirname "$0")/.."
while kill -0 2819310 2>/dev/null; do sleep 60; done
if grep -q "^finished" ../run_A2_full160_aug_hybrid_img30/logs/train.log; then
  echo "$(date) img30 finished, starting extreme" >> logs/launcher.log
  /home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python -u scripts/train_aug_hybrid_extreme.py --gpu 1 > logs/train.log 2>&1
  echo "$(date) extreme exited code $?" >> logs/launcher.log
else
  echo "$(date) img30 did NOT finish cleanly, extreme not started" >> logs/launcher.log
fi
