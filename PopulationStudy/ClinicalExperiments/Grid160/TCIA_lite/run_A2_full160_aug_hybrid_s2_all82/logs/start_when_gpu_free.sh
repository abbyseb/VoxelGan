#!/bin/bash
# Starts the all-82 run on whichever GPU frees first:
#   GPU0 = TCIA3.5_hybrid (PID 2688725) has exited;  GPU1 = the GPU1 queue (img30 -> extreme) is fully done.
cd "$(dirname "$0")/.."
PY=/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python
while true; do
  if ! kill -0 2688725 2>/dev/null; then GPU=0; break; fi
  if ! pgrep -f start_after_img30.sh >/dev/null && ! pgrep -f "train_aug_hybrid_(img30|extreme).py" >/dev/null; then GPU=1; break; fi
  sleep 60
done
echo "$(date) GPU$GPU free, starting all82" >> logs/launcher.log
$PY -u scripts/train_aug_hybrid_s2_all82.py --gpu $GPU > logs/train.log 2>&1
echo "$(date) all82 exited code $?" >> logs/launcher.log
