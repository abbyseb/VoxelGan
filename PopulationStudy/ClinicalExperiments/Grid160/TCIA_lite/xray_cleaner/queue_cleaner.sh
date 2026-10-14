#!/usr/bin/env bash
# Start the X-ray Cleaner on the first GPU that frees up: GPU 0 when X3S training ends, GPU 1 when the look-and-correct pilot ends.
set -u
PY=/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python3
cd /home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Grid160/TCIA_lite/xray_cleaner && mkdir -p run
while true; do
  if ! systemctl --user is-active --quiet xray3s-train; then GPU=0; break; fi
  if ! systemctl --user is-active --quiet lc-pilot; then GPU=1; break; fi
  sleep 60
done
echo "$(date) starting on GPU $GPU" > run/queue.log
exec $PY -u train_cleaner.py --gpu $GPU > run/train.log 2>&1
