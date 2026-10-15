#!/usr/bin/env bash
# Start the 10-epoch X4BS pilot on GPU 0 after the X5a pilot (and its scorer) finish.
set -u
PY=/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python3
R=/home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Grid160/TCIA_lite
while systemctl --user is-active --quiet x5a-train; do sleep 120; done
echo "$(date) starting X4BS on GPU 0" > $R/run_A2_full160_aug_x4bs/logs/queue.log
systemd-run --user --unit=x4bs-tre --working-directory=$R/analysis_2026-10-03 --setenv=CUDA_VISIBLE_DEVICES=0 bash -c "$PY -u score_x4bs.py > score_x4bs.log 2>&1"
cd $R/run_A2_full160_aug_x4bs && exec $PY -u scripts/train_x4bs.py --gpu 0 --stop-after 10 > logs/train.log 2>&1
