#!/usr/bin/env bash
# Wait for X2 training (one full-volume lite run at a time on GPU 1), then train X2R from scratch (40 epochs).
set -u
PY=/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python3
L=/home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Grid160/TCIA_lite
while systemctl --user is-active --quiet xray2-train; do sleep 60; done
cd $L/run_A2_full160_aug_xray2r && mkdir -p logs
exec $PY -u scripts/train_xray2r.py --gpu 1 > logs/train.log 2>&1
