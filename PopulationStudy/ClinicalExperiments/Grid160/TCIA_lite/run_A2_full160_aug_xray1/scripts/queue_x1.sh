#!/usr/bin/env bash
# Wait for the LNCC-only run (RAM: one full-volume lite run at a time), then train X1 from scratch (40 epochs).
set -u
PY=/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python3
L=/home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Grid160/TCIA_lite
while systemctl --user is-active --quiet unsup-lncc; do sleep 60; done
cd $L/run_A2_full160_aug_xray1 && mkdir -p logs
exec $PY -u scripts/train_xray1.py --gpu 1 > logs/train.log 2>&1
