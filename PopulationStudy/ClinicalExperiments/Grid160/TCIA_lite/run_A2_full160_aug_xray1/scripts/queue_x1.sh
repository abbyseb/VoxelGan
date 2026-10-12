#!/usr/bin/env bash
# Wait for the LNCC-only run (RAM: one full-volume lite run at a time), then X1, then its matched control.
set -u
PY=/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python3
L=/home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Grid160/TCIA_lite
while systemctl --user is-active --quiet unsup-lncc; do sleep 60; done
test -f $L/run_A2_full160_aug_unsup_lncc/DecoderCRB/checkpoints/epoch_040.pt || { echo "no epoch_040 - abort"; exit 1; }
cd $L/run_A2_full160_aug_xray1 && mkdir -p logs
$PY -u scripts/train_xray1.py --gpu 1 > logs/train.log 2>&1 || exit 1
mkdir -p $L/run_A2_full160_aug_xray1_ctrl/logs
$PY -u scripts/train_xray1.py --gpu 1 --zero-xray > $L/run_A2_full160_aug_xray1_ctrl/logs/train.log 2>&1
