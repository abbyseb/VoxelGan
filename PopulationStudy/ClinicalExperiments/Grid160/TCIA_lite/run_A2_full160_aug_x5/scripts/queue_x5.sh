#!/usr/bin/env bash
# Start X5 when the breath-hold Elastix labels are done AND X4 has finished (one full-volume lite run at a time).
set -u
PY=/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python3
R=/home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Grid160/TCIA_lite
while systemctl --user is-active --quiet bh-labels || systemctl --user is-active --quiet x4-full; do sleep 120; done
n=$(ls /media/abhishek/3CCA3CADCA3C6574/LungPairs/packed_r3/B*_pair.npy | wc -l)
[ "$n" -ge 76 ] || { echo "$(date) only $n breath-hold pair files, not starting" > $R/run_A2_full160_aug_x5/logs/queue.log; exit 1; }
GPU=1; echo "$(date) starting X5 on GPU $GPU ($n breath-hold pair files)" > $R/run_A2_full160_aug_x5/logs/queue.log
systemd-run --user --unit=x5-tre --working-directory=$R/analysis_2026-10-03 --setenv=CUDA_VISIBLE_DEVICES=$GPU bash -c "$PY -u score_x5.py > score_x5.log 2>&1"
cd $R/run_A2_full160_aug_x5 && exec $PY -u scripts/train_x5.py --gpu $GPU > logs/train.log 2>&1
