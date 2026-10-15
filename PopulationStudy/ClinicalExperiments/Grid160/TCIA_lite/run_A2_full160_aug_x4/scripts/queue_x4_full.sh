#!/usr/bin/env bash
# After the 10-epoch X4 pilot stops, resume the SAME run to 40 epochs (same cosine schedule) and restart the TRE watcher.
set -u
PY=/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python3
R=/home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Grid160/TCIA_lite
while systemctl --user is-active --quiet x4-train; do sleep 60; done
ep=$($PY -c "import torch;print(torch.load('$R/run_A2_full160_aug_x4/DecoderCRB/checkpoints/latest.pt',map_location='cpu',weights_only=False)['epoch_done'])")
[ "$ep" = "10" ] || { echo "$(date) unexpected epoch_done=$ep, not resuming" >> $R/run_A2_full160_aug_x4/logs/queue_full.log; exit 1; }
echo "$(date) resuming X4 from epoch $ep to 40" >> $R/run_A2_full160_aug_x4/logs/queue_full.log
while systemctl --user is-active --quiet x4-tre; do sleep 30; done
systemd-run --user --unit=x4-tre-full --working-directory=$R/analysis_2026-10-03 --setenv=CUDA_VISIBLE_DEVICES=1 bash -c "$PY -u score_x4.py >> score_x4.log 2>&1"
cd $R/run_A2_full160_aug_x4 && exec $PY -u scripts/train_x4.py --gpu 1 --resume >> logs/train.log 2>&1
