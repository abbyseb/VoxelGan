#!/usr/bin/env bash
# Start the 10-epoch X6 pilot on GPU 0 when the teacher keypoints are done.
set -u
PY=/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python3
R=/home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Grid160/TCIA_lite
while systemctl --user is-active --quiet teacher-0 || systemctl --user is-active --quiet teacher-1; do sleep 120; done
n=$(ls /media/abhishek/3CCA3CADCA3C6574/TCIA_4D-Lung/teacher_kp/*_kp.npz | wc -l)
echo "$(date) starting X6 on GPU 0 ($n teacher keypoint files)" > $R/run_A2_full160_aug_x6/logs/queue.log
systemd-run --user --unit=x6-tre --working-directory=$R/analysis_2026-10-03 --setenv=CUDA_VISIBLE_DEVICES=0 bash -c "$PY -u score_x6.py > score_x6.log 2>&1"
cd $R/run_A2_full160_aug_x6 && exec $PY -u scripts/train_x6.py --gpu 0 --stop-after 10 > logs/train.log 2>&1
