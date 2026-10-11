#!/bin/bash
# A2-TCIA training only (prep done 2026-10-08). Paths contain a space ("DIR EXPERIMENTS") -> pass them as a quoted array.
set -uo pipefail
cd "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS"
mapfile -t DIRS < <(ls -d "$PWD"/arms/A2_tcia/runs/TCIA_S*/ModelTraining/train/TCIA_S*)
echo "$(date -Is) TRAIN START gpu=1 scans=${#DIRS[@]}" >> arms/A2_tcia/logs/launcher.log
CUDA_VISIBLE_DEVICES=1 /home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python /home/abhishek/Documents/VoxelMap_Clinical/ml/trainer.py \
  --data_dirs "${DIRS[@]}" --architecture concatenated --epochs 50 --batch_size 8 --lr 1e-5 --val_split 0.1 --device cuda --num_workers 4 \
  --checkpoint_dir arms/A2_tcia/checkpoints --plots_dir arms/A2_tcia/plots_nofilm \
  --save_path arms/A2_tcia/checkpoints/a2_tcia_concat_nofilm.pt > arms/A2_tcia/logs/train_a2_tcia.log 2>&1
echo "$(date -Is) TRAIN EXIT $?" >> arms/A2_tcia/logs/launcher.log
