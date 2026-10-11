#!/bin/bash
# A2-TCIA: prepare TCIA scans (GPU1 for DRR, 8 Elastix jobs in parallel on CPU), then train the concatenated NoFiLM VoxelMap
# with A2's exact recipe on GPU1 only. S01 was prepared as the test scan.
set -uo pipefail
cd "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS"
echo "$(date -Is) PREP START" >> arms/A2_tcia/logs/launcher.log
seq 2 82 | xargs -P 8 -I{} bash -c '/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python scripts/prepare_a2_tcia.py --scan {} --gpu 1 > arms/A2_tcia/logs/prep_S{}.log 2>&1 || echo "$(date -Is) PREP FAILED S{}" >> arms/A2_tcia/logs/launcher.log'
N=$(ls -d arms/A2_tcia/runs/TCIA_S*/ModelTraining/train/TCIA_S* 2>/dev/null | wc -l)
echo "$(date -Is) PREP DONE: $N scans ready" >> arms/A2_tcia/logs/launcher.log
DIRS=$(ls -d "$PWD"/arms/A2_tcia/runs/TCIA_S*/ModelTraining/train/TCIA_S*)
echo "$(date -Is) TRAIN START gpu=1 scans=$N" >> arms/A2_tcia/logs/launcher.log
CUDA_VISIBLE_DEVICES=1 /home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python /home/abhishek/Documents/VoxelMap_Clinical/ml/trainer.py --data_dirs $DIRS   --architecture concatenated --epochs 50 --batch_size 8 --lr 1e-5 --val_split 0.1 --device cuda --num_workers 4   --checkpoint_dir arms/A2_tcia/checkpoints --plots_dir arms/A2_tcia/plots_nofilm   --save_path arms/A2_tcia/checkpoints/a2_tcia_concat_nofilm.pt > arms/A2_tcia/logs/train_a2_tcia.log 2>&1
echo "$(date -Is) TRAIN EXIT $?" >> arms/A2_tcia/logs/launcher.log
