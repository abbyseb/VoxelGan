#!/usr/bin/env bash
# Full-P1 train (no leave-out) → P2 zero-shot → P2 fine-tune → QC
# Overwrites P2/{our_warp,dans_gan,our_warp_ft,dans_gan_ft}.
#
# GPU0: Our Warp full → Dan full → Our P2 pipeline → Dan P2 pipeline
# GPU1: Dan CRB LOOCV (runs in parallel with GPU0 work)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python}"
mkdir -p plots P2/our_warp/qc P2/dans_gan/qc P2/our_warp_ft/qc P2/dans_gan_ft/qc
mkdir -p "Dan'sPaperGan/plots/loocv_crb" P2/logs

echo "=== $(date -Is) start master pipeline ===" | tee P2/logs/master.log

# --- GPU1: Dan LOOCV (long, background) ---
echo "=== GPU1: Dan LOOCV start ===" | tee -a P2/logs/master.log
(
  cd "$ROOT/Dan'sPaperGan"
  CUDA_VISIBLE_DEVICES=1 "$PY" -u train_loocv_crb.py
) > P2/logs/dan_loocv.log 2>&1 &
LOOCV_PID=$!
echo "Dan LOOCV pid=$LOOCV_PID" | tee -a P2/logs/master.log

# --- GPU0: Our Warp full P1 ---
echo "=== GPU0: Our Warp full P1 ===" | tee -a P2/logs/master.log
CUDA_VISIBLE_DEVICES=0 "$PY" -u train_full_p1_warp_d.py 2>&1 | tee P2/logs/train_our_full.log

# --- GPU0: Dan full P1 (overlaps with LOOCV on GPU1) ---
echo "=== GPU0: Dan CRB full P1 ===" | tee -a P2/logs/master.log
(
  cd "$ROOT/Dan'sPaperGan"
  CUDA_VISIBLE_DEVICES=0 "$PY" -u train_full_p1_crb.py
) 2>&1 | tee P2/logs/train_dan_full.log

# --- P2 zero-shot Our Warp (overwrite) ---
echo "=== P2 zero-shot Our Warp ===" | tee -a P2/logs/master.log
CUDA_VISIBLE_DEVICES=0 "$PY" -u scripts/qc_warp_d_all_pairs.py \
  --ckpt weights/spare_mc_p1_full_warp_d_generator.pth \
  --data_dir P2/data/all \
  --out_dir P2/our_warp/qc \
  --view_config configs/dvf_view_config.json \
  --summary P2/our_warp/metrics.tsv \
  2>&1 | tee P2/logs/qc_our_zs.log

# --- P2 fine-tune Our Warp from full-P1 ---
echo "=== P2 fine-tune Our Warp ===" | tee -a P2/logs/master.log
CUDA_VISIBLE_DEVICES=0 "$PY" -u scripts/finetune_p2_warp_d.py \
  --init_g weights/spare_mc_p1_full_warp_d_generator.pth \
  --init_d weights/spare_mc_p1_full_warp_d_discriminator.pth \
  --out_prefix spare_mc_p2_warp_d_finetune \
  --epochs 30 --lr_g 3e-5 --lr_d 1e-5 \
  2>&1 | tee P2/logs/ft_our_warp.log

# --- P2 tuned QC Our Warp (overwrite) ---
echo "=== P2 tuned QC Our Warp ===" | tee -a P2/logs/master.log
CUDA_VISIBLE_DEVICES=0 "$PY" -u scripts/qc_warp_d_all_pairs.py \
  --ckpt weights/spare_mc_p2_warp_d_finetune_generator.pth \
  --data_dir P2/data/all \
  --out_dir P2/our_warp_ft/qc \
  --view_config configs/dvf_view_config.json \
  --summary P2/our_warp_ft/metrics.tsv \
  2>&1 | tee P2/logs/qc_our_ft.log

# --- P2 zero-shot Dan ---
echo "=== P2 zero-shot Dan CRB ===" | tee -a P2/logs/master.log
CUDA_VISIBLE_DEVICES=0 "$PY" -u "Dan'sPaperGan/scripts/qc_crb_pairs.py" \
  --ckpt "Dan'sPaperGan/weights/dans_crb_p1_full_warp_d_generator.pth" \
  --data_dir P2/data/all \
  --out_dir P2/dans_gan/qc \
  --view_config configs/dvf_view_config.json \
  2>&1 | tee P2/logs/qc_dan_zs.log
cp -f P2/dans_gan/qc/metrics.tsv P2/dans_gan/metrics.tsv

# --- P2 fine-tune Dan from full-P1 ---
echo "=== P2 fine-tune Dan CRB ===" | tee -a P2/logs/master.log
CUDA_VISIBLE_DEVICES=0 "$PY" -u "Dan'sPaperGan/scripts/finetune_p2_crb_warp_d.py" \
  --init_g "Dan'sPaperGan/weights/dans_crb_p1_full_warp_d_generator.pth" \
  --init_d "Dan'sPaperGan/weights/dans_crb_p1_full_warp_d_discriminator.pth" \
  --out_prefix dans_crb_p2_warp_d_finetune \
  --epochs 30 --lr_g 3e-5 --lr_d 1e-5 \
  2>&1 | tee P2/logs/ft_dan_crb.log

# --- P2 tuned QC Dan ---
echo "=== P2 tuned QC Dan CRB ===" | tee -a P2/logs/master.log
CUDA_VISIBLE_DEVICES=0 "$PY" -u "Dan'sPaperGan/scripts/qc_crb_pairs.py" \
  --ckpt "Dan'sPaperGan/weights/dans_crb_p2_warp_d_finetune_generator.pth" \
  --data_dir P2/data/all \
  --out_dir P2/dans_gan_ft/qc \
  --view_config configs/dvf_view_config.json \
  2>&1 | tee P2/logs/qc_dan_ft.log
cp -f P2/dans_gan_ft/qc/metrics.tsv P2/dans_gan_ft/metrics.tsv

echo "=== waiting for Dan LOOCV (pid=$LOOCV_PID) ===" | tee -a P2/logs/master.log
wait "$LOOCV_PID"
echo "=== $(date -Is) ALL DONE ===" | tee -a P2/logs/master.log
