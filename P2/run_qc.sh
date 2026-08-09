#!/usr/bin/env bash
# Build P2/data/all and run frozen QC for our_warp + dans_gan (no training).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python}"

ALL="$ROOT/P2/data/all"
mkdir -p "$ALL" "$ROOT/P2/our_warp/qc" "$ROOT/P2/dans_gan/qc"

# Merge train+val into all (hardlink/copy CTs once, all pairs)
for split in train val; do
  for f in "$ROOT/P2/data/$split"/*; do
    base="$(basename "$f")"
    dest="$ALL/$base"
    if [[ ! -e "$dest" ]]; then
      ln "$f" "$dest" 2>/dev/null || cp -a "$f" "$dest"
    fi
  done
done

n_pairs=$(ls "$ALL"/*_pair.npy 2>/dev/null | wc -l)
echo "P2/data/all has $n_pairs pair files"

echo "=== our_warp (FiLM warp-D) ==="
"$PY" scripts/qc_warp_d_all_pairs.py \
  --ckpt weights/spare_mc_p1_scenario_warp_d_generator.pth \
  --data_dir P2/data/all \
  --out_dir P2/our_warp/qc \
  --view_config configs/dvf_view_config.json \
  --summary P2/our_warp/metrics.tsv

echo "=== dans_gan (UNetCRB) ==="
"$PY" "Dan'sPaperGan/scripts/qc_crb_pairs.py" \
  --ckpt "Dan'sPaperGan/weights/dans_crb_warp_d_generator.pth" \
  --data_dir "$ROOT/P2/data/all" \
  --out_dir "$ROOT/P2/dans_gan/qc" \
  --view_config "$ROOT/configs/dvf_view_config.json"
# promote metrics next to qc/
cp -f "$ROOT/P2/dans_gan/qc/metrics.tsv" "$ROOT/P2/dans_gan/metrics.tsv"

echo "Done. QC under P2/our_warp/qc and P2/dans_gan/qc"
