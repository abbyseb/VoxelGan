#!/usr/bin/env bash
# Prepare Elastix DVF (if needed) + 100-pair Elekta QC for P1–P5 V_01.
# QC panels: {Encoder,Decoder}CRB/plots/qc_elekta/P{n}/CE_P{n}_V_01/
set -euo pipefail
E1="$(cd "$(dirname "$0")/.." && pwd)"
CE="$(cd "$E1/.." && pwd)"
PY="${PY:-/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python}"
GPU="${GPU:-1}"
LOG="$E1/logs/prepare_and_qc_elekta.log"
mkdir -p "$E1/logs"
if [[ -t 1 ]]; then
  exec > >(tee -a "$LOG") 2>&1
else
  exec >>"$LOG" 2>&1
fi

cd "$CE"
SCANS=(CE_P1_V_01 CE_P2_V_01 CE_P3_V_01 CE_P4_V_01 CE_P5_V_01)

pair_count() {
  local d="$E1/data/$1/all"
  [[ -d "$d" ]] || { echo 0; return; }
  ls "$d"/*_pair.npy 2>/dev/null | wc -l
}

for scan in "${SCANS[@]}"; do
  n="$(pair_count "$scan" | tr -d '[:space:]')"
  echo "=== PREP $scan (have ${n:-0} pairs) $(date) ==="
  if [[ "${n:-0}" -ge 100 ]]; then
    echo "[skip prep] $scan already has $n pairs"
  else
    echo "=== Elastix $scan ==="
    PYTHONUNBUFFERED=1 "$PY" scripts/prepare_clinical_dvf_library.py \
      --scan "$scan" --skip_existing --no_save_native_pad
  fi

  for arch in encoder decoder; do
    echo "=== QC $arch $scan $(date) ==="
    cd "$E1"
    PYTHONPATH=. PYTHONUNBUFFERED=1 "$PY" scripts/qc_elekta.py \
      --arch "$arch" --scan "$scan" --all --gpu "$GPU"
    cd "$CE"
  done
done

echo "=== DONE $(date) ==="
