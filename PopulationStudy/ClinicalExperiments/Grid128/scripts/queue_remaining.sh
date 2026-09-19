#!/usr/bin/env bash
# ClinicalExperiments — remaining train/QC queue (idempotent).
#
# Safe to re-run: skips steps whose outputs already exist.
# Does NOT kill running jobs; waits for them / for a free GPU.
#
# Usage:
#   nohup bash ClinicalExperiments/scripts/queue_remaining.sh \
#     >> ClinicalExperiments/logs/queue_remaining.log 2>&1 &
set -euo pipefail

CE="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python}"
LOG_DIR="$CE/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/queue_remaining.log"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*"; }

wait_pid() {
  local pid="$1" name="$2"
  if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
    say "wait_pid: $name not running (ok)"
    return 0
  fi
  say "waiting for $name pid=$pid"
  while kill -0 "$pid" 2>/dev/null; do sleep 60; done
  say "finished $name"
}

# First GPU with memory.used < threshold (MiB)
free_gpu() {
  local thr="${1:-500}"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | while IFS=',' read -r idx used; do
        idx=$(echo "$idx" | tr -d ' ')
        used=$(echo "$used" | tr -d ' ')
        if [[ "$used" -lt "$thr" ]]; then
          echo "$idx"
          return 0
        fi
      done
  return 1
}

wait_free_gpu() {
  local thr="${1:-500}"
  local g=""
  while true; do
    g="$(free_gpu "$thr" || true)"
    if [[ -n "$g" ]]; then
      echo "$g"
      return 0
    fi
    say "no free GPU (<${thr} MiB); sleep 120s"
    sleep 120
  done
}

has_weight() {
  local f="$1"
  [[ -f "$f" && -s "$f" ]]
}

qc_all_scans() {
  local exp_dir="$1" arch="$2" script="$3" prefix="$4" gpu="$5"
  local scans=()
  if [[ "$prefix" == "CV" ]]; then
    scans=(CV_P1_V_01 CV_P2_V_01 CV_P3_V_01 CV_P4_V_01 CV_P5_V_01)
  else
    scans=(CE_P1_V_01 CE_P2_V_01 CE_P3_V_01 CE_P4_V_01 CE_P5_V_01)
  fi
  local arch_dir vendor_dir
  case "$arch" in
    encoder) arch_dir=EncoderCRB ;;
    decoder) arch_dir=DecoderCRB ;;
    both) arch_dir=BothCRB ;;
    *) say "bad arch $arch"; return 1 ;;
  esac
  vendor_dir="qc_varian"
  [[ "$prefix" == "CE" ]] && vendor_dir="qc_elekta"
  if [[ ! -f "$exp_dir/scripts/$script" ]]; then
    say "WARN: missing $exp_dir/scripts/$script — skip"
    return 0
  fi
  cd "$exp_dir"
  for scan in "${scans[@]}"; do
    local patient
    patient="$(echo "$scan" | sed -E "s/C[VE]_(P[0-9]+)_.*/\1/")"
    local out="$exp_dir/$arch_dir/plots/$vendor_dir/$patient/$scan/summary.json"
    if [[ -f "$out" ]]; then
      say "skip QC $arch $scan (exists)"
      continue
    fi
    # need packed data
    if [[ ! -d "$exp_dir/data/$scan/all" && ! -d "$CE/Experiment1/data/$scan/all" ]]; then
      say "skip QC $arch $scan (no data)"
      continue
    fi
    say "QC $script --arch $arch --scan $scan gpu=$gpu"
    PYTHONPATH=. PYTHONUNBUFFERED=1 "$PY" "scripts/$script" \
      --arch "$arch" --scan "$scan" --all --gpu "$gpu"
  done
}

say "=== queue_remaining START ==="
say "CE=$CE"

# --- Discover live training PIDs (by cwd + cmdline) ---
E1_BOTH_PID=""
E2_CHAIN_PID=""
E2_ENC_PID=""
E3_BOTH_PID=""
while read -r pid cmd; do
  cwd="$(readlink -f /proc/$pid/cwd 2>/dev/null || true)"
  case "$cwd" in
    */ClinicalExperiments/Experiment1)
      if [[ "$cmd" == *"--arch both"* ]]; then E1_BOTH_PID="$pid"; fi
      ;;
    */ClinicalExperiments/Experiment2)
      if [[ "$cmd" == *"--arch encoder"* ]]; then E2_ENC_PID="$pid"; fi
      if [[ "$cmd" == *"--arch decoder"* ]]; then E2_ENC_PID="$pid"; fi
      ;;
    */ClinicalExperiments/Experiment3)
      if [[ "$cmd" == *"--arch both"* ]]; then E3_BOTH_PID="$pid"; fi
      ;;
  esac
done < <(pgrep -af 'python.*scripts/train_mse.py' | grep -v cursorsandbox | grep -v pgrep || true)

# Parent bash chain for E2 enc→dec
E2_CHAIN_PID="$(pgrep -f 'Experiment2.*train_mse.py --arch encoder' | head -1 || true)"
# Prefer the bash wrapper if present
WRAP="$(pgrep -f 'E2 encoder start|Experiment2.*encoder_then_decoder|cd \"\$E2\".*encoder' || true)"
# fallback: find bash with Experiment2 train chain
for pid in $(pgrep -f 'bash -c' || true); do
  if tr '\0' '\n' < /proc/$pid/cmdline 2>/dev/null | grep -q 'Experiment2' \
    && tr '\0' '\n' < /proc/$pid/cmdline 2>/dev/null | grep -q 'encoder'; then
    E2_CHAIN_PID="$pid"
  fi
done

say "detected E1_BOTH_PID=${E1_BOTH_PID:-none} E2_ENC/chain=${E2_ENC_PID:-none}/${E2_CHAIN_PID:-none} E3_BOTH_PID=${E3_BOTH_PID:-none}"

# ========== 1) Finish E2 Encoder → Decoder (already chained) ==========
if [[ -n "${E2_CHAIN_PID:-}" ]] && kill -0 "$E2_CHAIN_PID" 2>/dev/null; then
  wait_pid "$E2_CHAIN_PID" "E2 enc→dec chain"
elif [[ -n "${E2_ENC_PID:-}" ]] && kill -0 "$E2_ENC_PID" 2>/dev/null; then
  wait_pid "$E2_ENC_PID" "E2 active train"
else
  say "E2 enc/dec chain not running"
fi

# Also wait if decoder still going under a new pid
while pgrep -af 'ClinicalExperiments/Experiment2.*train_mse' | grep -v pgrep | grep -q .; do
  say "E2 train still active; sleep 60"
  sleep 60
done

# ========== 2) E2 Both (if missing) ==========
E2="$CE/Experiment2"
E2_BOTH_W="$E2/BothCRB/weights/crb_both_mse_cyclic_full_spare_generator.pth"
if has_weight "$E2_BOTH_W"; then
  say "skip E2 Both train (weight exists)"
else
  GPU="$(wait_free_gpu 800)"
  say "=== E2 Both train on GPU $GPU ==="
  cd "$E2"
  PYTHONPATH=. PYTHONUNBUFFERED=1 "$PY" scripts/train_mse.py --arch both --gpu "$GPU" --epochs 100 \
    2>&1 | tee -a "$E2/BothCRB/plots/train_crb_both_mse_cyclic_full_spare.log"
fi

# ========== 3) Wait E1 Both + E3 Both ==========
# Re-detect — only real python trainers (cursorsandbox shells embed these strings)
E1_BOTH_PID="$(pgrep -af 'python.*Experiment1.*train_mse.py --arch both' | grep -v cursorsandbox | grep -v pgrep | awk '{print $1}' | head -1 || true)"
E3_BOTH_PID="$(pgrep -af 'python.*Experiment3.*train_mse.py --arch both' | grep -v cursorsandbox | grep -v pgrep | awk '{print $1}' | head -1 || true)"
[[ -n "${E1_BOTH_PID:-}" ]] && wait_pid "$E1_BOTH_PID" "E1 Both" || say "E1 Both already done"
# E3 may still run
while pgrep -af 'python.*Experiment3.*train_mse' | grep -v cursorsandbox | grep -v pgrep | grep -q .; do
  say "E3 train still active; sleep 60"
  sleep 60
done
say "E3 Both train idle"

# ========== 4) QC backlog ==========
GPU="$(wait_free_gpu 500)"
say "QC using GPU $GPU"

# E1 Both QC (varian + elekta)
E1="$CE/Experiment1"
if has_weight "$E1/BothCRB/weights/crb_both_mse_full_spare_generator.pth"; then
  qc_all_scans "$E1" both qc_varian.py CV "$GPU"
  qc_all_scans "$E1" both qc_elekta.py CE "$GPU"
else
  say "WARN: E1 Both weight missing; skip Both QC"
fi

# E2 QC encoder/decoder/both × varian (+ elekta if data present)
for arch in encoder decoder both; do
  case "$arch" in
    encoder) w="$E2/EncoderCRB/weights/crb_enc_mse_cyclic_full_spare_generator.pth" ;;
    decoder) w="$E2/DecoderCRB/weights/crb_dec_mse_cyclic_full_spare_generator.pth" ;;
    both) w="$E2/BothCRB/weights/crb_both_mse_cyclic_full_spare_generator.pth" ;;
  esac
  if has_weight "$w"; then
    qc_all_scans "$E2" "$arch" qc_varian.py CV "$GPU"
    qc_all_scans "$E2" "$arch" qc_elekta.py CE "$GPU"
  else
    say "skip E2 $arch QC (no weight)"
  fi
done

# E3 Both QC
E3="$CE/Experiment3"
if has_weight "$E3/BothCRB/weights/crb_both_mse_cyclic_fov_aug_generator.pth"; then
  qc_all_scans "$E3" both qc_varian.py CV "$GPU"
  qc_all_scans "$E3" both qc_elekta.py CE "$GPU"
else
  say "skip E3 Both QC (no weight)"
fi

say "=== queue_remaining DONE ==="
