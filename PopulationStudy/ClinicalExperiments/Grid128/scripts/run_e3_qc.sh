#!/usr/bin/env bash
set -euo pipefail
CE="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python}"
E3="$CE/Experiment3"
GPU="${GPU:-0}"
cd "$E3"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*"; }

say "=== E3 Both QC START gpu=$GPU ==="
for scan in CV_P1_V_01 CV_P2_V_01 CV_P3_V_01 CV_P4_V_01 CV_P5_V_01; do
  patient="$(echo "$scan" | sed -E 's/CV_(P[0-9]+)_.*/\1/')"
  out="$E3/BothCRB/plots/qc_varian/$patient/$scan/summary.json"
  if [[ -f "$out" ]]; then say "skip $scan"; continue; fi
  say "QC varian $scan"
  PYTHONPATH=. PYTHONUNBUFFERED=1 "$PY" scripts/qc_varian.py --arch both --scan "$scan" --all --gpu "$GPU"
done
for scan in CE_P1_V_01 CE_P2_V_01 CE_P3_V_01 CE_P4_V_01 CE_P5_V_01; do
  patient="$(echo "$scan" | sed -E 's/CE_(P[0-9]+)_.*/\1/')"
  out="$E3/BothCRB/plots/qc_elekta/$patient/$scan/summary.json"
  if [[ -f "$out" ]]; then say "skip $scan"; continue; fi
  say "QC elekta $scan"
  PYTHONPATH=. PYTHONUNBUFFERED=1 "$PY" scripts/qc_elekta.py --arch both --scan "$scan" --all --gpu "$GPU"
done
say "=== E3 Both QC DONE ==="
# print summaries
"$PY" - <<'PY'
import json
from pathlib import Path
E3 = Path("/home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Experiment3/BothCRB/plots")
for vendor in ("qc_varian", "qc_elekta"):
    print(f"\n== {vendor} ==")
    for p in sorted(E3.glob(f"{vendor}/P*/C*_V_01/summary.json")):
        s = json.loads(p.read_text())
        print(f"{s['scan']}: L1/zero={s['mean_L1_over_zero']:.3f}  cos={s['mean_cos']:.3f}  beat={s['beat_zero_pct']:.0f}%")
PY
