#!/usr/bin/env bash
set -euo pipefail
CE="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python}"
E2="$CE/Experiment2"
GPU="${GPU:-0}"
cd "$E2"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*"; }

say "=== E2 QC START gpu=$GPU (enc/dec/both × varian+elekta) ==="
for arch in encoder decoder both; do
  case "$arch" in
    encoder) adir=EncoderCRB ;;
    decoder) adir=DecoderCRB ;;
    both) adir=BothCRB ;;
  esac
  for scan in CV_P1_V_01 CV_P2_V_01 CV_P3_V_01 CV_P4_V_01 CV_P5_V_01; do
    patient="$(echo "$scan" | sed -E 's/CV_(P[0-9]+)_.*/\1/')"
    out="$E2/$adir/plots/qc_varian/$patient/$scan/summary.json"
    if [[ -f "$out" ]]; then say "skip $arch $scan"; continue; fi
    say "QC varian $arch $scan"
    PYTHONPATH=. PYTHONUNBUFFERED=1 "$PY" scripts/qc_varian.py --arch "$arch" --scan "$scan" --all --gpu "$GPU"
  done
  for scan in CE_P1_V_01 CE_P2_V_01 CE_P3_V_01 CE_P4_V_01 CE_P5_V_01; do
    patient="$(echo "$scan" | sed -E 's/CE_(P[0-9]+)_.*/\1/')"
    out="$E2/$adir/plots/qc_elekta/$patient/$scan/summary.json"
    if [[ -f "$out" ]]; then say "skip $arch $scan"; continue; fi
    say "QC elekta $arch $scan"
    PYTHONPATH=. PYTHONUNBUFFERED=1 "$PY" scripts/qc_elekta.py --arch "$arch" --scan "$scan" --all --gpu "$GPU"
  done
done
say "=== E2 QC DONE ==="
"$PY" - <<'PY'
import json
from pathlib import Path
E2 = Path("/home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Experiment2")
for arch in ("Encoder", "Decoder", "Both"):
  for vendor in ("qc_varian", "qc_elekta"):
    paths = sorted((E2 / f"{arch}CRB/plots" / vendor).glob("P*/C*_V_01/summary.json"))
    print(f"\n== {arch} {vendor} ({len(paths)}) ==")
    for p in paths:
      s = json.loads(p.read_text())
      print(f"  {s['scan']}: L1/zero={s['mean_L1_over_zero']:.3f} cos={s['mean_cos']:.3f} beat={s['beat_zero_pct']:.0f}% phase={s.get('phase','?')}")
PY
