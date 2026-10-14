#!/usr/bin/env bash
# Auto-stop rule for X3R (agreed 14 Oct 2026): once epochs 18-20 are scored, if the front X-ray helps DIR-Lab TRE by less
# than 0.15 mm on average versus no X-ray (zero), stop X3R training and its scorer. Otherwise leave it running.
A=/home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Grid160/TCIA_lite/analysis_2026-10-03
LOG=$A/x3r_stop_check.log
while true; do
  r=$(python3 - <<'PY'
import json, numpy as np
o = json.load(open("/home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Grid160/TCIA_lite/analysis_2026-10-03/xray3r_tre.json"))
E = ["18", "19", "20"]
if not all(e in o and "ap1" in o[e] and "zero" in o[e] for e in E):
    print("WAIT"); raise SystemExit
gain = np.mean([np.mean(o[e]["zero"]["dir_plain"]) - np.mean(o[e]["ap1"]["dir_plain"]) for e in E])
ap = np.mean([np.mean(o[e]["ap1"]["dir_plain"]) for e in E])
print(f"{'STOP' if gain < 0.15 else 'KEEP'} gain_vs_no_xray={gain:.3f}mm ap1_tre={ap:.3f}mm")
PY
)
  case "$r" in
    WAIT*) systemctl --user is-active --quiet xray3r-train || { echo "$(date) X3R no longer running; check ends" >> $LOG; exit 0; }; sleep 300 ;;
    STOP*) echo "$(date) $r -> stopping X3R" >> $LOG; systemctl --user stop xray3r-train xray3r-tre; exit 0 ;;
    *)     echo "$(date) $r -> X3R kept running" >> $LOG; exit 0 ;;
  esac
done
