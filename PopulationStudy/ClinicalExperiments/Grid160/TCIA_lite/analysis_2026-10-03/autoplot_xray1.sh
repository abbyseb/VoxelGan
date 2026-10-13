#!/usr/bin/env bash
# Redraw the X1 graphs whenever the training history or TRE results change; stop when training and scoring are done.
P=/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python3
A=/home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Grid160/TCIA_lite/analysis_2026-10-03
H=$A/../run_A2_full160_aug_xray1/logs/history.json
cd $A; last=""
while true; do
  now="$(stat -c %Y $H xray1_tre.json 2>/dev/null | tr '\n' ' ')"
  if [ "$now" != "$last" ]; then
    $P plot_xray1.py xray1_progress.png && $P plot_xray1_trainval.py xray1_train_val.png > /dev/null && last="$now"
  fi
  systemctl --user is-active --quiet xray1-train || systemctl --user is-active --quiet xray1-tre || { $P plot_xray1.py xray1_progress.png; $P plot_xray1_trainval.py xray1_train_val.png > /dev/null; exit 0; }
  sleep 60
done
