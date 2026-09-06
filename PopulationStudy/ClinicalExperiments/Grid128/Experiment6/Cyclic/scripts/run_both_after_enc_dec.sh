#!/bin/bash
# Queue Both training after encoder+decoder finish.
set -euo pipefail
CYCLIC=/home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Experiment6/Cyclic
cd "$CYCLIC"
while pgrep -f 'scripts/train_mse.py --arch encoder --gpu 0' >/dev/null; do sleep 60; done
while pgrep -f 'scripts/train_mse.py --arch decoder --gpu 1' >/dev/null; do sleep 60; done
exec python3 scripts/train_mse.py --arch both --gpu 1 2>&1 | tee BothCRB/plots/train_crb_both_mse_cyclic_full128.log
