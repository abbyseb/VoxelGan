# Experiment 2 — A1 normalize-only

Shape targets `û = u/A_p`, but **no patient amplitude in the condition** (`log(r̂)=0`).
At QC, scale predictions by **train-pool mean** `Ā`, not patient `A_p`.

Same split/grid as Experiment 1. Isolates shape-space loss from amp conditioning.

```bash
cd PopulationStudy/CRBExperiments/Experiment2
PYTHONPATH=. PYTHONUNBUFFERED=1 python DecoderCRB/train_crb_dec_amp_mse.py --gpu 0 --epochs 100
```
