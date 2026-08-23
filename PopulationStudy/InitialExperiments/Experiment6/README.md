# Experiment 6 — A0 baseline on common iso grid

Decoder-CRB **without** amplitude conditioning, on the P0-A regrid. Same patient split as Experiment 1 / CRBExperiments E1 so you can compare baseline vs oracle-amp side by side.

| | |
|--|--|
| **Grid** | `PopulationStudy/data_iso/` — **2.0 mm** isotropic, **160³** |
| **Split** | Train P1,P2,P6–P9 · hold-out **P3,P4,P5** |
| **Arch** | Decoder-CRB only (`cond = [θ_ref, θ_tgt]`) |
| **Target** | Raw DVF in **iso-voxels** (1 vx = 2 mm) |
| **Arm** | AmplitudeConditioning **A0** |

```bash
cd PopulationStudy/InitialExperiments/Experiment6
python scripts/build_pooled_dataset.py
PYTHONPATH=. PYTHONUNBUFFERED=1 python DecoderCRB/train_crb_dec_mse.py --gpu 1 --epochs 100
```

Plot/log: `DecoderCRB/plots/crb_dec_mse_pop_e6_iso.png` · `train_crb_dec_mse.log`

Companion (oracle amp on same grid/split): [`../../CRBExperiments/Experiment1/`](../../CRBExperiments/Experiment1/).
