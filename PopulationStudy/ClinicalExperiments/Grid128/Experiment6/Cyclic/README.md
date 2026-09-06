# Experiment 6 — Cyclic (128³)

Cyclic phase encoding (`cond_dim=4`), train on **full 128³** volumes.

```bash
cd PopulationStudy/ClinicalExperiments/Experiment6/Cyclic
python scripts/train_mse.py --arch encoder --gpu 0
python scripts/train_mse.py --arch decoder --gpu 1
python scripts/train_mse.py --arch both --gpu 1
```

Weights: `{Encoder,Decoder,Both}CRB/weights/crb_*_mse_cyclic_full128_generator.pth`
