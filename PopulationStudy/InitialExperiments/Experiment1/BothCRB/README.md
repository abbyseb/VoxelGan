# Experiment 1 — Both CRB

CRB on encoder + bottleneck + decoder. MSE only. See `../README.md` and `../seed.json`.

```bash
cd PopulationStudy/InitialExperiments/Experiment1
PYTHONPATH=. PYTHONUNBUFFERED=1 python BothCRB/train_crb_both_mse.py --gpu 0
```

Weights: `weights/crb_both_mse_pop_e1_generator.pth`
