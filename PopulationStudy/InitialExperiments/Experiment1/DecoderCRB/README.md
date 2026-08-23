# Experiment 1 — Decoder CRB

CRB on decoder only. MSE only. See `../README.md` and `../seed.json`.

```bash
cd PopulationStudy/InitialExperiments/Experiment1
PYTHONPATH=. PYTHONUNBUFFERED=1 python DecoderCRB/train_crb_dec_mse.py --gpu 1
```

Weights: `weights/crb_dec_mse_pop_e1_generator.pth`
