# Experiment 1 — Encoder CRB

CRB on encoder + bottleneck, plain decoder. MSE only. See `../README.md` and `../seed.json`.

```bash
cd PopulationStudy/Experiment1
PYTHONPATH=. PYTHONUNBUFFERED=1 python EncoderCRB/train_crb_enc_mse.py --gpu 0
```

Weights: `weights/crb_enc_mse_pop_e1_generator.pth`
