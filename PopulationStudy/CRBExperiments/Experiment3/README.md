# Experiment 3 — cyclic phase + oracle amp

Same as Experiment 1 (oracle `log(r_p)` + shape targets), but phases are
`[cos 2πθ, sin 2πθ]` for ref and tgt (`cond_dim=5`) instead of linear `θ/9`.

```bash
cd PopulationStudy/CRBExperiments/Experiment3
PYTHONPATH=. PYTHONUNBUFFERED=1 python DecoderCRB/train_crb_dec_amp_mse.py --gpu 1 --epochs 100
```
