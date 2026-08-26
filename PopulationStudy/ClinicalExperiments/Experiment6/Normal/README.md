# Experiment 6 — Normal (linear, 128³)

Linear phase encoding (`cond_dim=2`), train on **full 128³** volumes (not 64³ patches).

- Data: symlink → `Experiment1/data`
- Nets: InitialExperiments linear CRB (same as E1)
- `patches_per_pair=1` (volume already 128³ → crop is the full volume)

## Status

- [ ] Train Encoder / Decoder / Both (100 epochs)
- [ ] QC Varian + Elekta (after train; prefer µ air/water like E5)

## Logs

`{Encoder,Decoder,Both}CRB/plots/train_crb_*_mse_linear_full128.log`
