# Experiment 6 — Normal (linear, 128³)

Linear phase encoding (`cond_dim=2`), train on **full 128³** volumes (not 64³ patches).

- Data: symlink → `Experiment1/data`
- Nets: InitialExperiments linear CRB (same as E1)
- `patches_per_pair=1` (volume already 128³ → crop is the full volume)

## Status

- [x] Train Encoder / Decoder / Both (100 epochs)
- [x] QC Varian + Elekta (minmax + µ air/water; metrics under `{Arch}CRB/plots/qc_{varian,elekta}/{minmax,mu}/`)

## Clinical QC vs E1 (directed pairs, lung mask)

| Arch | Norm | Varian cos | Elekta cos | vs E1 Both µ |
|------|------|------------|------------|--------------|
| Both | minmax | 0.547 | 0.344 | Elekta +0.21 (large) |
| Both | µ | 0.584 | 0.478 | ~parity (+0.02 / +0.00) |
| Decoder | µ | 0.588 | 0.474 | Varian +0.02 |

**Takeaway:** E6 val MSE is worse on SPARE, but **clinical zero-shot gain** is real — especially Elekta under minmax, and Varian under µ (e.g. CV_P3 cos 0.671 vs E1 µ 0.623).

Summary: `plots/qc_summary/summary_*_minmax_mu.json`, comparison `plots/qc_summary/compare_e1_e6.tsv`.

## Logs

`{Encoder,Decoder,Both}CRB/plots/train_crb_*_mse_linear_full128.log`
