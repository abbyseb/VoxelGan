# ClinicalExperiments — E2 retrain status

Updated: 2026-08-26 ~10:54

## E2 cyclic retrain (in progress)
Deleted linear (cond_dim=2) weights + QC. Retraining **encoder + decoder + both** in parallel on **GPU 0**.

| Job | PID | Log |
|-----|-----|-----|
| Encoder | 2833115 | `Experiment2/EncoderCRB/plots/train_crb_enc_mse_cyclic_full_spare.log` |
| Decoder | 2833116 | `Experiment2/DecoderCRB/plots/train_crb_dec_mse_cyclic_full_spare.log` |
| Both | 2833117 | `Experiment2/BothCRB/plots/train_crb_both_mse_cyclic_full_spare.log` |

Confirmed at start: `cond_dim=4`, network from `Experiment2/networks/`.
GPU 0 ~4.2 GiB / 87% util with all three.

## After finish
Re-run E2 Varian + Elekta QC (will auto-detect `cond_dim=4` / cyclic).
