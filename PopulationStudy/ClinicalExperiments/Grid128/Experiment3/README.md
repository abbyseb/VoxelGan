# ClinicalExperiments — Experiment 3

**Full SPARE train → zero-shot Clinical Varian + Elekta**, with **source-side FOV / CBCT augmentation**.

- Phase: **cyclic** (`cond_dim=4`), same as Experiment 2  
- Primary arch: **BothCRB** (Encoder/Decoder scripts available too)  
- Data: symlinked from Experiment1 pooled + clinical packs  

## Train augment (per sample, equal ¼)

| Mode | Prob |
|------|------|
| Normal | ¼ |
| Half-FOV (random L/R + cut jitter) | ¼ |
| CBCT noise (full FOV) | ¼ |
| Half-FOV + CBCT noise | ¼ |

Val = **Normal only**. Loss = lung-masked MSE on visible FOV.

## Status

- [x] Scaffold + cyclic nets + FOV aug dataset  
- [ ] Train Both *(queued — waits for free GPU)*  
- [ ] Zero-shot QC Varian / Elekta  

## Commands

```bash
cd PopulationStudy/ClinicalExperiments/Experiment3
PYTHONPATH=. PYTHONUNBUFFERED=1 python scripts/train_mse.py --arch both --gpu 0 --epochs 100
# log: BothCRB/plots/train_crb_both_mse_cyclic_fov_aug.log

PYTHONPATH=. python scripts/qc_varian.py --arch both --scan CV_P1_V_01 --all --gpu 1
PYTHONPATH=. python scripts/qc_elekta.py --arch both --scan CE_P1_V_01 --all --gpu 1
```

Demo augs: [`../plots/fov_aug_demo/`](../plots/fov_aug_demo/)
