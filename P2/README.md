# SPARE MC Patient 2 — Elastix + frozen-model QC

No training. P1-trained generators are tested on P2 anatomy.

## Layout

```
P2/
  raw/MC_T_P2_SC → Data/.../MC_T_P2_SC   # SPARE GT (Training has SC only; no NS)
  data/{train,val,all}/                 # 128³ CTs + masked B-spline DVFs
  our_warp/qc/                          # FiLM warp-D (P1 weights)
  dans_gan/qc/                          # Dan UNetCRB warp-D (P1 weights)
  elastix_prepare.log
```

## Source

- GroundTruth: `MonteCarloDatasets/Training/P2/MC_T_P2_SC` from `SPARE_GroundTruth.7z`
- Same prep as P1: lung-crop → 128³, all 100 directed pairs (held-out split 5 & 9 for bookkeeping only)

## Models

Fair transfer uses **full-P1** weights (trained on all 100 pairs, no leave-out):

| Subfolder | Meaning | Checkpoint |
|-----------|---------|------------|
| `our_warp` | Full-P1 → P2 zero-shot | `weights/spare_mc_p1_full_warp_d_generator.pth` |
| `our_warp_ft` | Full-P1 → P2 fine-tune | `weights/spare_mc_p2_warp_d_finetune_generator.pth` |
| `dans_gan` | Full-P1 CRB → P2 zero-shot | `Dan'sPaperGan/weights/dans_crb_p1_full_warp_d_generator.pth` |
| `dans_gan_ft` | Full-P1 CRB → P2 fine-tune | `Dan'sPaperGan/weights/dans_crb_p2_warp_d_finetune_generator.pth` |

Pipeline: `P2/run_full_p1_transfer.sh` (GPU0 transfer; GPU1 Dan LOOCV).
Logs: `P2/logs/`.
