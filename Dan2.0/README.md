# Dan 2.0 — UNetCRB without discriminator (MSE-only)

Ablation of [`Dan'sPaperGan`](../Dan'sPaperGan/): **same UNetCRB** architecture, **no PatchGAN**, train objective = **lung-masked MSE vs Elastix DVF only**.

## vs Dan'sPaperGan

| | Dan CRB (GAN) | Dan 2.0 |
|--|---------------|---------|
| Generator | UNetCRB | same |
| Discriminator | warp-D PatchGAN | **none** |
| Loss | L1 + 0.5 NCC + 0.1 smooth + 0.05 adv | **MSE only** |
| Hold-out | phases 5 & 9 | same |
| Epochs / lr | 100 / 1e-4 | same |

QC still reports **L1 + cos** (same as Dan) for fair tables; training uses MSE.

## Data

Uses parent SPARE pairs (not copied here):

```text
../data/spare/all
```

## Train

```bash
cd Dan2.0
PYTHONUNBUFFERED=1 python train_crb_mse.py
```

Artifacts: `weights/dans_crb_mse_generator.pth`, `plots/dans_crb_mse.png`.  
(`weights/` and `*.pth` are gitignored by the parent repo.)

## QC

```bash
cd Dan2.0
PYTHONPATH=. python scripts/qc_crb_pairs.py
```

Writes `plots/qc_test_mse/` (PNG per pair + `metrics.tsv`).
