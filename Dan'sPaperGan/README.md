# Dan'sPaperGan — Sang & Ruan (2023) CRB generator experiment

Standalone replica of Sang & Ruan, *A conditional registration network for
continuous 4D respiratory motion synthesis* (Med Phys 2023), Figures 2–3,
adapted for **one reference CT + two phase codes**.

## Deviations from the paper

1. Input is 1 CT + `[t_ref, t_tgt]` (not two images + scalar `t`).
2. CRB scale/shift `(a, b)` are **per-channel** (not scalar).
3. Training uses Elastix-supervised + warp-space PatchGAN (not unsupervised NCC + bending energy).
4. Output is a **direct 3-ch DVF** (as in the paper) — **no** scaling-and-squaring / diffeomorphism.
5. 1×1 residual projection when CRB channel counts change.

## Data

Uses parent SPARE pairs only (not copied here):

```text
../data/spare/all
```

Leave-out phases **5 & 9** (0-indexed 4 & 8), same as the main Voxel_GAN baseline.

## Train

```bash
cd "Dan'sPaperGan"
PYTHONUNBUFFERED=1 python train_crb_gan.py
```

Artifacts: `weights/dans_crb_warp_d_*.pth`, `plots/dans_crb_warp_d.png`.

## QC

```bash
PYTHONPATH=. python scripts/qc_crb_pairs.py
```
