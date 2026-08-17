# My v1.0 — FiLM + PatchGAN (original line)

Original Voxel_GAN experiment: phase-conditioned UNet-FiLM generator, PatchGAN, leave-phase-out on SPARE P1, P2 transfer/fine-tune.

Run train scripts from **this folder** (or set `PYTHONPATH=.`) and point data at the repo-root pools:

```bash
cd "My v1.0"
PYTHONPATH=. python train_dvf_gan.py
# data lives at ../data/spare/{train,val,all}
```

Sibling eras at repo root: `Dan2.0/`, `Dan'sPaperGan/`, `PopulationStudy/`.
