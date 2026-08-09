# Dan'sPaperGan Diary

## 2026-08-05 — Project scaffolded

Standalone Sang & Ruan CRB U-Net under this folder. Parent Voxel_GAN FiLM+SVF untouched.

## 2026-08-05 ~13:44–14:35 — Train + QC (`dans_crb_warp_d`)

- **Generator:** `UNetCRB` (~1.06M params), direct 3-ch DVF, encoder CRBs only, avg-pool down, no SVF.
- **Train:** warp-D PatchGAN, leave-out phases 5 & 9, dense 16/8, 100 epochs (~49 min).
- **Best val G:** **0.223** @ epoch 90. Late D loss ~**0.047** (alive).
- **TEST (full 128³, all pairs):**

| Set | mean L1 | mean cos |
|-----|---------|----------|
| Directed (90) | **0.199** | **0.968** |
| Leave-out 5/9 (34) | **0.219** | **0.957** |

**vs parent baselines (same full-vol QC style):**

| Model | Directed L1 | Leave-out L1 | Notes |
|-------|-------------|--------------|-------|
| phase-MLP FiLM+SVF | **0.172** | **0.192** | best accuracy |
| FiLM warp-D | 0.189 | 0.210 | living GAN |
| **CRB warp-D (this)** | 0.199 | 0.219 | living GAN; smaller net; no diffeomorphism |

Artifacts: `weights/dans_crb_warp_d_*.pth`, `plots/dans_crb_warp_d.png`, `plots/qc_test_crb/`, archived under `results/dans_crb_warp_d/`.
