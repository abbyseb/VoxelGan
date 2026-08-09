# Voxel-GAN Results (to date)

Patient-specific, phase-conditioned DVF synthesis. Supervised by lung-masked B-spline Elastix. Eval = leave-phase-out / transfer — **not** inter-patient generalization unless noted.

**L1** = lung-masked mean absolute DVF error vs Elastix (3 channels). Units: 128³ grid with forced spacing `(1,1,1)` → labeled **mm** (`1 voxel = 1 mm` after crop/resample). Lower is better.  
**cos** = lung-masked cosine similarity of predicted vs Elastix DVF. Higher is better.  
Identity pairs excluded from directed means unless noted.

QC view (all panels): View 1, axial slice 52 ⊥ LR, H→SI V→AP (`configs/dvf_view_config.json`).

---

## 1. Models trained

| ID | Architecture | Disc | Train data | Hold-out | Epochs | Weights |
|----|--------------|------|------------|----------|--------|---------|
| **phase-MLP** | UNetFiLM + SVF (`int_steps=6`), continuous phase MLP | 4-ch `[ref, DVF]` | P1 leave-out | phases 5 & 9 | 100 | `weights/spare_mc_p1_dvf_gan_phase_mlp_*.pth` |
| **warp-D** | same FiLM+SVF | 2-ch `[warp(ref,DVF), target]` | P1 leave-out | 5 & 9 | 100 | `weights/spare_mc_p1_scenario_warp_d_*.pth` |
| **full-P1 warp-D** | same FiLM+SVF | warp-D | **all** P1 pairs | none | 100 | `weights/spare_mc_p1_full_warp_d_*.pth` |
| **LOOCV warp-D** | same FiLM+SVF | warp-D | P1, 10 folds | one phase each | 100×10 | `weights/spare_mc_p1_loocv_warp_d_holdXX_*.pth` |
| **Dan CRB** | UNetCRB, direct 3-ch DVF (no SVF) | warp-D | P1 leave-out | 5 & 9 | 100 | `Dan'sPaperGan/weights/dans_crb_warp_d_*.pth` |
| **full-P1 Dan** | UNetCRB | warp-D | **all** P1 pairs | none | 100 | `Dan'sPaperGan/weights/dans_crb_p1_full_warp_d_*.pth` |
| **LOOCV Dan** | UNetCRB | warp-D | P1, 10 folds | one phase each | 100×10 | `Dan'sPaperGan/weights/dans_crb_loocv_warp_d_holdXX_*.pth` |
| **P2 FT Our Warp** | FiLM+SVF, init = full-P1 warp-D | warp-D | P2 leave-out split | 5 & 9 val | 30 | `weights/spare_mc_p2_warp_d_finetune_*.pth` |
| **P2 FT Dan** | UNetCRB, init = full-P1 Dan | warp-D | P2 leave-out split | 5 & 9 val | 30 | `Dan'sPaperGan/weights/dans_crb_p2_warp_d_finetune_*.pth` |

**Scenarios (ablation, P1 leave-out 5/9; no full TEST table here):** weak_d, more_g, higher_adv — see `Diary.md` / `results/scenario_*`. Takeaway: 4-ch DVF-space D dies; warp-D keeps D alive (`λ_adv=0.05`).

**Data**
- **P1:** SPARE MC `MC_T_P1_NS` → `data/spare/{train,val,all}/` (128³, 100 directed pairs).
- **P2:** SPARE MC `MC_T_P2_SC` (Training GT has SC only; no NS) → `P2/data/{train,val,all}/`.

---

## 2. Patient 1 — same-patient leave-phase-out (phases 5 & 9)

Trained with held-out 5 & 9; full-volume QC on all 100 pairs. Directed = 90 non-identity pairs. Leave-out = pairs touching phase 5 or 9.

| Model | Directed L1 ↓ | Directed cos ↑ | Leave-out 5/9 L1 ↓ | Leave-out cos ↑ | D alive? |
|-------|---------------|----------------|--------------------|-----------------|----------|
| phase-MLP (FiLM+SVF) | **0.172** | **0.975** | **0.192** | ~0.97 | No (~0) |
| warp-D (FiLM+SVF) | 0.189 | 0.971 | 0.210 | ~0.97 | Yes (~0.06) |
| Dan CRB warp-D | 0.199 | 0.968 | 0.219 | ~0.96 | Yes (~0.05) |

**Read:** Best supervised accuracy = phase-MLP. Warp-D trades a little L1 for a living GAN. Dan CRB is smaller / no diffeomorphism; slightly worse L1 than FiLM warp-D.

**QC folders**
- phase-MLP: `plots/qc_test_phase_mlp/` (+ `metrics.tsv`)
- warp-D: `plots/qc_test_warp_d/`
- Dan CRB: `Dan'sPaperGan/plots/qc_test_crb/`

---

## 3. Patient 1 — leave-one-phase-out LOOCV (train val G)

Best **validation G loss** per fold (supervised terms; not full-volume DVF L1). Lower is better.

### 3.1 Our Warp (FiLM warp-D)

| Hold-out phase | Best val G |
|----------------|------------|
| 1 | 0.386 |
| 2 | 0.227 |
| 3 | 0.171 |
| 4 | **0.146** |
| 5 | 0.186 |
| 6 | 0.211 |
| 7 | 0.261 |
| 8 | 0.215 |
| 9 | 0.179 |
| 10 | 0.289 |
| **Mean ± std** | **0.227 ± 0.066** |

Log: `plots/loocv_warp_d/summary.log`. Curves: `plots/spare_mc_p1_loocv_warp_d_holdXX.png`.  
*(Aggregate full-vol TEST L1 across folds not yet tabulated.)*

### 3.2 Dan CRB warp-D

| Hold-out phase | Best val G |
|----------------|------------|
| 1 | 0.371 |
| 2 | 0.264 |
| 3 | 0.219 |
| 4 | 0.215 |
| 5 | **0.192** |
| 6 | 0.261 |
| 7 | 0.336 |
| 8 | 0.263 |
| 9 | 0.239 |
| 10 | 0.301 |
| **Mean ± std** | **0.266 ± 0.053** |

Log: `Dan'sPaperGan/plots/loocv_crb/summary.log`.

**Read:** Our Warp LOOCV val G is lower on average than Dan CRB (0.227 vs 0.266). Hardest folds tend to be extreme phases (1, 7, 10).

---

## 4. Patient 2 — zero-shot vs fine-tune (from **full-P1**, no leave-out)

Fair transfer: init from models trained on **all** P1 pairs (`spare_mc_p1_full_warp_d`, `dans_crb_p1_full_warp_d`), then optional P2 fine-tune (30 epochs, lr_g=3e-5). QC overwrites `P2/{our_warp,our_warp_ft,dans_gan,dans_gan_ft}/`.

| Model | Setting | Directed L1 ↓ | Directed cos ↑ | Leave-out 5/9 L1 ↓ |
|-------|---------|---------------|----------------|--------------------|
| Our Warp | P1→P2 zero-shot | 0.614 | 0.547 | 0.612 |
| Our Warp | P2 fine-tuned | **0.132** | **0.964** | **0.144** |
| Dan CRB | P1→P2 zero-shot | 0.740 | 0.428 | 0.730 |
| Dan CRB | P2 fine-tuned | 0.134 | 0.958 | 0.157 |

**vs P1 same-patient leave-out (context)**

| | P1 leave-out L1 | P2 zero-shot L1 | P2 tuned L1 |
|--|-----------------|-----------------|-------------|
| Our Warp | 0.210 (leave-out train) / ~0.19 directed | 0.614 | **0.132** |
| Dan CRB | 0.219 leave-out | 0.740 | **0.134** |

**Read**
- Zero-shot P1→P2 fails (~0.6–0.7 L1): anatomy/motion do not transfer.
- Patient-specific fine-tune recovers strongly (~0.13 L1), matching or beating P1 leave-out accuracy on this metric.
- After tune, Our Warp slightly edges Dan (0.132 vs 0.134 directed).

**QC**
- Zero-shot: `P2/our_warp/qc/`, `P2/dans_gan/qc/`
- Tuned: `P2/our_warp_ft/qc/`, `P2/dans_gan_ft/qc/`
- Pipeline: `P2/run_full_p1_transfer.sh`, logs `P2/logs/`

---

## 5. Head-to-head summary

### Same patient (P1)

1. **Accuracy:** phase-MLP ≥ warp-D ≥ Dan CRB.  
2. **Adversary health:** only warp-space D stays alive at `λ_adv=0.05`.  
3. **LOOCV val G:** Our Warp better mean than Dan CRB.

### Across patients (P1 → P2)

1. **Zero-shot does not work** for either model.  
2. **Fine-tune on P2 Elastix** is required and works for both.  
3. Tuned P2 L1 ≈ **0.13** for both; Our Warp marginally better.

### Design choices that stuck

- Continuous phase MLP (discrete embeddings failed leave-out).  
- FiLM at bottleneck + decoder; SVF + scaling-and-squaring for Our Warp.  
- Warp-D PatchGAN over raw DVF-space D.  
- Full-P1 pretrain before P2 tune (not leave-out 5/9 weights).

---

## 6. Artifact map

| What | Path |
|------|------|
| This summary | `results.md` |
| Diary (narrative) | `Diary.md`, `Dan'sPaperGan/Diary.md` |
| P1 QC metrics | `plots/qc_test_*/metrics.tsv` |
| P2 metrics | `P2/*/metrics.tsv` |
| Our Warp LOOCV | `plots/loocv_warp_d/` |
| Dan LOOCV | `Dan'sPaperGan/plots/loocv_crb/` |
| Weights (local, gitignored) | `weights/`, `Dan'sPaperGan/weights/` |
| Prepared volumes (gitignored) | `data/spare/`, `P2/data/` |

---

## 7. Still open

- Aggregate **full-volume TEST L1** for each LOOCV fold (both models).  
- Optional: longer P2 fine-tune / LR sweep; P2 NS if GT becomes available.  
- DIR-Lab / multi-patient only if claim expands beyond patient-specific.
