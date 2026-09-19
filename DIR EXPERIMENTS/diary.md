# DIR EXPERIMENTS — Diary

Journal of the VoxelMap TRE line on DIR-Lab.  
Primary KPI: **75-point** Sampled4D TRE (T00→T50), mm. Not Elastix L1/cos.

---

## 2026-09-07 — Scaffold & arm design

### Goal
Stand up a clean experiment line: **VoxelMap localization TRE on DIR-Lab**, separate from `PopulationStudy/DIR-Experiments` (CRB DVF / E6).

### What we did
- Created `DIR EXPERIMENTS/` with packs under `data/dirlab_packs/`, TRE harness `scripts/dirlab_tre.py`.
- Arms sketched:
  - **A0** identity (lower bound)
  - **A1** oracle VoxelMap on patient’s own DIR 4D-CT
  - **A2** generic SPARE prior
  - **A3** synth-conditioned VoxelMap (**main**)
  - **A4/A5** optional mismatch / CBCT FT

### A0 done
- Identity TRE check vs published 300-pt means: **PASS**.
- Cohort identity (later locked to 75-pt): ~**8.69 mm**; Case 1 ~**3.91 mm**.
- Rule of thumb: beat per-case identity; lower is better.

### Design wobble (reverted same day)
- **Hypothesis:** A2 should be “DIR population LOO VoxelMap.”
- Renamed arms (SPARE → A6).
- User: go back; **drop population arm for now**.
- Restored: A2 = SPARE generic, A3 = synth main, no A6.

### Hard rule locked
- **Always 75-pt** for arm scores (`--set 75`).
- 300-pt only for pack QA (`--check` / `--allow-300`).
- Updated READMEs, A0 `summary.json` / TSV, `dirlab_tre.py` enforcement.

### DRR geometry
- **Finding:** DIR-Lab ships **no** official `Geometry.xml`.
- Online / SPARE norm: Varian half-fan SID 1000 / SDD 1500 / OffsetX 148 / 680 views.
- User placed `arms/A1_oracle_dirlab/Geometry.xml` (same family; OffsetY `+2` vs SPARE gold `-2`).
- **Decision:** freeze that file as DIR DRR standard for A1+.

### Git
- Pushed scaffold to `origin/VoxelMap_Experiments` (after auth via existing PAT).

---

## 2026-09-07 (afternoon) — A1 Case 1 smoke

### Hypothesis
Mirror clinical **elastix** VoxelMap path on DIR Case 1: real phases → DRR → Elastix DVF → NoFiLM train → TRE.

### Pipeline built
- `scripts/prepare_a1_case.py`
  - Stage from `PopulationStudy/DIR-Experiments/data/P1_DIR` (GTVol_01=T00 … GTVol_06=T50).
  - Frozen `Geometry.xml` + MC/Varian DRR opts.
  - Synthetic `RespBin.csv` (680 views).
  - Elastix fixed = `sub_CT_06` (T50).
- Scan id: `DIR_C01`.

### Smoke results
| Step | Result |
|------|--------|
| Prepare | ~2.8 min, 6800 DRR bins, 9 DVFs |
| Train 2 ep | best val **0.0161** |
| Full 50 ep (started eve) | finished overnight |

### Open issue noted
- TRE bridge (grid / SI-flip / DVF units / sign) not trusted yet.
- Quick Elastix TRE attempt ≈ identity → deferred to next day.

---

## 2026-09-08 — 75-pt TRE bridge

### Hypothesis
Official landmarks live on Emory packs; `P1_DIR` volumes are **SI-flipped**. LEARN `sub_CT` is 128³ with **spacing forced to 1** (unit voxels). prep_train stores DVF as `(H,W,D,3)` not `(z,y,x,3)`.

### What we proved
1. Official ↔ pack: `z_p = (nz−1) − z` (matches `P1_DIR` landmark files ≈ flip of official).
2. Pack ↔ sub: `s = n × 128 / N`.
3. npy layout: transpose `(2,0,1)` → sitk `(z,y,x)` for sampling.
4. Script: `scripts/eval_a1_tre.py`.

### First TRE numbers (HU-**clipped** sub_CT — LEARN default)

| Field | TRE 75-pt T00→T50 | vs identity 3.913 |
|-------|-------------------|-------------------|
| Identity | 3.913 | — |
| Elastix DVF | 3.611 | −0.30 |
| VoxelMap `best.pt` | 3.633 | −0.28 |

Bridge **verified** (both beat identity; Elastix ≈ VoxelMap, L1 ~0.04).  
But gains were tiny → next hypothesis.

---

## 2026-09-08 — HU clip kills lung contrast

### Hypothesis
LEARN `downsample.py` does `arr[arr < 0] = 0`. DIR is HU (air ≈ −1024). Clipping erases lung air contrast → Elastix sees almost no features → tiny DVF → TRE≈identity.

### Proof (images)
![HU kept vs clipped — coronal/axial + |diff|](arms/A1_oracle_dirlab/runs/DIR_C01/plots/hu_kept_vs_clipped_sub_CT_01.png)

![Air window + histogram](arms/A1_oracle_dirlab/runs/DIR_C01/plots/hu_kept_vs_clipped_lung_window.png)

- ~**74%** of `sub_CT_01` voxels were HU &lt; 0 before clip.
- Air-window view of clipped volume is empty; histogram piles everything at 0.

### What we changed
- `prepare_a1_case.py`: **HU-preserving** downsample (no negative clip); `--redo-hu` keeps DRRs, rebuilds sub_CT + Elastix + prep.
- Archived clipped run as `*_hu_clip_bak/`.

### Elastix TRE after fix (same Case 1, same bridge)

| Labels | TRE | vs identity |
|--------|-----|-------------|
| HU-clipped (old) | 3.611 | −0.30 |
| **HU kept (new)** | **1.848** | **−2.07** |

Mean |DVF| rose ~0.16 → ~0.50 sub-voxels (real SI motion).

### Sign convention correction
- With real HU fields, ITK-consistent mapping is:
  - T00→T50: **−disp** @ T00  
  - T50→T00: **+disp** @ T50  
- Clipped-era “locked” opposite signs were an artifact of near-zero fields. Updated `eval_a1_tre.py`.

### VoxelMap retrain
- Full **50 epochs** on HU-kept labels started (~09:45), GPU 1.  
- Log: `runs/DIR_C01/logs/train_nofilm_full_hu.log`  
- VoxelMap TRE on new labels: **pending** train finish.

---

## 2026-09-08 — Case 8 HU check (next patient)

Largest identity TRE case (16.0 mm). `P8_DIR` present: **512×512×128**, spacing 0.97/0.97/2.5, SI-flipped, intensity_offset −1024.

| Volume | min | max | frac HU&lt;0 | frac &lt;−900 |
|--------|-----|-----|-------------|---------------|
| Case 8 GTVol_01 | −1024 | 3071 | **93%** | 59% |
| Case 8 GTVol_06 | −1024 | 3071 | **93%** | 58% |
| Case 1 GTVol_01 (ref) | −1024 | 13616 | 74% | — |

**Finding:** Case 8 is *more* air-dominated than Case 1. LEARN clip would wipe almost the entire lung FOV. **Must use keep-HU downsample** from the start.

![Case 8 HU kept vs clip](arms/A1_oracle_dirlab/plots/case8_hu_kept_vs_clipped.png)

---

## 2026-09-08 ~12:41 — Case 8 prepare started

- `prepare_a1_case.py --case 8 --gpu 0` (keep-HU), scan `DIR_C08`.
- Case 1 HU train still on GPU 1 (~26+/50 earlier; check log).
- Log: `arms/A1_oracle_dirlab/logs/prepare_case8.stdout`

---

## 2026-09-08 ~14:37 — Case 1 HU VoxelMap TRE + Case 8 train

### Case 1 (HU-kept) 75-pt T00→T50
| Field | TRE | vs identity 3.913 |
|-------|-----|-------------------|
| Identity | 3.913 | — |
| Elastix | **1.848** | −2.07 |
| VoxelMap (50 ep, best@45) | **2.024** | −1.89 |

VoxelMap close to Elastix oracle; both far better than clipped-era ~3.6 mm.

### Case 8
- Prepare done earlier; LowRes train started then **stopped** (labels were bad — see next entry).

---

## 2026-09-08 ~15:25 — Case 8 Elastix: LowRes failed, masked fine-grid fixed it

### Why LowRes ≈ identity
LEARN `Elastix_BSpline_Sliding_LowRes` (grid **32**, 2000 iters, **no lung mask**) on unit-spaced 128³ underfits Case 8’s ~14 mm SI motion:
- Elastix SI at landmarks ~**2.2 mm** (needed ~14.3); field `|u|` mean **0.78**
- 75-pt T00→T50 TRE **15.90** vs identity **16.00** (Δ +0.11)

Not a TRE/sign bug (same bridge as Case 1).

### Fix
New param `arms/A1_oracle_dirlab/Elastix_BSpline_DIR_masked.txt` (clinical-style: grid **16**, 4 res, 4000 iters, **4096** samples) + downsampled `Mask_Lung` in `prepare_a1_case.run_dvf`.

| Elastix | TRE T00→T50 | vs identity 16.00 | SI absmean @ lm |
|---------|-------------|-------------------|-----------------|
| LowRes unmasked | 15.90 | −0.11 | 2.2 mm |
| **Masked grid16** | **3.76** | **−12.24** | **15.0 mm** (corr_z 0.97) |

T50→T00 QA: **2.12 mm**. LowRes artifacts archived as `*_lowres_bak/`. ModelTraining re-prepped; VoxelMap train **not** restarted yet.

---

## 2026-09-08 ~15:29 — Ablation: mask vs params (Case 8, 75-pt T00→T50)

2×2 on `sub_CT_06←01` only. Same HU-kept 128³. Artifacts: `runs/DIR_C08/ablation_elastix/`.

| | **unmasked** | **+ lung mask** |
|--|--------------|-----------------|
| **LowRes** (grid32, 2k iter) | 15.90 | **3.69** |
| **Fine** (grid16, 4k, 4 res) | 8.92 | **3.76** |

**Verdict:** mask is the dominant fix (15.9→3.7 even on LowRes). Finer grid without mask helps (→8.9) but still ~5 mm worse than masked. Fine≈LowRes once masked (stochastic tie). Not a TRE cheat — mask is teacher-only; keep it for A1 oracle labels.

---

## 2026-09-08 ~15:37 — Case 1 check: does masked Elastix beat 1.85?

Probe only (`DVF_01`, no retrain yet). Artifacts: `runs/DIR_C01/ablation_elastix/`.

| Method | TRE T00→T50 |
|--------|-------------|
| Current prod (LowRes unmasked, HU-kept) | **1.848** |
| Fine unmasked | 1.561 |
| LowRes + mask | 1.253 |
| **Fine + mask** | **1.229** (−0.62 vs current) |

**Yes — clearly better than 1.85.** Worth re-Elastix + retrain Case 1 when ready (same recipe as Case 8).

---

## 2026-09-08 ~15:47 — Retrain Case 1 + Case 8 (masked Elastix labels)

| Case | GPU | Elastix oracle (75-pt) | Train log |
|------|-----|------------------------|-----------|
| **1** | **0** | labels TRE **1.24** (was 1.85) | `runs/DIR_C01/logs/train_nofilm_full_masked.log` |
| **8** | **1** | labels TRE **3.76** (was 15.90) | `runs/DIR_C08/logs/train_nofilm_full_masked.log` |

Both: 50 ep NoFiLM concat, bs=8, lr=1e-5. C1 old unmasked archived `*_unmasked_bak/`; C8 lowres `*_lowres_bak/`.

---

## 2026-09-09 — Masked-label VoxelMap TRE (trains done)

| Case | Set | Identity | Elastix | VoxelMap |
|------|-----|----------|---------|----------|
| **1** | 75 | 3.913 | **1.242** | **1.327** |
| **1** | 300 | 3.892 | **1.121** | **1.180** |
| **8** | 75 | 16.000 | **3.759** | **4.942** |
| **8** | 300 | 14.995 | **3.310** | **4.493** |

(C1/C8 VoxelMap ≈ stride-10; 300 from same fields.) Both beat identity; C1 near oracle; C8 ~1.2 mm behind Elastix on 75.

**Policy:** report **75 + 300** from now on (`eval_a1_tre.py`).

---

## 2026-09-09 ~09:04 — A1 Cases 2 & 3 launched

Locked recipe: keep-HU downsample → DRR → **masked** Elastix → prep → 50-ep NoFiLM → dual TRE.

| Case | GPU | Pipeline log |
|------|-----|--------------|
| **2** | **0** | `arms/A1_oracle_dirlab/logs/pipeline_case2_gpu0.log` |
| **3** | **1** | `arms/A1_oracle_dirlab/logs/pipeline_case3_gpu1.log` |

Prepare started (staged + HU downsample). Auto-continues to train + `eval_a1_tre.py`.

### Results (both finished ~13:38)

| Case | Set | Identity | Elastix | VoxelMap |
|------|-----|----------|---------|----------|
| **2** | 75 | 4.65 | **1.09** | **1.38** |
| **2** | 300 | 4.34 | **1.06** | **1.32** |
| **3** | 75 | 7.25 | **1.30** | **1.86** |
| **3** | 300 | 6.94 | **1.30** | **1.80** |

---

## 2026-09-09 ~16:14 — A1 queue: C6→C7 on GPU0, C4→C5 on GPU1

Sequential per GPU (one train at a time). Same locked recipe.

| GPU | Order | Log |
|-----|-------|-----|
| **0** | **6 → 7** | `logs/pipeline_gpu0_cases6_7.log` |
| **1** | **4 → 5** | `logs/pipeline_gpu1_cases4_5.log` |

After these: cases **9–10** left (then full cohort table).

### Results (queues finished ~2026-09-10 01:13)

| Case | Set | Identity | Elastix | VoxelMap |
|------|-----|----------|---------|----------|
| **4** | 75 | 9.69 | **1.89** | **3.10** |
| **4** | 300 | — | **1.77** | **3.18** |
| **5** | 75 | 7.41 | **2.08** | **2.32** |
| **5** | 300 | — | **1.91** | **2.22** |
| **6** | 75 | 11.77 | **2.93** | **3.83** |
| **6** | 300 | — | **2.60** | **3.72** |
| **7** | 75 | 10.71 | **2.25** | **5.03** |
| **7** | 300 | — | **2.34** | **5.30** |

C7 is the softest so far (VoxelMap ~2× Elastix on 75). C5 is closest to teacher.

---

## 2026-09-10 ~10:14 — A1 Cases 9 & 10 launched

Same locked recipe. GPUs were idle after C4–C7.

| Case | GPU | Pipeline log |
|------|-----|--------------|
| **9** | **0** | `logs/pipeline_gpu0_case9.log` |
| **10** | **1** | `logs/pipeline_gpu1_case10.log` |

### Results (both finished ~14:57)

| Case | Set | Identity | Elastix | VoxelMap |
|------|-----|----------|---------|----------|
| **9** | 75 | 7.16 | **2.07** | **4.72** |
| **9** | 300 | — | **1.96** | **4.96** |
| **10** | 75 | 8.33 | **2.19** | **3.14** |
| **10** | 300 | — | **2.17** | **2.82** |

C9 soft like C7 (large Elastix→VoxelMap gap). C10 closer to teacher.

### A1 cohort summary (provisional — superseded by freeze below)

75-pt only; see frozen dual table.

---

## 2026-09-10 ~15:32 — A1 **frozen** (C1–C10)

C8 re-eval’d so `tre/tre_summary.json` has dual **75 + 300**.  
Artifacts: `arms/A1_oracle_dirlab/results/cohort_tre_frozen.{json,tsv}`.

**Recipe (locked):** keep-HU downsample → lung-masked Elastix (grid 16) → NoFiLM concat 50 ep → mean DVF over phase-01 projs `--stride 10`.

### Frozen TRE (mm) — T00→T50

| Case | Id 75 | El 75 | VM 75 | Id 300 | El 300 | VM 300 | gap75 (VM−El) |
|------|-------|-------|-------|--------|--------|--------|---------------|
| 1 | 3.91 | 1.24 | 1.33 | 3.89 | 1.12 | 1.18 | +0.08 |
| 2 | 4.65 | 1.09 | 1.38 | 4.34 | 1.06 | 1.32 | +0.29 |
| 3 | 7.25 | 1.30 | 1.86 | 6.94 | 1.30 | 1.80 | +0.56 |
| 4 | 9.69 | 1.89 | 3.10 | 9.83 | 1.77 | 3.18 | +1.22 |
| 5 | 7.41 | 2.08 | 2.32 | 7.48 | 1.91 | 2.22 | +0.23 |
| 6 | 11.77 | 2.93 | 3.83 | 10.89 | 2.60 | 3.72 | +0.90 |
| 7 | 10.71 | 2.25 | **5.03** | 11.03 | 2.34 | 5.30 | **+2.78** |
| 8 | 16.00 | 3.76 | 4.91 | 14.99 | 3.31 | 4.47 | +1.16 |
| 9 | 7.16 | 2.07 | **4.72** | 7.92 | 1.96 | 4.96 | **+2.65** |
| 10 | 8.33 | 2.19 | 3.14 | 7.30 | 2.17 | 2.82 | +0.95 |
| **mean** | **8.69** | **2.08** | **3.16** | **8.46** | **1.95** | **3.10** | — |

**Notes (not blockers):** C7 / C9 soft (largest El→VM gaps). C4 / C8 moderate. Do not mix 75 vs 300 when citing papers.

A1 oracle arm **closed** for comparison baselines.

---

## 2026-09-10 ~16:38 — A2 MC Val Prior prep launched

**Definition:** pool MC Val **Prior** P1–P9 → one NoFiLM VoxelMap; zero-shot TRE on DIR C1–C10.

Prep (same recipe as A1 labels: keep-HU → DRR → masked Elastix → prep_train):

| GPU | Order | Log |
|-----|-------|-----|
| **0** | P1→P5 | `arms/A2_generic_spare/logs/pipeline_gpu0_mc_p1_5.log` |
| **1** | P6→P9 | `arms/A2_generic_spare/logs/pipeline_gpu1_mc_p6_9.log` |

Script: `scripts/prepare_a2_mc_prior.py` · runs under `arms/A2_generic_spare/runs/MC_V_P*_Prior/`.  
Lung mask: SPARE GT `Mask_Lung` resampled onto Prior grid. Geom: each patient’s NS_01 (P9=SC_01), OffsetY −2.

**Prep finished** ~18:00 (all P1–P9 `ModelTraining` ready).

### Train launched ~18:30 — pooled NoFiLM

One ckpt over all 9 dirs (55 080 tgt projs, ~9× A1/epoch). Same recipe: concat NoFiLM, 50 ep, bs 8, lr 1e-5. **GPU 1.**

| Item | Path |
|------|------|
| Log | `arms/A2_generic_spare/logs/train_a2_pooled_nofilm.log` |
| Ckpt | `arms/A2_generic_spare/checkpoints/a2_spare_mc_val_prior_p1to9_concat_nofilm.pt` |

Rough ETA ~35–40 h (scales with dataset size). Next: DIR C1–C10 zero-shot TRE when train finishes.

### Train finished 2026-09-12 ~09:22

50/50 done. **Best val 0.0360 @ epoch 45** (final train/val 0.036 / 0.040).  
Ckpt: `arms/A2_generic_spare/checkpoints/a2_spare_mc_val_prior_p1to9_concat_nofilm.pt`  
Plot: `arms/A2_generic_spare/plots_nofilm/loss_curves.png`

### 2026-09-12 ~17:14 — A2 DIR zero-shot TRE **frozen**

Script: `scripts/eval_a2_tre.py` · log: `arms/A2_generic_spare/logs/eval_a2_cohort.log`  
Artifacts: `arms/A2_generic_spare/results/cohort_tre_frozen.{json,tsv}`

| Case | Id 75 | A2 VM 75 | A1 VM 75 | A1 El 75 |
|------|-------|----------|----------|----------|
| 1 | 3.91 | 4.07 | 1.33 | 1.24 |
| 2 | 4.65 | 4.94 | 1.38 | 1.09 |
| 3 | 7.25 | 7.36 | 1.86 | 1.30 |
| 4 | 9.69 | 9.86 | 3.10 | 1.89 |
| 5 | 7.41 | 7.67 | 2.32 | 2.08 |
| 6 | 11.77 | 12.06 | 3.83 | 2.93 |
| 7 | 10.71 | 10.58 | 5.03 | 2.25 |
| 8 | 16.00 | 16.13 | 4.91 | 3.76 |
| 9 | 7.16 | 8.12 | 4.72 | 2.07 |
| 10 | 8.33 | 8.92 | 3.14 | 2.19 |
| **mean** | **8.69** | **8.97** | **3.16** | **2.08** |

**Takeaway:** generic SPARE MC Prior VoxelMap ≈ **identity** on DIR (mean 8.97 vs id 8.69; often slightly worse). A1 oracle still ~3× better. Conditioning (A3) has a large gap to fill.

---

## 2026-09-12 ~19:10 — A3 Case 1 smoke launched (GPU 1)

**Recipe:** DIR CT_06 → HU→µ → G160-A1 Decoder synth 10 phases → keep-HU downsample → synth DVF labels → A1 geom DRR → NoFiLM 50 ep → TRE on A1 DIR ModelTraining.

- Prepare **done** in ~1.1 min (`runs/DIR_C01/ModelTraining/…`, 6120 pairs).
- Train **started** on GPU 1. Log: `arms/A3_synth_conditioned/logs/pipeline_gpu1_case1.log`
- Scripts: `prepare_a3_dir_case.py`, `eval_a3_tre.py`

---

## 2026-09-13 — Stop A3 default cohort; hist-match (D) on C1+C8

- **Stopped** GPU1 default A3 cohort (C5+). GPU1 idle.
- Ablation D (`clip HU[-1000,500]` + p1–p99 match to SPARE µ) → `--mu-mode hist_match`, runs under `DIR_C0N_muhist`.
- **Queued** Case 1 then Case 8 only on GPU1: prepare → NoFiLM 50 → TRE.
- Log: `arms/A3_synth_conditioned/logs/pipeline_gpu1_c1_c8_muhist.log`

---

## 2026-09-14 ~06:05 — A3 holdout C3 with G160 DIR-FT

- Pipeline: hist_match µ + `experiments/G160_DIR_FT_C1C5C8/weights/best.pt` → NoFiLM 50 → TRE
- Run: `arms/A3_synth_conditioned/runs/DIR_C03_g160ft/`
- Log: `arms/A3_synth_conditioned/logs/pipeline_gpu1_c3_g160ft.log`
- Synth phase-01 `|u|` ≈ **3.42** (was ~1.5 with SPARE decoder) — closer to Elastix scale

---

## 2026-09-13/14 — Init G160 Decoder DIR fine-tune (C1, C5, C8)

- Experiment: `experiments/G160_DIR_FT_C1C5C8/`
- Data: D01/D05/D08 @160³ µ(`hist_match`) + Elastix 06→XX (24 train / 6 val)
- Train: init SPARE G160-A1 Decoder, lr=1e-5, 50 ep → `weights/best.pt` (best val MSE 0.57)
- **Gate cos vs Elastix @160 (phase 01):**

| Case | Role | cos old → FT |
|------|------|-------------:|
| C1 | train | 0.12 → **0.52** |
| C8 | train | 0.07 → **0.82** |
| C2 | holdout | 0.14 → **0.72** |
| C3 | holdout | 0.22 → **0.78** |

Direction transfer improves a lot even on held-out DIR. Next: A3 re-synth with FT ckpt → VoxelMap TRE.

---

## 2026-09-13 ~21:30 — Stopped muhist C8; P2 done; P3 running

- Killed C8 muhist on GPU1.
- **P2** (`results/p2_patient_sensitivity.json`): mean pairwise cos **0.27** → not a locked template (`PATIENT_SPECIFIC_DIRECTION`). Still wrong vs Elastix (P1).
- **P3** training now: A3 synth DRRs + A1 Elastix labels → TRE. Log: `logs/pipeline_gpu1_p2_p3.log`. Run: `runs/DIR_C01_p3_semioracle/`.

---

## Open threads / next

1. A3: bake Elastix-convention (−pull / invert) DVF into `prepare_a3`; then full VoxelMap train on R3 DIR.
2. Optional: A2 C10 already in final table; freeze Results.md from `cohort_tre_r3_final`.
3. 4D-Lung DICOM → NIfTI / SYNTH prep (download complete on 4TB).
4. Soft-case diagnosis (C7/C9) optional with corrected orbit.

---

## 2026-09-14 — DIR DRR orbit vs SPARE (geometry.xml unchanged)

### Symptom
SPARE reference orbit (`LEARN-GUI/.../DRR_360_Simulation.mp4`) rotates about patient **S-I**: chest ↔ spine (AP → lateral → PA), head/feet fixed in-frame.  
Frozen A1 DIR DRRs (and early SI→Y-only probes) did **not**: anatomy swung off detector / wrong axis — not a per-angle bug in the XML.

### Root cause (Opus-validated)
1. **SI axis:** SPARE GTVol has SI on **itk Y** (thin axis); DIR native has SI on **itk Z**. RTK circular geom always orbits **world Y** → DIR must put SI on Y.
2. **Origin:** SPARE volumes are **centred on isocenter**; DIR had `origin=(0,0,0)` (corner). RTK orbits world origin → gantry circled ~outside the patient. **SI→Y permute alone is not enough** (rejected probe).
3. **Handedness / upright:** after permute, need **AP flip**; R1 (permute+AP+recenter) had correct chest↔spine but was **upside-down** vs SPARE → add **SI flip** (**R3**).

**Do not edit** `Geometry.xml` matrices/angles. Reuse `Geometry_SPARE.xml` (OffsetY −2). Frozen A1 used `arms/A1_oracle_dirlab/Geometry.xml` (OffsetY +2) on un-reoriented DIR volumes.

### Locked fix recipe (**R3**)
- Reorient DIR CT: `arr.transpose(1,0,2)[::-1,::-1,:]` (permute SI→Y + flip AP + flip SI), spacing `(LR, SI, AP)`, identity direction, **origin = −(size−1)×spacing/2**.
- Helper: `arms/A1_oracle_dirlab/plots/soft_case_diagnosis/C01_orbit_fix_probe/reorient_dirlab.py`
- DRR with **unchanged** `modules/drr_generation/Geometry_SPARE.xml` + MC/Varian detector opts.

### Artifacts
| Item | Path |
|------|------|
| Verdict (SPARE / R1 upside-down / **R3 fix**) | `arms/A1_oracle_dirlab/plots/soft_case_diagnosis/C01_orbit_fix_probe/C01_orbit_fix_VERDICT.png` |
| Orbit mp4 | `arms/A1_oracle_dirlab/plots/soft_case_diagnosis/C01_orbit_fix_T50.mp4` |
| SPARE ref video | `/home/abhishek/Documents/LEARN-GUI/MATLAB_FUNCTIONALITIES/DRR-GENERATION-PYTHON/DRR_360_Simulation.mp4` |

### Impact
Frozen A1 C1–C10 ModelTraining DRRs (and VoxelMap trained on them) used SI-on-Z + corner origin → **compromised**. Elastix TRE on volume landmarks can still look fine; projection→DVF learning does not. Full A1 re-prep + retrain required before trusting oracle VM / A3 numbers on corrected DRRs. Soft C7 FOV issues partly overlap this (large FOV + wrong orbit).

---

## 2026-09-14/15 — Incorrect-DRR archive + A1 R3 redo

- Moved wrong-orbit A1/A2/A3 under `arms/Incorrect DRR/`. Fresh arm folders keep R3 + `Geometry_SPARE.xml`.
- Baked R3 into `prepare_a1_case.py`; `eval_a1_tre.py --r3` landmark chain.
- Pipeline: prepare_all_r3 + train_all_r3 on **GPU1 only** (`logs/pipeline_train_all_r3_gpu1.log`).
- **2026-09-16 ~11:33** — `ALL TRAIN QUEUE DONE fail=0` (C01–C10 train + VM TRE).

### A1 VM TRE vs Incorrect-DRR (cohort mean over 4 TRE cells, approx)
Elastix ≈ unchanged. VoxelMap **3.21 → 2.15 mm (−1.06)** on C01–C09; hardest gains on C04/C07/C09.

---

## 2026-09-15 — Disk + TCIA 4D-Lung

- Archived SpareDVFs clinical Elekta/Varian → `/media/abhishek/3CCA3CADCA3C6574/SpareDVFs_archive_20260915/` (+ symlinks); freed ~289 G on root.
- Downloaded TCIA **4D-Lung 4DCT only** (S≥300): **1620/1620 series**, ~49.2 GB →  
  `/media/abhishek/3CCA3CADCA3C6574/TCIA_4D-Lung/{dicom,metadata,logs}/`  
  Script: `Voxel_GAN/scripts/download_4d_lung.py` (venv `.venv_tcia`). CBCT skipped (~507 studies).

---

## 2026-09-16 — Final R3 TRE: A1 oracle + A2 zero-shot

**A2:** no retrain (SPARE MC P1–9 ckpt still valid). Re-eval on R3 A1 `ModelTraining` with `eval_a2_tre.py --r3`.  
Ckpt symlink: `arms/A2_generic_spare/checkpoints/a2_spare_mc_val_prior_p1to9_concat_nofilm.pt` → Incorrect DRR archive.

**Δid = identity − TRE** (set75 T00→T50, mm; + = beat identity):

| Case | Id | A1 El | A1 VM | A1 Δid | A2 VM | A2 Δid | old A2 Δid |
|------|---:|------:|------:|-------:|------:|-------:|-----------:|
| C01 | 3.91 | 1.26 | 1.40 | +2.51 | 3.85 | +0.07 | −0.16 |
| C02 | 4.65 | 1.08 | 1.40 | +3.25 | 4.65 | +0.00 | −0.29 |
| C03 | 7.25 | 1.30 | 1.44 | +5.81 | 6.79 | +0.46 | −0.12 |
| C04 | 9.69 | 1.91 | 1.88 | +7.81 | 7.86 | +1.83 | −0.17 |
| C05 | 7.41 | 2.02 | 2.19 | +5.22 | 6.15 | +1.25 | −0.26 |
| C06 | 11.77 | 2.90 | 3.56 | +8.21 | 9.21 | +2.57 | −0.28 |
| C07 | 10.71 | 2.34 | 2.63 | +8.08 | 8.65 | +2.06 | +0.13 |
| C08 | 16.00 | 3.77 | 4.48 | +11.52 | 13.66 | +2.34 | −0.13 |
| C09 | 7.16 | 2.05 | 2.08 | +5.08 | 6.62 | +0.54 | −0.96 |
| C10 | 8.33 | 2.18 | 2.12 | +6.22 | 5.22 | +3.11 | −0.59 |
| **mean** | **8.69** | **2.08** | **2.32** | **+6.37** | **7.26** | **+1.42** | **−0.28** |

**Takeaway:** R3 A1 VM ≈ Elastix (mean 2.32 vs 2.08). A2 zero-shot now **beats identity by ~1.4 mm** (was slightly worse than id on wrong orbit); still far from A1. set300 mean Δid: A1VM +6.23 · A2 +1.37.

**Artifacts:** `arms/{A1_oracle_dirlab,A2_generic_spare}/results/cohort_tre_r3_final.{json,tsv}`

---

## 2026-09-16 — A3 synth smoke: DVF sign is convention, not a free tweak

### Setup
R3 A1 CT_06 → G160-A1 Decoder synth 10 phases → write `DVF_sub_*.mha` → TRE vs A1 Elastix on same R3 grid (`eval_a3_synth_vs_elastix.py --r3`). No VoxelMap train yet. Plots: `arms/A3_synth_conditioned/plots/synth_phase_panels/{axial,coronal,coronal_dvf_raw}/`.

### Cohort TRE75 T00→T50 (mm) — log `arms/A3_synth_conditioned/logs/signflip_all_cases.log`

| Case | Id | as-is (raw synth) | −DVF | Elastix |
|------|---:|------------------:|-----:|--------:|
| C01 | 3.91 | 6.60 | 2.15 | 1.26 |
| C02 | 4.65 | 8.25 | 2.76 | 1.08 |
| C03 | 7.25 | 10.87 | 4.14 | 1.30 |
| C04 | 9.69 | 13.65 | 6.13 | 1.91 |
| C05 | 7.41 | 11.40 | 4.52 | 2.02 |
| C06 | 11.77 | 16.05 | 8.00 | 2.90 |
| C07 | 10.71 | 13.57 | 8.24 | 2.34 |
| C08 | 16.00 | 19.51 | 13.04 | 3.77 |
| C09 | 7.16 | 9.51 | 6.25 | 2.05 |
| C10 | 8.33 | 11.14 | 6.09 | 2.18 |
| **mean** | **8.69** | **12.06** | **6.13** | **2.08** |

−DVF better than as-is **10/10**; −DVF beats identity **10/10**. Residual vs Elastix ≈ **+4.05 mm** (under-motion / model gap, not sign).

### Why −DVF is required (scientific — not an ad-hoc flip)

Two different displacement conventions are in play:

1. **Elastix / ITK (A1)** — fixed=`sub_CT_06` (T50), moving=phase 01 (T00). Transformix DF:
   \(\mathbf{x}_{\mathrm{moving}} \approx \mathbf{x}_{\mathrm{fixed}} + \mathbf{d}(\mathbf{x}_{\mathrm{fixed}})\).
   So \(\mathbf{d}\) is **fixed→moving** (T50→T00) on the fixed grid.
   TRE harness (`eval_a1_tre.py`): T00→T50 uses `pred = lm00 − disp`; T50→T00 uses `pred = lm50 + disp`.

2. **G160 synth pull field (A3)** — `warp` does `grid_sample` with `sample_grid = identity + flow` (`PopulationStudy/.../Experiment6/utilities/warp.py`):
   \(I_{\mathrm{tgt}}(\mathbf{x}) = I_{06}(\mathbf{x} + \mathbf{u}(\mathbf{x}))\).
   \(\mathbf{u}\) is a **sampling / pull** field on the **output (tgt)** grid: where to read in the reference. That is **tgt→ref** (~T00→T50) — opposite transport sense to Elastix \(\mathbf{d}\).

**First-order inverse:** \(\mathbf{d}_{\mathrm{Elastix}} \approx -\mathbf{u}_{\mathrm{synth}}\). Empirically SI component corr(Elastix, raw) ≈ **−0.83**, corr(Elastix, −synth) ≈ **+0.83**.

**Synth CT can look fine with raw \(\mathbf{u}\)** — CT is warped consistently with that pull field. CT QA does **not** validate Elastix/TRE convention. Feeding raw \(\mathbf{u}\) into the Elastix TRE path is the wrong sign → worse than identity.

**Caveat:** exact DF inverse ≠ exactly \(-\mathbf{d}\); \(-\mathbf{u}\) is the small-strain / first-order alignment. Bake Elastix-convention DVFs in `prepare_a3` long-term; do not treat sign as a free TRE knob.

µ-hist match (smoke C01) did **not** fix the gap; sign convention did.

### Open / next for A3
- Write −DVF (or true invert) into prepare when exporting labels for VoxelMap / TRE.
- Then A3 full train on R3 DIR.

---

## Quick reference paths

| Item | Path |
|------|------|
| Root README | `DIR EXPERIMENTS/README.md` |
| Results snapshot | `DIR EXPERIMENTS/Results.md` |
| TRE harness | `scripts/dirlab_tre.py` |
| A1 prepare (R3) | `scripts/prepare_a1_case.py` |
| A1 TRE eval | `scripts/eval_a1_tre.py --r3` |
| A1 R3 final cohort | `arms/A1_oracle_dirlab/results/cohort_tre_r3_final.json` |
| A1 train queue log | `arms/A1_oracle_dirlab/logs/pipeline_train_all_r3_gpu1.log` |
| A2 TRE eval | `scripts/eval_a2_tre.py --r3` |
| A2 R3 final cohort | `arms/A2_generic_spare/results/cohort_tre_r3_final.json` |
| A2 prepare | `scripts/prepare_a2_mc_prior.py` |
| A2 SPARE ckpt | `arms/A2_generic_spare/checkpoints/a2_spare_mc_val_prior_p1to9_concat_nofilm.pt` |
| Incorrect-DRR archive | `arms/Incorrect DRR/` |
| SPARE Geometry (R3) | `LEARN-GUI/.../Geometry_SPARE.xml` |
| DIR→SPARE reorient (R3) | `scripts/reorient_dirlab.py` / orbit_fix_probe helper |
| TCIA 4D-Lung 4DCT | `/media/abhishek/3CCA3CADCA3C6574/TCIA_4D-Lung/` |
| 4D-Lung download script | `Voxel_GAN/scripts/download_4d_lung.py` |
| A0 identity results | `arms/A0_identity/results/` |
| A3 synth vs Elastix TRE | `scripts/eval_a3_synth_vs_elastix.py` |
| A3 signflip cohort log | `arms/A3_synth_conditioned/logs/signflip_all_cases.log` |
| A3 phase / DVF panels | `arms/A3_synth_conditioned/plots/synth_phase_panels/` |

---

*Last updated: 2026-09-16 (A1/A2 R3 TRE frozen; A3 synth DVF pull-vs-Elastix sign convention + cohort TRE; 4D-Lung on 4TB).*
