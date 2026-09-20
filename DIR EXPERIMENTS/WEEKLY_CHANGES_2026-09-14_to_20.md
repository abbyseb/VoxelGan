# DIR EXPERIMENTS — what changed (14–20 Sep 2026)

**KPI throughout:** mean TRE (mm), DIR-Lab **75-point** landmarks, **T00→T50**, R3 frame after the DRR fix.

Two bugs were fixed (**DRR orbit**, **DVF sign**), then a better motion teacher (**TCIA2**) was trained. Tables below are **per arm**. Narrative “why” sections follow.

Detail: `diary.md`, `docs/DVF_pull_vs_push_sign.md`.

---

## Summary (cohort means)

| Arm / field | Wrong-orbit / wrong-sign era | After fixes (R3 + correct DVF) |
|-------------|-----------------------------:|-------------------------------:|
| A0 identity | 8.69 | 8.69 |
| **A1** Elastix | 2.08 | **2.08** |
| **A1** VoxelMap | **3.16** | **2.32** |
| **A2** VoxelMap (SPARE prior) | **8.97** (worse than id) | **7.27** (beats id) |
| **A3** synth oracle (SPARE G160) | 12.06 (raw \(u\)) | **6.13** (\(-\,u\)) |
| **A3** VoxelMap (SPARE G160 teacher) | ~id / worse | **6.49** |
| **A3** synth oracle (TCIA2 best) | — | **5.17** |
| **A3** VoxelMap (TCIA2 teacher) | — | **4.95** (n=6; C06/07/09/10 pending) |

---

## A1 — patient oracle (real 4D → DRR → NoFiLM)

**What changed:** volumes reoriented to **R3** + SPARE geometry; full re-prep + retrain. Elastix is volume-landmark registration (barely moved). VoxelMap learns from projections (large gain).

### Cohort

| Field | Incorrect DRR | After R3 | Δ |
|-------|--------------:|---------:|--:|
| Identity | 8.69 | 8.69 | 0 |
| Elastix | 2.08 | 2.08 | ~0 |
| **VoxelMap** | **3.16** | **2.32** | **−0.84** |

### Per-case TRE75 (mm) — A1 after R3

| Case | Identity | Elastix | VoxelMap | Δid (id − VM) |
|-----:|---------:|--------:|---------:|--------------:|
| C01 | 3.91 | 1.26 | 1.40 | +2.51 |
| C02 | 4.65 | 1.08 | 1.40 | +3.25 |
| C03 | 7.25 | 1.30 | 1.44 | +5.81 |
| C04 | 9.69 | 1.91 | 1.88 | +7.81 |
| C05 | 7.41 | 2.02 | 2.19 | +5.22 |
| C06 | 11.77 | 2.90 | 3.56 | +8.21 |
| C07 | 10.71 | 2.34 | 2.63 | +8.08 |
| C08 | 16.00 | 3.77 | 4.48 | +11.52 |
| C09 | 7.16 | 2.05 | 2.08 | +5.08 |
| C10 | 8.33 | 2.18 | 2.12 | +6.22 |
| **mean ± SD** | **8.69 ± 3.55** | **2.08 ± 0.81** | **2.32 ± 1.01** | **+6.37** |

### Per-case — A1 VoxelMap before vs after R3

| Case | VM (incorrect DRR) | VM (R3) | Δ |
|-----:|-------------------:|--------:|--:|
| C01 | 1.33 | 1.40 | +0.07 |
| C02 | 1.38 | 1.40 | +0.02 |
| C03 | 1.86 | 1.44 | −0.42 |
| C04 | 3.10 | 1.88 | −1.22 |
| C05 | 2.32 | 2.19 | −0.13 |
| C06 | 3.83 | 3.56 | −0.27 |
| C07 | 5.03 | 2.63 | −2.40 |
| C08 | 4.91 | 4.48 | −0.43 |
| C09 | 4.72 | 2.08 | −2.64 |
| C10 | 3.14 | 2.12 | −1.02 |
| **mean** | **3.16** | **2.32** | **−0.84** |

Hard / soft cases (C07, C09) improved most once projections matched anatomy.

---

## A2 — SPARE MC prior zero-shot on DIR

**What changed:** same SPARE-trained ckpt; only **re-evaluated on R3 A1** projections (no A2 retrain). Wrong-orbit A2 was ~identity or slightly worse; R3 A2 now beats identity but stays far from A1.

### Cohort

| Field | Incorrect DRR | After R3 | Δ |
|-------|--------------:|---------:|--:|
| Identity | 8.69 | 8.69 | 0 |
| **A2 VoxelMap** | **8.97** | **7.27** | **−1.70** |
| vs identity (Δid) | **−0.28** (worse) | **+1.42** (better) | — |

### Per-case TRE75 (mm) — A2

| Case | Identity | A2 (incorrect) | A2 (R3) | Δid R3 |
|-----:|---------:|---------------:|--------:|-------:|
| C01 | 3.91 | 4.07 | 3.85 | +0.07 |
| C02 | 4.65 | 4.94 | 4.65 | +0.00 |
| C03 | 7.25 | 7.36 | 6.79 | +0.46 |
| C04 | 9.69 | 9.86 | 7.86 | +1.83 |
| C05 | 7.41 | 7.67 | 6.15 | +1.25 |
| C06 | 11.77 | 12.06 | 9.21 | +2.57 |
| C07 | 10.71 | 10.58 | 8.65 | +2.06 |
| C08 | 16.00 | 16.13 | 13.66 | +2.34 |
| C09 | 7.16 | 8.12 | 6.62 | +0.54 |
| C10 | 8.33 | 8.92 | 5.22 | +3.11 |
| **mean ± SD** | **8.69** | **8.97 ± 3.49** | **7.27 ± 2.82** | **+1.42** |

Takeaway: generic SPARE prior **transfers poorly** to DIR; R3 helps, but A2 is not a substitute for A1/A3.

---

## A3 — synth-conditioned (G160 teacher → DIR VoxelMap)

**What changed:** (1) R3 DRRs, (2) bake **`DVF_sub = −u`** (Elastix convention), (3) optional **TCIA2** teacher instead of SPARE G160.

### 3a. Synth oracle only (no VoxelMap) — DVF sign

| Case | Identity | Raw \(u\) | **\(-\,u\)** (Elastix) | A1 Elastix |
|-----:|---------:|----------:|-----------------------:|-----------:|
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

\(-\,u\) beats raw **10/10** and beats identity **10/10**. Residual to Elastix ≈ under-motion / model gap, not sign.

### 3b. Synth oracle — SPARE G160 vs TCIA2 teacher

| Case | SPARE G160 oracle | TCIA2 best oracle | Δ (TCIA − SPARE) |
|-----:|------------------:|------------------:|-----------------:|
| C01 | 2.15 | 2.13 | −0.02 |
| C02 | 2.76 | 2.12 | −0.64 |
| C03 | 4.14 | 3.41 | −0.73 |
| C04 | 6.13 | 5.89 | −0.24 |
| C05 | 4.52 | 3.78 | −0.74 |
| C06 | 8.00 | 6.59 | −1.41 |
| C07 | 8.24 | 7.70 | −0.54 |
| C08 | **13.04** | **10.07** | **−2.97** |
| C09 | 6.25 | 4.74 | −1.51 |
| C10 | 6.09 | 5.23 | −0.86 |
| **mean ± SD** | **6.13 ± 3.15** | **5.17 ± 2.51** | **−0.96** |

Tough cases (C06–C08) move most; easy cases were already near the floor.

### 3c. A3 VoxelMap — SPARE teacher (full cohort)

Student ≈ teacher ceiling (~6.1 mm oracle → ~6.5 mm A3).

| Case | Identity | A1 Elastix | A3 VM (SPARE) | Δid |
|-----:|---------:|-----------:|--------------:|----:|
| C01 | 3.91 | 1.26 | 2.16 | +1.76 |
| C02 | 4.65 | 1.08 | 2.75 | +1.90 |
| C03 | 7.25 | 1.30 | 3.91 | +3.34 |
| C04 | 9.69 | 1.91 | 7.30 | +2.39 |
| C05 | 7.41 | 2.02 | 4.51 | +2.89 |
| C06 | 11.77 | 2.90 | 8.94 | +2.83 |
| C07 | 10.71 | 2.34 | 9.33 | +1.38 |
| C08 | 16.00 | 3.77 | 13.61 | +2.39 |
| C09 | 7.16 | 2.05 | 6.18 | +0.98 |
| C10 | 8.33 | 2.18 | 6.16 | +2.17 |
| **mean ± SD** | **8.69** | **2.08** | **6.49 ± 3.48** | — |

### 3d. A3 VoxelMap — TCIA2 teacher (6/10 done; C06/C07 training, C09–C10 queued)

| Case | A3 SPARE | A3 TCIA2 | Δ |
|-----:|---------:|---------:|--:|
| C01 | 2.16 | **1.97** | −0.19 |
| C02 | 2.75 | **2.20** | −0.55 |
| C03 | 3.91 | **3.97** | +0.06 |
| C04 | 7.30 | **6.06** | **−1.24** |
| C05 | 4.51 | **4.03** | −0.48 |
| C06 | 8.94 | *training* | — |
| C07 | 9.33 | *training* | — |
| C08 | 13.61 | **11.45** | **−2.16** |
| C09 | 6.18 | *queued* | — |
| C10 | 6.16 | *queued* | — |
| **mean (n=6)** | **5.71** | **4.95 ± 3.51** | **−0.76** |

C08 / C04 move most with the better teacher; C03 flat. Student still tracks oracle ceiling (~5.2 mm TCIA2 vs ~6.1 SPARE).

---

## Why the DRR was wrong

RTK / SPARE circular DRR orbits **world Y** (patient SI) around the **volume isocentre**.

DIR had instead:

1. **SI on itk Z** (SPARE uses itk Y) → gantry spun about the wrong patient axis.  
2. **Corner origin** `(0,0,0)` (SPARE is centred) → orbit centre sat outside the patient.  
3. **Permute alone insufficient** — need AP + SI flips as well (**R3**).

We did **not** edit `Geometry.xml` angles. SPARE XML (OffsetY **−2**) was already correct for SPARE-shaped volumes; DIR volumes had to enter that frame.

Elastix TRE (volume landmarks) can look fine with bad DRRs. **VoxelMap learns from projections** — wrong orbit inflated A1 VM until R3 re-prep + retrain.

**Figures:** `arms/Incorrect DRR/A1_oracle_dirlab/plots/soft_case_diagnosis/` — start with `C01_orbit_fix_probe/C01_orbit_fix_VERDICT.png` and `C01_orbit_fix_T50.mp4` (see README there).

---

## Why the negative (−u) is needed

**Problem:** two tools define “displacement” in **opposite** directions — not wrong, just different standards.

| Check | Raw \(u\) (G160 pull) | \(-\,u\) (Elastix / TRE) |
|-------|----------------------:|-------------------------:|
| Corr vs Elastix \(u_{SI}\) (lung, C01–C10 mean) | **−0.64** | **+0.64** |
| Synth oracle TRE75 | **12.1 mm** (worse than identity 8.7) | **6.1 mm** (better than identity) |

| Convention | Meaning |
|------------|---------|
| **Elastix / TRE** | Fixed=T50, \(x_{\mathrm{mov}} \approx x_{\mathrm{fix}}+d\) → \(d\) is **fixed→moving** (push / landmark) |
| **G160 warp** | \(I_{\mathrm{phase}}(x)=I_{06}(x+u(x))\) → \(u\) is a **pull / sampling** field on the **output** grid |

**We do synthesize from reference → other phases** (`CT_06` / T50 is the source volume; we build `CT_01`…). That narrative is correct. The mismatch is **what the stored vector means**: for each voxel \(x\) on the new phase, `warp` **samples** the reference at \(x+u\). Tissue that lands at \(x\) came from \(x+u\) on T50 → anatomical motion T50→phase ≈ **\(-\,u\)**. So \(u\) is **not** “push this voxel along the breath.”

**Figures**
- SI sign flip (raw vs −u): `arms/A3_synth_conditioned/plots/synth_phase_panels/coronal_dvf_raw/vs_elastix/`
- Voxel scatter + \(r\): `…/dvf_sign_scatter/DIR_C01_dvf_si_elastix_vs_synth_sign_scatter.png` (also cohort pooled PNG); script `scripts/plot_dvf_sign_scatter.py`

**Why it went unnoticed:** synthetic CTs looked fine either way — image warping is self-consistent with whatever sign the warper expects. Only landmark TRE exposed the Elastix mismatch.

**Why \(-\,u\) isn’t a perfect fix:** negation ≈ first-order field inversion, not a true inverse. Residual oracle ~**6.1 mm** vs Elastix ~**2 mm** is real teacher/model error, not leftover sign bug.

**Fix going forward:** convert once at the source in `prepare_a3_dir_case.py` (`DVF_sub = -u`, `--dvf-convention elastix`) — don’t flip ad hoc at every eval/plot.

---

## Why SPARE A3 is bounded — and why TCIA2

A3 VoxelMap is a **student of the synthesizer**. Whatever DVF the G160 teacher can invent on DIR `CT_06` is the ceiling the student can learn from projections.

| Teacher | Synth oracle TRE75 | A3 student (same cases) |
|---------|-------------------:|------------------------:|
| SPARE G160 (MC prior) | **6.13 mm** | **6.49** (full) / **5.71** on n=6 done |
| **TCIA2** (4D-Lung Decoder) | **5.17 mm** | **4.95** (n=6 so far) |

**Why SPARE wasn’t enough:** SPARE motion is a narrow domain. On easy DIR cases the oracle was already near the floor (C01 ~2 mm). On **large-motion / hard** cases the teacher **under-moves** → student stuck near identity-ish TRE (C08 SPARE oracle **13.0**, A3 **13.6**).

**Why TCIA2 is needed:** more patients, more breath diversity, same R3/A3-compatible frame → a better motion prior to warp DIR anatomy. That lifts the **oracle** first; A3 follows because it is trained on those synth DVFs/DRRs.

**Where the gain shows (hard cases):**

| Case | SPARE oracle | TCIA2 oracle | A3 SPARE | A3 TCIA2 | Δ student |
|-----:|-------------:|-------------:|---------:|---------:|----------:|
| C04 | 6.13 | 5.89 | 7.30 | **6.06** | **−1.24** |
| C06 | 8.00 | 6.59 | 8.94 | *training* | — |
| C07 | 8.24 | 7.70 | 9.33 | *training* | — |
| **C08** | **13.04** | **10.07** | **13.61** | **11.45** | **−2.16** |
| C09 | 6.25 | 4.74 | 6.18 | *queued* | — |

Easy cases barely move (C01 2.16→1.97). The story is **hard-case headroom**, not a uniform −1 mm everywhere. Residual to A1 Elastix (~2 mm) is still synthesizer/domain quality — not DRR/sign bugs.

---

## TCIA2 — data, why, and training steps

### What it is trained on

- **82** TCIA 4D-Lung scans × **10** phases = **820** phase CTs  
- Packed to **160³ @ 2 mm**, **R3** (SPARE/A3 axes + centred)  
- **Elastix on HU** (good MI); network trains on **µ** (`CT_*_mu.npy`) — matches A3 synth intensity  
- **100 phase-pairs / scan** (incl. identity) → **8200** pairs → **7380 train / 820 val** (10% pairs held out per scan)

One-liner (`seed.json`): **A1 FOV — R3 — Elastix HU / train µ — A3-compatible**.

### Why this recipe (training steps)

1. **Repack / R3** — put TCIA volumes in the same patient frame as DIR A3 (SI on Y, centred) so warped DIR + SPARE geometry stay consistent.  
2. **Elastix library on HU** — build phase-pair DVF labels with contrast that registration likes.  
3. **Train Decoder on µ + FOV aug** — same intensity and half-fan/CBCT corruptions A3/VoxelMap see, so the teacher isn’t SPARE-MC-only.  
4. **Pick best by val MSE** — then run as DIR synth oracle (`CT_06` → 10 phases) and bake `DVF_sub = −u` for TRE/A3.  
5. **A3 student** — train VoxelMap on those TCIA2 synth DRRs/DVFs; TRE on real DIR landmarks.

### Optim / schedule

| Item | Setting |
|------|---------|
| Model | **UNetCRBDecoder**, phase-conditioned (`n_phases=10`) |
| Loss | Lung-masked **MSE** vs Elastix DVF |
| Crops | **64³**; train random + FOV ¼; val fixed, no FOV |
| Patches | 16/pair train, 8/pair val |
| Optim | Adam **1e-4**, batch **1**, no weight decay, **100** epochs |
| Best | min val MSE **0.453** @ ep98 (ep100 val 0.491) |

Train≪val gap (~0.22 from ~ep20) is a stable generalization gap, not a broken run. Resume past 100 needs `--epochs` raised (`start_train.sh` defaults to 100).

Path: `PopulationStudy/ClinicalExperiments/Grid160/TCIA2/` (`seed.json`, `scripts/train_mse.py`, `scripts/start_train.sh`).

---

## Tooling

- TRE viewer (`tools/tre_viewer`): overlays, DRR/RTK, Phase Performance; on `main` / `TRE-VIZ`.

---

## Where to look

| Topic | Path |
|-------|------|
| This note | `DIR EXPERIMENTS/WEEKLY_CHANGES_2026-09-14_to_20.md` |
| Diary | `DIR EXPERIMENTS/diary.md` |
| DVF pull vs push | `DIR EXPERIMENTS/docs/DVF_pull_vs_push_sign.md` |
| A1/A2 R3 cohort JSON | `arms/A1_oracle_dirlab/results/cohort_tre_r3_final.json` |
| Incorrect-orbit archive | `arms/Incorrect DRR/` |
| A3 SPARE runs | `arms/A3_synth_conditioned/runs/DIR_C0N/` |
| A3 TCIA2 runs | `arms/A3_synth_conditioned/runs/DIR_C0N_tcia2/` |
| TCIA2 oracle JSON | `…/TCIA2/DecoderCRB/plots/qc_dir_oracle/tre75_final_best_vs_ep100_vs_spare.json` |
| A3 phase / DVF panels | `arms/A3_synth_conditioned/plots/synth_phase_panels/` |
| DVF sign scatter (\(r\)) | `…/plots/synth_phase_panels/dvf_sign_scatter/` |

*Updated 2026-09-21 — TCIA2 data (82 scans), why/hard-case gains, training steps.*
