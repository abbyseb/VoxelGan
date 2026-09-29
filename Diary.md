# 2026-09-29 11:40 — Literature comparison for single-CT motion synthesis

No published method found that predicts full T00↔T50 motion from **one CT only** and scores it with DIR-Lab / POPI landmark TRE. Closest work:

| Study | Input | Landmark error |
|---|---|---|
| Ehrhardt et al., 4D mean motion model (Lung workshop 2009; IEEE TMI 2011) | 1 CT **+ spirometric ΔV_air** (breath depth) | 3.3 ± 1.8 mm normal lungs, 4.2 ± 2.2 mm impaired; own 10 patients, not DIR-Lab. Registration lower bound 1.6 mm. Cite TMI 2011 for final numbers. |
| Fuerst et al., personalised biomechanical model | **2 CTs** (inhale + exhale), predicts in-between phases | 3.88 ± 1.54 mm, DIR-Lab C06–C10 |
| Cao et al. 2024 (arXiv 2404.00163) | 1 CT + belt amplitude (AdaIN) | tumour centre error 2.35 mm, not DIR-Lab |
| Vaurs et al., EMBC 2025 (HAL hal-05302619) | MRI lung **masks**, previous N phases → next phase (LSTM) | Dice/IoU only, no TRE. Healthy volunteers. Not comparable. |
| DIR-Lab registration (both images) | 2 images | best ~0.8–1.0 mm; DL ~1.0–2.5 mm. Our Elastix labels 1.96 mm. |

- Ours: 1 CT only → 4.25 (TCIA3) / 4.08 (MagFT ep30) on DIR-Lab TRE300, 4.78 on POPI. For C06–C10 only: 5.41 (TCIA3).
- Our oracle with one head-foot scale per case: 3.60 → matches Ehrhardt's 1 CT + breath depth range. Literature and oracle agree: 1 CT ≈ 4 mm; 1 CT + breath depth ≈ 3.3–3.6 mm.
- Our Elastix labels (1.96 mm) are ~1 mm worse than best registration → cleaner labels could trim a few tenths.

# 2026-09-29 11:25 — POPI re-scored correctly: TCIA3 ep100 4.78 ± 1.75 mm (identity 8.12)

Old POPI 6.89 mm is **invalid**. Two bugs in `eval_popi_tcia3.py`:
1. POPI `.mhd` says "RAI". In ITK/MetaIO that code means the **LPS** world (+x left, +y posterior, +z superior). The script read it literally (+x right, +y anterior, +z inferior). Head-foot errors cancelled, but front-back and left-right were both reversed → every chest rotated 180° about the head-foot axis.
2. Lung mask (−950…−300 HU) picked up table foam/streaks → 160³ box pulled towards the back, table inside, front of lung cut off.

Fix: `DIR EXPERIMENTS/scripts/eval_popi_tcia3_lps.py` (read as LPS; lung = air inside the filled body; spine-end check vs TCIA S1 stops the run on failure). New cubes `/media/.../POPI/packed_r3_tcia3_lps/`, result `popi_tcia3/tre_tcia3_ep100_lps.json`. Check script `check_popi_orientation.py --packed …` uses the same mask.

QC: all 6 spine at TCIA end, all phase-00 landmarks inside the cube, left-right OK, axial pictures correct. `lm_in_lung` 0.62–0.82 is expected (landmarks sit on vessels/airways brighter than −400 HU).

| Patient | Identity | TCIA3 |
|---|---:|---:|
| bl (00→06) | 5.90 | 3.39 |
| ng | 14.04 | 7.98 |
| dx | 7.67 | 5.61 |
| gt | 7.33 | 4.89 |
| mm2 | 7.09 | 4.31 |
| bh | 6.68 | 2.48 |
| **Mean** | **8.12** | **4.78 ± 1.75** |

- POPI feeds phase 00 (breathed-in) CT; DIR feeds phase 06 (breathed-out). Second test set, not like-for-like.
- ng (14 mm breath) is POPI's C08: breath-size problem.
- Leftover FLIPPED head-foot flags came from an older mask in the check script; pictures and spine check say orientation is correct.

# 2026-09-29 10:45 — Scorer fully verified (`verify_iso2mm.py`): OVERALL PASS

`DIR EXPERIMENTS/scripts/verify_iso2mm.py`, TCIA3 ep100, all 10 cases. JSON `.../qc_dir_oracle/verify_iso2mm_20260929_104430.json`.
1. Round trip (Elastix field through the new 2 mm path vs old evaluator, 75+300, both directions): within 0.002 mm on every case.
2. Per-channel r vs Elastix in lung: head-foot 0.70–0.85, front-back 0.5–0.8, left-right 0.3–0.7. All > 0. **C07 front-back r = +0.01** (its AP motion is uncorrelated; with size ratio 0.53 → second-worst case).
3. Warped CT_06 beats unwarped against real CT_01 on all 10 (NCC), below Elastix warp.
4. Reverse T50→T00 with exact point inversion (fixed-point, residual < 0.01 vox): network 4.48 vs forward 4.25; Elastix 2.12 vs 1.96. Gap is mostly geometry (lung shrinks on exhale, errors stretch going back); network-specific inconsistency ≈ 0.08 mm. Not a priority. **Report forward numbers.**

# 2026-09-29 09:50 — Corrected DIR geometry: TCIA3 4.25 mm, MagFT ep30 4.08 mm; size is the gap, direction is fine

**Bug found in all A3 / oracle DIR results before today:** `synth_phase01` / `prepare_a3_dir_case.py` squeezed the whole native DIR CT into 160³ (`resize_volume_zyx`), so voxels were 1.5–3.1 mm and anisotropic instead of 2 mm iso (C06–C10 chests squashed to ~65% width), and brightness used `norm_mu` (water 0.67) instead of the per-scan min-max used in training (water ~0.3).

Fix: `DIR EXPERIMENTS/scripts/eval_dir_tcia3_iso2mm_v2.py` — resample to 2 mm, 160³ centred on the lung, per-scan µ min-max, **no flip** (the v1 flip fed upside-down chests → 13.3 mm), landmarks via the verified chain official → pack → R3 (v1 skipped official→pack and mirrored landmarks head-foot; identity check can't see a mirror). Orientation picture `orientation_tcia_s1_vs_dir_c01.png`.

TCIA3 ep100, TRE300 (TRE75): **4.25 (4.44)**, was 4.70 (4.94) on the squeezed input. All landmarks inside the cube.

Re-score of old checkpoints on the corrected path (`rescore_oracle_iso2mm_v3.py`), TRE300 / C08:
MagFT ep30 **4.08** / 8.63 · MagFT ep12 4.13 · TCIA3.5 ep92 4.18 · MagMatch ep6 4.18 · MagFT ep27 4.21 · AmpHead ×1.04 4.23 · TCIA3 ep100 4.25 / 9.15 · TCIA3.5 ep100 4.28 · TCIA3.1 ep74 4.47 · TCIA4 ep26 4.47. MagMatch Part 2 not run (no checkpoint beat 4.08 or got C08 < 7.5).
**MagFT ep30 = new best synthesiser** (last epoch of its run, not DIR-picked).

Oracle on TCIA3 ep100 (Elastix-assisted, ceiling only), TRE300: base 4.25 · Elastix 1.96 · perfect size (mag_el) **2.85** · perfect direction 3.73 · smooth size σ4 3.14 · σ8 3.85 · one head-foot scale per case 3.60. Saved as `rescore_oracle_iso2mm_v3_with_oracle.json`.

Direction vs Elastix (lung cos) went **0.32 → 0.85**, size ratio 0.44 → 0.82. The 21–27 Sep update's "C08 points the wrong way" was caused by the squeezed input. Remaining error is **per-patient breath size**, both ways: C01 ×1.44 (too big), C02/03/05/09 ≈ 1, C04/06/10 ≈ 0.65–0.70, C07 0.53, C08 0.39. A global "move more" push (MagFT/MagMatch) can't fix both ends.

Consequences:
- A1 and A2 are unaffected (never used the synthesiser path).
- **All A3 arms must be redone** with the corrected input (`prepare_a3_dir_case.py` lines ~288–289 still squeeze + norm_mu). Use MagFT ep30.
- Old AmpHead / FeatAmpHead / MagFT / MagMatch / scale-map / amp4 / lung-crop conclusions were all on the broken input.

Other audit findings (not yet fixed):
- TCIA3 validation split = 10% of pairs from the same 82 patients (`build_pooled_dataset.split_val`) → val MAE is not a new-patient estimate. TCIA4 uses a proper 10-scan holdout.
- `fast_dataset.py` (TCIA3, TCIA4, SPARECRB) seeds augmentation with `aug_seed + idx…` only → same augmentation every epoch. `utilities/dataset.py` has `set_epoch` but the fast loaders ignore it and `train_mae.py` never calls it.
- TCIA3 resume at ep49: `cuda rng restore failed`.
- Several files are 0–1 bytes with the same 26 Sep timestamp (e.g. `Voxel_GAN/scripts/*.py`, `repack_tcia_r3_*.py`, `TCIA3/README.md`, `qc_synth_oracle_tre75.py`) → restore from git.
- Per-scan min-max differs from standard fixed HU window (water 0.25 TCIA vs 0.34 DIR); switch at next retrain.

# 2026-09-25 13:46 — C08 amp4 TRE at epoch 40

Val-best so far (val 0.205). Real A1 projections, stride 10, R3. TRE75 **11.31 mm**, TRE300 **10.61 mm**. Epoch 16 was 11.34 / 10.60, epoch 25 was 11.48 / 10.81, one-depth was 11.82. Elastix on the same case is 3.77. Loss kept falling; landmark error did not. Snapshot: `tre/tre_summary_ep040.json`. Training left running to epoch 50.

# 2026-09-25 13:45 — TCIA3.5 launched (MAE, no FOV aug)

Same recipe as TCIA3 (lung-masked MAE, 82 scans, 100 ep, lr 1e-4, seed 20260918) with FOV/CBCT augmentation off. New folder `TCIA3.5/` only. TCIA3 epoch_100 is not read or written. A full TCIA3 run was **41 h 54 min**. This one shares GPU0 with the C08 amp4 VoxelMap job until that job finishes. Script: `TCIA3.5/scripts/train_mae_noaug.py`.

# 2026-09-25 12:35 — Why amp4 C08 does not move: depth is learned, but not seen on real views

Probe with epoch-25 weights, phase 01, lung-masked p95 |u| (sub-voxels, 128³):

| Input projections | Label p95 | Predicted p95 |
|---|---:|---:|
| synth a=0.8 | 2.80 | 2.62 |
| synth a=1.0 | 3.50 | 3.29 |
| synth a=1.3 | 4.55 | 4.62 |
| synth a=2.0 | 7.00 | 6.19 |
| **real C08 (A1)** | **13.39 (Elastix)** | **3.06** |

- On synthetic views the output tracks the depth label at all four scales. The network can read depth from projections.
- On the real C08 views it answers ~3.1, i.e. about the a=0.9 depth. Real motion is ~3.8× the Decoder's. Two gaps: (1) 2.0× is far below what C08 needs; (2) the real views are not recognised as deep at all — the prediction sits near the shallow end, not at the top of the trained range.
- More angles at the same depths (the ~28 h version) would not change either gap. Not worth running.

# 2026-09-25 12:26 — C08 amp4 TRE at epoch 25 (still training)

best.pt is epoch 25 (val 0.304). Real-projection TRE75 **11.48** mm, TRE300 **10.81**. Epoch 16 was **11.34**. One-depth A3 C08 was **11.82**. Loss fell, TRE did not. Snapshot: `runs/DIR_C08_tcia3_amp4/tre/tre_summary_ep025.json`.

# 2026-09-25 11:44 — C08 amp4 mid TRE (epoch 16, still training)

best.pt is epoch 16 (val 0.417). Real-projection TRE75 **11.34** mm, TRE300 **10.60**. One-depth A3 C08 was **11.82**. Down 0.48 mm, not yet near the gate of 8. Training continues to epoch 50. Snapshot: `runs/DIR_C08_tcia3_amp4/tre/tre_summary_ep016.json`.

# 2026-09-25 10:05 — C08 multi-depth VoxelMap started (GPU0)

Train VoxelMap on the same anatomy at four Decoder-motion scales: **0.8, 1.0, 1.3, 2.0**. Each depth keeps every 4th angle (170 views, same angles at every depth). 170 × 9 phases × 4 depths = **6120** pairs, same budget as the one-depth run (~7 h).

- New run only: `A3_synth_conditioned/runs/DIR_C08_tcia3_amp4/`. TCIA3 epoch_100 and the GPU1 A3 sequence are untouched.
- TRE still uses real A1 projections and landmarks.
- Gate: if C08 TRE75 falls from 11.8 toward 8 or below, run the other cases.
- Script: `prepare_a3_dir_case.py --amplitudes 0.8,1.0,1.3,2.0`. Log: `logs/case_DIR_C08_tcia3_amp4.log`.

# 2026-09-25 09:34 — Projections for SI stretch: noted, not a mid-CT win

Real phase-01/06 projections exist **before** synth (scanner data, 680×128², RTK geom in A1 `train/`). Decoder never sees them. A3 order: Decoder (mid-CT) → synth 4D → **synth** DRRs → VoxelMap. Those DRRs inherit under-motion.

Idea: measure diaphragm row shift on **real** 01 vs 06 views → mm via geometry → stretch Decoder SI by (measured / predicted). No Elastix, no landmarks.

**Ruling:** cheating if the claim is “Decoder / mid-CT teacher improved.” Not cheating only if the product is VoxelMap and those projections are already a test-time input. User rejected it for the fair mid-CT test. Not implemented.

# 2026-09-25 09:23 — MagMatch KILLED at ep10 (gate fail)

- Direct |û|↔|El| + SI mag-match + TCIA3.1 oversample, init TCIA3 ep100
- DIR TRE75: ep5 **5.15** (gain −0.22), ep10 **5.13** (gain −0.19) vs TCIA3 **4.94**
- Worse than baseline; gate required +0.3 mm → stopped
- TCIA3 unchanged. Weights: `TCIA3_magMatch/DecoderCRB/weights/..._generator.pth` (not useful)
- Gate log: `TCIA3_magMatch/logs/gate_log.json`

# 2026-09-24 23:00 — MagMatch FT launched (stronger |û|↔|El|)

- New folder `TCIA3_magMatch/` — does **not** touch TCIA3 or MagFT
- Init: TCIA3 `epoch_100.pt` (read-only)
- Loss: MAE + λ_mag=1.0·L1(|û|,|El|) + λ_si=0.5·L1(|SI|)
- Data: TCIA3.1 **oversampled** train (24109 pairs → 96436 patches/ep), val 820
- LR 3e-5, max 15 ep, GPU0
- **Gate:** DIR TRE75 @ ep5 & ep10; kill if gain < 0.3 mm vs 4.938
- Script: `TCIA3_magMatch/scripts/train_finetune_magmatch.py`
- Log: `TCIA3_magMatch/logs/train_20260924_225909.log`

# 2026-09-24 22:53 — MagFT done; final DIR TRE

- 30/30 finished; val-best ep27 (0.5592); TCIA3 ep100 untouched
- Synth-oracle cohort:

| Model | TRE75 | TRE300 | Δ75 vs TCIA3 |
|-------|------:|-------:|-------------:|
| TCIA3 ep100 | 4.94 | 4.70 | — |
| MagFT **ep12** (best TRE) | **4.76** | **4.52** | **−0.18** |
| MagFT val-best (ep27) | 4.89 | 4.63 | −0.04 |
| MagFT ep30 | 4.97 | 4.71 | +0.03 |

- C08 TRE75: 10.65 → 9.80 (ep12)
- Verdict: hinge MagFT max ~−0.2 mm; not the −1 mm path. Prefer `epoch_012.pt` if using MagFT at all.
- JSON: `arms/A3_synth_conditioned/results/magft_final_tre75_300.json`

# 2026-09-24 14:15 — MagFT ep001 DIR TRE75

- Ckpt: `TCIA3_magFT/.../epoch_001.pt` (best val so far)
- Cohort TRE75 **4.863 ± 2.47** vs TCIA3 ep100 **4.938** (**+0.075 mm** only)
- C08 10.31 (was 10.65); C07 7.01 (was 6.67, worse); mixed per-case
- Under-move FT not delivering ~1 mm. JSON: `amp_oracle_magft_ep001_a1.json`

# 2026-09-24 12:30 — MagFT: TCIA3 ep100 fine-tune (copy, not overwrite)

- **Does NOT edit** `TCIA3/DecoderCRB/checkpoints/epoch_100.pt` (read-only init)
- New run: `Grid160/TCIA3_magFT/` — MAE + SI under-move hinge (λ=1, eps=0.25)
- LR 3e-5, 30 ep, GPU0; ckpts → `TCIA3_magFT/DecoderCRB/checkpoints/`
- Script: `TCIA3_magFT/scripts/train_finetune_mag.py`
- Log: `TCIA3_magFT/logs/train_*.log` (train size 59040 patches)

# 2026-09-24 12:17 — FeatAmpHead DIR TRE: FAIL

- Train done: best val L1(a)=**0.166** @ ep97 (`AmpHeadFeat/amp_head_feat_best.pt`)
- DIR: predicted **a=0.800 for all 10 cases** (hit floor)
- Cohort TRE75: a=1 **4.94** → SI×a **5.50** (−0.56) / iso×a **5.43** (−0.49) — **worse**
- Likely domain mismatch: TCIA 64³ µ-crop feats vs DIR 128³ HU+|û| stats; net collapsed to min a
- JSON: `arms/A3_synth_conditioned/results/amp_head_feat_dir_tre75_tcia3_ep100.json`

# 2026-09-24 12:13 — FeatAmpHead (mid-CT + |û| → a)

Fair (no other-phase peek): tiny MLP predicts scale from mid-CT anatomy + Decoder motion stats.
- Net: Linear(8→64)→ReLU→Linear(64→64)→ReLU→Linear(64→1), a=0.8+1.7σ(z)
- Feats: φref, φtgt, log p95/mean |û|, log p95/mean |û_SI|, lung_frac, lung_mean
- Train: TCIA only, freeze TCIA3 ep100; target a*=p95|El|/p95|û|
- Out: `TCIA3/AmpHeadFeat/`; script `train_amp_head_feat.py` (GPU0, 100 ep)
- DIR eval after train: `DIR EXPERIMENTS/scripts/eval_amp_head_feat_dir_tre.py`

# 2026-09-24 11:03 — A3×TCIA3 seq GPU1 (item 1) running

- Pipeline already live: `pipeline_a3_tcia3_seq_gpu1.sh` (pid watch via log)
- **Done:** C01 + C08 (train+TRE). C01 TRE75~2.13; C08~11.82
- **Now:** C03 VoxelMap ~ep9/50 on GPU1 (~8.5 min/ep → ~5.8 h left)
- **Next:** C02 retrain+TRE → C04,5,6,7,9,10 full (prepare→train→TRE), GPU1 only
- Log: `DIR EXPERIMENTS/arms/A3_synth_conditioned/logs/pipeline_a3_tcia3_seq_gpu1.log`

# 2026-09-24 10:58 — AmpHead DIR TRE (TCIA3 ep100)

- AmpHead `a(φ=5→0)` = **1.042** (near identity; phase-only → one a for all DIR cases)
- Cohort TRE75: a=1 **4.938** → AmpHead **4.861** (**+0.077 mm**)
- Per-case Δ mostly +0.05–0.19 mm; C01/C02/C09 slightly worse
- Conclusion: TCIA-trained phase MLP does **not** unlock DIR; far below a-oracle best-per-case (4.18) and even below best global a=1.3 (~+0.18)
- JSON: `DIR EXPERIMENTS/arms/A3_synth_conditioned/results/amp_head_dir_tre75_tcia3_ep100.json`

# 2026-09-24 11:25 — Magnitude vs direction oracle (TCIA3 ep100, DIR TRE75)

Offline, uses Elastix, not deployable. Base 4.94, Elastix 2.08.
- Perfect per-voxel magnitude, our direction: **3.64** (−1.30)
- Perfect direction, our magnitude: 4.31 (−0.63)
- Smooth scale map σ=4: 3.92; σ=8: 4.51
- **Per-case SI-only scale** (channel 1, a≈1.2–2.2): **3.86** (−1.08)
- One global SI ×1.5 for all cases: 4.34 (−0.60). Chosen on DIR, so this is test-set tuning, not a result.
- Our SI p95 is ~4.5–6.9 vox on every case; Elastix ranges 6.4–13.9. The Decoder outputs roughly "average TCIA breathing" whatever the patient.
- Conclusion: the gap is per-patient SI amplitude. A single CT can't supply it; the patient's projections can.
- Script: `DIR EXPERIMENTS/scripts/oracle_mag_dir_decomp.py`; JSON: `arms/A3_synth_conditioned/results/oracle_mag_dir_decomp_tcia3_ep100.json`

# Voxel_GAN Diary

**Central lab notebook** (repo root). Newest entries at the **top** (prepend).  
Timestamps local (AEST / UTC+10).

---

## How to use

- **Prepend** new entries immediately below this section.
- Point to artifacts; keep failures honest.

---

# Timeline (newest first)

## 2026-09-24 ~10:47 — AmpHead restarted on GPU0

Earlier AmpHead jobs died (import clash, then killed when A3 moved to sequential GPU1). Restarted: `--gpu 0`, freeze TCIA3 ep100, precompute a* then 30 ep @ lr=1e-3.
Log: `TCIA3/AmpHead/logs/train_amp_head_gpu0_*.log`

---

## 2026-09-24 ~10:45 — A3×TCIA3: sequential on GPU1 only

Stopped parallel C02‖C03. Queue: **finish C03 → retrain C02 → C04,5,6,7,9,10** one-at-a-time on **GPU1**.
Script: `DIR EXPERIMENTS/arms/A3_synth_conditioned/logs/pipeline_a3_tcia3_seq_gpu1.sh`  
Log: `pipeline_a3_tcia3_seq_gpu1.log`

---

## 2026-09-24 ~10:39 — AmpHead phase-only: start train (freeze TCIA3 ep100)

### Spec
| | |
|--|--|
| **MLP** | `Linear(2→64)→ReLU→Linear(64→64)→ReLU→Linear(64→1)` |
| **Map** | \(a = 0.8 + 1.7\cdot\sigma(z)\) ∈ [0.8, 2.5] |
| **In** | \((\phi_\mathrm{ref}/9,\;\phi_\mathrm{tgt}/9)\) only |
| **Target** | \(a^\*=\mathrm{clip}(\mathrm{p95}_\mathrm{lung}\|u_\mathrm{El}\| / \mathrm{p95}\|û\|,\,0.8,\,2.5)\) |
| **G** | Frozen TCIA3 `epoch_100.pt` |
| **Data** | TCIA pooled pairs (not DIR) |
| **Epochs** | **30** |
| **LR** | **1e-3** Adam |
| **Batch** | 256 (on cached \(a^\*\) table) |

Script: `PopulationStudy/ClinicalExperiments/Grid160/TCIA3/scripts/train_amp_head.py`  
Out: `TCIA3/AmpHead/amp_head_best.pt`, `amp_head_loss.png`  
**GPU:** 1 (launched 2026-09-24 ~10:43; precompute \(a^\*\) then 30 ep)

### Prior: offline \(a\)-oracle (DIR, no train)
TCIA3 ep100 × scalar \(a\) sweep TRE75:
- a=1 → **4.94 mm**
- best-per-case → **4.18 mm** (−0.76 mm) — not full 1 mm / not ≤4 alone
- Artifact: `DIR EXPERIMENTS/arms/A3_synth_conditioned/results/amp_oracle_a_sweep_tcia3_ep100.json`

---

## 2026-09-24 — A3×TCIA3 status; TCIA3.1 oracle mid-train

- C01 TRE75 **2.13**; C08 TRE75 **11.82** (train done). Rest (C02+) relaunched after wiped scripts.
- TCIA3.1 @ ep35: synth-oracle TRE75 **5.39** (ep27 best) / **5.59** (ep35); still behind TCIA3 ep100 **4.94**.

---

*Last updated: 2026-09-24 (AmpHead launch + amp oracle).*
