# 2026-10-03 06:30 — Hybrid seed 2 PASSES; free checks (flip / weight-average / ensemble); two new lite runs queued

All TRE = plain STANDARD TRE300 via `rescore_oracle_iso2mm_v3.py --skip-oracle` (mm), unless "mirror". Scorer JSON `Grid160/TCIA3/DecoderCRB/plots/qc_dir_oracle/rescore_oracle_iso2mm_v3_20261003_055111.json`. Analysis scripts + JSON copied to `Grid160/TCIA_lite/analysis_2026-10-03/`.

**1. Hybrid seed 2 (`TCIA_lite/run_A2_full160_aug_hybrid_s2`, seed 20261002) — PASSED the fixed rule.** Finished 00:54 (40 ep, best val ep 19 = 0.501, val MAE flat 0.50–0.55 throughout, as in seed 1 and full160-aug).
- Last-5 (ep 36–40): s2 **4.163**, seed 1 4.174, full160-aug 4.470 → −0.31 mm. Ep 40 cases lower than full160-aug: **9/10** (C05 a near-tie 4.47 vs 4.49; C06 worse). Hybrid gain confirmed on two seeds.

| case | full160-aug | hybrid s1 | hybrid s2 | s2 − aug |
|---|---:|---:|---:|---:|
| C01 | 2.37 | 2.10 | 2.02 | −0.35 |
| C02 | 3.03 | 2.42 | 2.39 | −0.63 |
| C03 | 3.14 | 2.85 | 3.06 | −0.08 |
| C04 | 4.26 | 3.74 | 3.91 | −0.35 |
| C05 | 4.48 | 3.90 | 4.48 | 0.00 |
| C06 | 4.06 | 4.17 | 4.19 | +0.12 |
| C07 | 6.23 | 6.66 | 6.17 | −0.06 |
| C08 | 8.83 | 7.91 | 7.61 | −1.22 |
| C09 | 4.38 | 4.00 | 3.91 | −0.47 |
| C10 | 3.93 | 3.99 | 3.90 | −0.04 |
| **mean** | **4.47** | **4.17** | **4.16** | **−0.31** |
(per-case = mean over ep 36–40). Gains in both seeds: C01, C02, C04, C08, C09. C06 worse in both (~+0.1). C05 and C07 differ between seeds.

**2. Validation MAE does not track TRE in the lite runs.** s2 TRE fell 4.16 (ep 13) → 4.09 (ep 20) → 4.15 (ep 36–40) while val MAE never beat ep 1. `*best*` (val-picked) checkpoint = wrong one to score; keep scoring fixed epochs.

**3. TCIA3.5-hybrid (`Grid160/TCIA3.5_hybrid`, GPU0, running) — interim, same-epoch vs TCIA3.5:**
ep 10 4.192 vs 4.482 · ep 16 4.148 vs 4.338 · ep 22 4.128 vs 4.062 · ep 25 4.143 vs 4.124 · ep 46 **4.129 vs 4.353 (7/10 lower)**. TCIA3.5 (constant lr) swings ±0.15 between epochs, so single-epoch comparisons mislead; ep 22/25 "tie" was baseline noise. Hybrid itself is flat ~4.13 from ep 16. C08 not improved (9.1–9.4). Fixed rule (ep 96–100) still pending (~midnight 3 Oct).

**4. Free checks on saved checkpoints (no training; `analysis_2026-10-03/free_checks.py`):**

| model | plain | mirror |
|---|---:|---:|
| hybrid s1 ep40 | 4.171 | 4.133 |
| hybrid s2 ep40 | 4.172 | 4.145 |
| s1 weight-avg ep36–40 | 4.174 | 4.138 |
| s2 weight-avg ep36–40 | 4.162 | 4.135 |
| TCIA3.5-hybrid ep46 | 4.129 | 4.011 |
| ENS s1+s2 (avg weights) | 4.076 | 4.059 |
| **ENS s1+s2+TCIA3.5-hyb ep46** | **3.916** | **3.909** |

Earlier (2 Oct 21:53, `mirror_tta_iso2mm_20261002_215330.json`): TCIA3.5-hyb ep22 mirror 3.919 vs TCIA3.5 ep22 mirror 3.990; s2 ep28 4.193→4.160; full160-aug ep28 4.427→4.371.
- Mirror: −0.12 to −0.21 on crop-trained TCIA3.5-hybrid, only −0.03 on whole-volume lite models.
- Weight averaging of last 5 epochs: ~0 → dropped.
- **Ensembles now help** (earlier "ensembles ~0" was among models with shared errors): whole-volume lite + crop-trained TCIA3.5-hybrid differ enough → −0.21 vs best single. C08 worse in the 3-model ensemble (8.1 vs 7.6) because TCIA3.5-hyb is weak on C08. Best number so far **3.91** (interim, TCIA3.5-hyb not finished). Ensemble members chosen after seeing results → report as exploratory.

**5. Scale oracle on hybrid models (2 Oct 21:51, `scale_oracle_*_20261002_2151*.json`):** plain / one shared k / per-case lsq k:
s2 ep28 4.19 / 4.08 (k 1.26) / 3.81 · full160-aug ep28 4.43 / 4.45 (k 1.15) / 3.88 · TCIA3.5-hyb ep22 4.13 / 4.13 (k 1.20) / 3.61 · TCIA3.5 ep22 4.06 / 4.09 (k 1.20) / 3.58. C08 k_lsq ≈ 2.1 (TCIA3.5 models), 1.6 (s2). → Hybrid gain on lite is mostly NOT shape (lsq 3.88→3.81, −0.07) but plain −0.24; breath-depth ceiling unchanged (same finding as 29 Sep).

**6. Breath depth: training vs DIR-Lab (`analysis_2026-10-03/breath.py`).** CORRECTION during analysis: pair DVFs are in **2 mm voxels**, not mm (matches `TCIA_4D-Lung_dvf_characteristics`: S1 06→01 8.39 mm = 2 × 4.20). In mm: training 06→01 lung-mean |u| median 6.2, p90 9.3, max 12.3; p95 median 16.9, max 37.9. DIR-Lab landmark mean |T00→T50|: C01 3.9 … C08 15.0. Rank of DIR case in training distribution (mean): C01 13 %, C02 19 %, C03 63 %, C04 96 %, C05 71 %, C06 97 %, C07 97 %, **C08 100 %**, C09 76 %, C10 69 %. → Only C08 is beyond every training scan; C04/C06/C07 at the top 3–4 %. (First pass, before the unit fix, wrongly said 8/10 beyond every training scan.) Landmark vs lung-average measures differ (landmarks oversample moving regions) → rough comparison only. Lung volume / SI height / density vs motion: corr ≤ 0.13, LOO R² 0.16 (agrees with 29 Sep `amp_from_ct_check.py`).

**7. Label-stretch augmentation alone — NOT run.** Already reasoned out: amp4 (25 Sep, different model, broken input) and `ChangesNeeded.md` P1-B say amplitude aug only helps together with an amplitude input; clinic has a single planning CT (30 Sep). Without a depth input, stretching ≈ one shared k, which the scale oracle bounds at ~0–0.1 mm.

**8. Queued / running on GPU1 (lite, copies of hybrid seed 1, seed 20260929, one change each; rule: last-5 > 0.1 mm better than s1 4.174 AND ≥ 7/10 cases lower than s1 ep40):**
- `TCIA_lite/run_A2_full160_aug_hybrid_img30/` — **LAMBDA_IMG 10 → 30**. Gate passed (`--check-only`). Auto-started 00:54 after s2. Ep 17 4.173 vs s1 ep17 4.138 (level). Finishes ~12:30.
- `TCIA_lite/run_A2_full160_aug_hybrid_extreme/` — **deepest-breath label**: for each train scan, the (5→0)/(0→5) slots use that scan's largest-motion pair (`TCIA_4D-Lung_dvf_characteristics/metrics_per_scan.tsv` max_pair; inhale = member cyclically closer to 01). Max pair is 01↔06 in 48/82 scans; 29/65 train scans changed; val unchanged. Why: DIR-Lab T00/T50 are true extremes, TCIA 06/01 often are not (29 Sep note). Gate passed; label override checked on S05 (06_to_01 slot → 07_to_01, labels 5→0). Auto-starts after img30 (`logs/start_after_img30.sh`), ~11 h.

**Target 3.8.** Single-CT literature/oracle floor ≈ 3.9–4.0 (29 Sep, 30 Sep). Current best 3.91 (3-model ensemble). Remaining levers: TCIA3.5-hyb final epochs, extreme-pair run, img30.

---

# 2026-10-02 11:50 — A1 / A2 TRE re-scored with an exact inverse (old T00→T50 numbers were too high)

**Audit of `DIR EXPERIMENTS/scripts/eval_a1_tre.py` + `dirlab_tre.py`.**
- OK: official → pack z-flip is present (the step the old synthesiser scorer v1 skipped); pack → R3 → sub_128 chain and its reverse are consistent; displacements scaled per axis back to pack voxels.
- OK: T50→T00 = lm50 + disp(lm50). The Elastix field (fixed = T50 = phase 06) lives on the T50 grid → exact.
- **FLAW: T00→T50 (the headline) = lm00 − disp(lm00)** — the T50-grid field sampled at T00 points = first-order inverse. Error grows with motion size; inflates large-motion cases. Applies to Elastix and VoxelMap alike, in A1 and A2. Synthetic test: old method off by up to 0.84 vox, exact ~1e-14.
- Minor, left unchanged: pack → sub uses s = x·128/N (no half-voxel term); only shifts where the smooth field is read.
- Synthesiser scores (4.17, 4.00, …) are NOT affected: `eval_dir_tcia3_iso2mm_v2.score_set` samples the T00-grid pull field at T00 points (exact).

**Fix (new scripts; originals untouched).** `scripts/eval_a1_tre_v2.py` (A1) and `scripts/eval_a1_a2_tre_v2.py` (A1 + A2, `--arms`). Solves v(q) = −disp(q + v(q)) by fixed-point iteration at each landmark on the 128³ grid; frame (native/r3) read from each case's `tre_summary.json`; uses the saved fields only (no GPU, no retraining). Self-checks: (1) T50→T00 reproduces the saved values exactly; (2) Elastix inverse round trip < 0.01 sub-voxel. Both passed for A1 and A2; no case skipped. Note: the A1+A2 version of `eval_a1_tre_v2.py` was overwritten on disk by the A1-only version before running, so A2 was run with `eval_a1_a2_tre_v2.py --arms A2`.

**A1 (oracle, patient's own 4D-CT), T00→T50 TRE75 / TRE300, mm:**

| case | Elastix old | **Elastix exact** | VoxelMap old | **VoxelMap exact** |
|---|---:|---:|---:|---:|
| C01 | 1.26 / 1.13 | 1.14 / 1.06 | 1.40 / 1.29 | 1.31 / 1.25 |
| C02 | 1.08 / 1.05 | 0.92 / 0.95 | 1.40 / 1.31 | 1.37 / 1.34 |
| C03 | 1.30 / 1.30 | 1.22 / 1.23 | 1.44 / 1.40 | 1.39 / 1.35 |
| C04 | 1.91 / 1.79 | 1.49 / 1.44 | 1.88 / 1.77 | 1.53 / 1.48 |
| C05 | 2.02 / 1.88 | 1.83 / 1.65 | 2.19 / 2.09 | 2.04 / 1.91 |
| C06 | 2.90 / 2.58 | 1.57 / 1.61 | 3.56 / 3.47 | 2.66 / 2.82 |
| C07 | 2.34 / 2.39 | 1.89 / 1.93 | 2.63 / 2.83 | 2.32 / 2.53 |
| C08 | 3.77 / 3.32 | 1.98 / 1.67 | 4.48 / 4.04 | 3.23 / 2.81 |
| C09 | 2.05 / 1.96 | 1.82 / 1.69 | 2.08 / 1.96 | 1.88 / 1.74 |
| C10 | 2.18 / 2.18 | 1.63 / 1.66 | 2.12 / 2.14 | 1.76 / 1.76 |
| **mean** | 2.08 / 1.96 | **1.55 / 1.49** | 2.32 / 2.23 | **1.95 / 1.90** |

T50→T00 (exact all along): Elastix 1.64 / 1.57, VoxelMap 2.02 / 1.97. JSON `arms/A1_oracle_dirlab/results/cohort_tre_v2_exact_inverse.json`, log `logs/eval_a1_a2_tre_v2.log`.

**A2 (SPARE population prior), VoxelMap exact T00→T50 TRE75 / TRE300:** C01 3.85/3.78, C02 4.64/4.36, C03 6.79/6.62, C04 7.90/8.14, C05 6.16/6.08, C06 9.27/8.53, C07 8.67/8.88, C08 13.78/13.00, C09 6.63/7.24, C10 5.33/4.63. **Mean 7.30 / 7.12** (old 7.27 / 7.09; +0.04 — A2's arrows are small, so the approximation hardly mattered). Elastix rows identical to A1. JSON `arms/A2_generic_spare/results/cohort_tre_v2_exact_inverse.json`.

**A2 checkpoint — PROVISIONAL.** Symlink `arms/A2_generic_spare/checkpoints/a2_spare_mc_val_prior_p1to9_concat_nofilm.pt` (16 Sep 10:26) points into `arms/Incorrect DRR/A2_generic_spare/checkpoints/`; the target file is not on disk, `Incorrect DRR/` has no A2 folder (only A1 plots), and no A2 weights exist anywhere else. So A2 cannot be re-run or checked, and was probably trained on the DRRs later found to be wrong. Saved fields exist (hence re-scoring worked). Decision: label A2 provisional; decide on retraining after A3 is redone (if A3 is clearly better than A2 the conclusion holds; if close, A2 must be retrained on the correct DRRs first).

**Corrected arm table (T00→T50, exact):** A0 identity 8.69 / 8.46 · **A1 Elastix 1.55 / 1.49** · **A1 VoxelMap 1.95 / 1.90** · **A2 VoxelMap 7.30 / 7.12 (provisional)** · A3 to be redone with the final synthesiser. Old Results.md values (2.08 / 3.16 / 8.97) superseded; Results.md updated, previous copy kept as `Results_pre_2026-10-02.md`.

**Note:** the 2026-10-02 11:20 hybrid entry below was found missing from Diary.md (file had been reverted to the 22:40 version) and was re-added unchanged.

---

# 2026-10-02 11:20 — Hybrid loss (image match): FIRST CHANGE TO PASS the fixed rule

**Idea (from literature, see 2026-10-01 22:40 §6).** Every method change so far gave < 0.1 mm; measured causes left are (a) breath depth invisible in one CT and (b) the model copying Elastix's own errors. Hybrid loss targets (b): grade the model also on whether the CT actually lines up, a signal that does not share Elastix's mistakes.

**Code (new copy; original untouched).** `Grid160/TCIA_lite/run_A2_full160_aug_hybrid/scripts/train_aug_hybrid.py`, copy of full160-aug (`run_A2_full160_aug/scripts/train_aug.py`). Same data (`data_allscans` manifest, 6500/1700 pairs), net (UNetCRBDecoder 1.07M), augmentation, cosine lr 1e-4→1e-6, 40 epochs, seed 20260929. ONE change, the loss:

    loss = L_dvf + 10 × L_img + 0.1 × L_smooth
    L_dvf    = lung-masked L1 |label − pred| (= DVFLoss, unchanged)
    L_img    = mean |target CT − warp(ref_geo, pred)| over lung mask dilated 2 vox (5³ max-pool)
    L_smooth = mean of squared neighbour differences of pred along z, y, x (whole volume)

- `ref_geo` = input CT after the SAME flip/shift/zoom as the target but BEFORE brightness/noise aug (target never gets brightness aug), so the image term compares like with like. The network still sees the augmented CT.
- warp = pull convention ref(p + u(p)), same as labels. Dilated mask so lung edges (diaphragm) count.
- **Weights 10 and 0.1 fixed before training, never tuned.** Chosen by size balance: image L1 ≈ 0.02 → ×10 ≈ 0.2 ≈ DVF L1 ≈ 0.25; smoothness is a light touch. Not tuned on DIR because DIR is the test (tuning on it inflates the result). They are an educated guess, almost certainly not optimal. Fair tuning, if ever: try ×3/×10/×30 on a different landmark set (e.g. POPI), pick there, score DIR once. Validation loss cannot tune it (it measures agreement with Elastix, which the image term deliberately moves away from).
- Logged "train MAE" = L_dvf only (comparable with full160-aug); a "parts" line shows all three terms each epoch.
- Gate before training (stops if any fails): (a) lambdas = 0 gives exactly DVFLoss; (b) Elastix label beats zero motion on the image term without aug and with forced flip/shift/zoom; (c) extra terms give a non-zero gradient. Synthetic test (shifted Gaussian blob): all pass; wrong-sign arrows score 0.083 vs 0.0 for the right ones; image term alone pulls a zero guess toward the true shift (0 → 0.68 of 2.0 in 200 steps). Gradient check first gave 0 because it started exactly at the label (L1 minimum) — fixed by adding noise to the start point.

**Rule (fixed before results).** Last-5 (ep 36–40) plain STANDARD TRE300 > 0.1 mm better than full160-aug 4.470 (i.e. ≤ 4.37) AND ≥ 7/10 cases lower.

**Results (STANDARD TRE300, mm).** Epochs 1–30 plain:

| ep | TRE | C08 | ep | TRE | C08 | ep | TRE | C08 |
|---|---:|---:|---|---:|---:|---|---:|---:|
| 1 | 5.730 | 11.78 | 11 | 4.291 | 8.57 | 21 | 4.318 | 8.30 |
| 2 | 4.354 | 7.67 | 12 | 4.369 | 8.38 | 22 | 4.088 | 7.95 |
| 3 | 4.403 | 8.85 | 13 | 4.402 | 8.67 | 23 | 4.257 | 7.95 |
| 4 | 4.464 | 9.08 | 14 | 4.401 | 8.80 | 24 | 4.136 | 8.05 |
| 5 | 4.196 | 8.54 | 15 | 4.005 | 7.80 | 25 | 4.025 | 7.56 |
| 6 | 4.531 | 9.16 | 16 | 4.294 | 8.45 | 26 | 4.192 | 8.05 |
| 7 | 4.192 | 7.82 | 17 | 4.138 | 8.02 | 27 | 4.145 | 7.84 |
| 8 | 4.282 | 8.41 | 18 | 4.173 | 8.10 | 28 | 4.082 | 7.70 |
| 9 | 4.176 | 8.45 | 19 | 4.031 | 7.77 | 29 | 4.213 | 7.99 |
| 10 | 4.219 | 8.47 | 20 | 4.178 | 7.80 | 30 | 4.213 | 8.11 |

Epochs 21–30 mean 4.17 (push run in the same stretch ≈ 4.38). Ep 15 (4.005; C01 2.01, C02 2.52, C03 3.00, C04 2.97, C05 3.86, C06 3.94, C07 6.22, C08 7.80, C09 3.90, C10 3.84) is an unofficial best-seen only.

Final epochs:

| epoch | plain | mirror | C08 plain | C08 mirror |
|---|---:|---:|---:|---:|
| 36 | 4.175 | 4.141 | 7.92 | 7.80 |
| 37 | 4.187 | 4.150 | 7.94 | 7.82 |
| 38 | 4.167 | 4.130 | 7.87 | 7.74 |
| 39 | 4.173 | 4.138 | 7.94 | 7.81 |
| 40 | 4.171 | 4.133 | 7.90 | 7.78 |
| **last-5** | **4.174** | **4.138** | | |

Per case ep 40 plain, hybrid vs full160-aug: C01 2.09/2.37, C02 2.43/3.02, C03 2.85/3.14, C04 3.73/4.29, C05 3.90/4.49, C06 4.19/4.05, C07 6.66/6.24, C08 7.90/8.84, C09 3.97/4.37, C10 4.00/3.93 → **7/10 lower**. Ep 40 mirror: C07 6.63, C08 7.78.

**VERDICT: PASSES.** −0.296 mm plain (3× the bar), last 5 epochs very stable (4.167–4.187). First change in the project to clear the fixed rule.

**Caveats.**
- One seed. 0.3 mm is large vs the ~±0.1 seed swing, but 7/10 cases is exactly on the line → seed 2 needed before building on it.
- C07 worse again (6.24 → 6.66), as with push. C06, C10 slightly worse. C07 resists both changes — worth a look.
- C08 (−0.94) is the largest contributor; without it the mean gain would be ~0.2 mm.
- Mirror adds only −0.036 here. Comparison table (plain / mirror): full160-aug 4.47 / ~4.40 · **hybrid 4.17 / 4.14** · TCIA3.5 4.18 / **4.00**. Hybrid ties TCIA3.5 plain with < half the training, but TCIA3.5 + mirror is still the best number (mirror helps the crop-trained model −0.18 vs −0.04 for whole-view models).

**Why val and train don't fall together (notes from discussion).** They do fall together for ~10 epochs while the general motion rule is learned. After that, what is left is pair-specific (Elastix's errors in that exact label, that scan's exact breath depth), which lowers train only. Val MAE measures agreement with Elastix, not truth, so with the hybrid loss val can stay flat or worsen while DIR improves. DIR is the only score that counts.

**TCIA3.5 vs TCIA-lite (for the write-up).** Same net and base loss. TCIA3.5: 64³ lung-biased crops ×16 per pair, `pooled` manifest 82 scans, 7380/820 pairs (phase-pair split inside scans), 100 epochs, Adam 1e-4 constant, no aug, seed 20260918, ~34 h. Lite full160-aug: whole 160³ ×1 per pair, `data_allscans` 6500/1700 pairs (split method not yet checked), 40 epochs, cosine 1e-4→1e-6, flip/shift/zoom/brightness/noise aug, seed 20260929. TCIA3.5 sees ~3× more data in volume terms (118 080 crops/epoch ≈ 7 400 volumes × 100 ≈ 740k vs 6 500 × 40 = 260k); crops act like augmentation; mirror helps it more.

**Plan.**
1. Seed 2 of hybrid (same script, new folder, `--seed 20261002`), same rule. Confirms the gain.
2. Hybrid on the TCIA3.5 setup (82 scans, crops, 100 epochs, ~35 h) on GPU 0 when free — best shot. Honest guess 3.85–4.0 with mirror, uncertain.
3. Optional, after 1: fair weight tuning on POPI (×3/×10/×30), DIR scored once.
4. Ceiling: breath depth (~0.6 mm, mostly C08) cannot be removed from one planning CT. 3.9–4.0 looks reachable; 3.8 maybe 30–40%. A measured floor is a defensible thesis result.
- TCIA3.5-push v2 keeps running to epoch 100 (comparison at ep 92 and 96–100); push alone did not pass, so it can be stopped early if GPU 0 is needed.

---

# 2026-10-01 22:40 — Push verdict, TCIA3.5-push crop bug, val-loss diagnosis, brightness ruled out

**1. Full-volume push run (TCIA_lite/run_A2_full160_aug_push) — NOT KEPT.**
Rule fixed before results: last-5 mean (ep 36–40) > 0.1 mm better than full160-aug (4.470) AND ≥ 7/10 cases better.

| epoch | plain | mirror | C07 | C08 |
|---|---:|---:|---:|---:|
| 36 | 4.376 | 4.361 | 6.76 | 8.24 |
| 37 | 4.381 | 4.364 | 6.79 | 8.23 |
| 38 | 4.382 | 4.366 | 6.79 | 8.17 |
| 39 | 4.369 | 4.356 | 6.76 | 8.15 |
| 40 | 4.375 | 4.359 | 6.79 | 8.17 |
| **last-5** | **4.377** | **4.361** | | |

Gain 0.093 mm (bar is > 0.1), 6/10 cases better at ep 40. Per case push vs aug (ep 40): C01 2.08/2.37, C02 2.64/3.02, C03 3.67/3.14, C04 3.62/4.29, C05 3.88/4.49, C06 5.35/4.05, C07 6.79/6.24, C08 8.17/8.84, C09 3.52/4.37, C10 4.03/3.93.
- Push moves error between cases rather than lowering it: C08 −0.67, C09 −0.85, C04 −0.67, but C06 +1.30, C07 +0.55, C03 +0.53. Prediction "C07 and C08 improve most" is half falsified (C07 got worse).
- Epoch 7 (3.993) remains an unofficial best-seen only (choosing it = picking by DIR). Epochs 15–27 sat at 4.24–4.50.
- SWA skipped for this run: epochs 36–40 differ by only 0.013 mm, so averaging near-identical weights cannot change much.
- **Lesson:** I oversold push from matched epochs 6–11 (4.31 vs 4.43). Early-epoch gaps of ~0.1 mm are inside the ±0.1 mm epoch-to-epoch swing. From now on every expected gain is given as a range with the noise, and nothing is called a win before the fixed rule.

**2. Bug in TCIA3.5_push (v1) — my error. Run stopped, 16 epochs (~5.5 h GPU) wasted.**
- Symptom: train MAE flat at ~0.59 for 16 epochs (original TCIA3.5: 0.48 → 0.31 by epoch 10; predicting zero motion = 0.685). DIR TRE ~6.9.
- Cause: the dataset subclass called `_crop_params()` a second time to cut the label. With `random_crop=True` that draws a NEW random crop → CT crop and label crop came from different places. Unlearnable.
- Why the full-volume push run was fine: whole 160³ view, crop position is fixed, both draws agree.
- Why the check missed it: `verify()` built its dataset with `random_crop=False` (fixed crops agree).
- Fix: `Grid160/TCIA3.5_push_v2/scripts/train_mae_noaug_push_v2.py` (v1 left untouched). Reuses the crop drawn by the base `__getitem__` (stored in an overridden `_crop_params`), raises if the crop is drawn ≠ 1 time, handles both label layouts (3,Z,Y,X) and (Z,Y,X,3), and the check now runs with fixed AND random crops. Unit test with a dummy label and 5 random crops: label crop == CT crop every time. Writes only to `TCIA3.5_push_v2/`.
- v2 early scores (STANDARD TRE300, plain/mirror): ep1 4.757/4.832, ep2 4.575/4.537, ep3 4.313/4.197, ep4 4.834/4.710, ep5 4.718/4.642, ep6 4.500/4.281, ep7 4.646/4.566, ep8 4.380/4.253, ep9 4.584/4.486, ep10 4.472/4.406, ep11 4.443/4.314, ep12 4.356/4.313, ep13 4.447/4.523. Ep 9–13 mean 4.46 plain / 4.41 mirror. C08 8.72–11.0. Mirror helps more here (~0.08) than in the full-volume run (~0.03). Comparison still fixed: epoch 92 and last-5 (96–100) vs TCIA3.5 4.18 plain / 4.00 mirror.

**3. TCIA3.5 validation split (corrected wording).** Validation is split by **phase pair inside every scan**, not by patient: 7380 train / 820 val pairs, all 82 scans and all phases appear in training. Validation only hides some r→g combinations of lungs already trained on. So validation loss is optimistic and must not pick epochs; DIR-Lab (patients never seen) stays the only fair test.

**4. Why TCIA3.5 val loss stops falling (diagnosed).**
- `label_quality/label_noise_floor.py` (new, reads labels only). Inverse consistency of r→g then g→r on 120 val pairs, in DVFLoss units (lung-masked L1 per component, iso-voxels; Experiment6 and Grid160/Experiment1 losses.py confirmed identical): zero-motion MAE 0.685, ICE 0.234, **label-noise floor ≈ 0.165** (ICE/√2, a lower bound — shared bias is invisible), rough breath-depth floor 0.130, combined ≈ 0.21–0.30.
- TCIA3.5 log: train MAE 0.248 (ep 90–100 mean) — at the floor; val MAE 0.355 (best 0.351 at ep 92). Gap grows 0.06 (ep 5) → 0.08 (ep 10) → 0.10 (ep 40) → 0.107 (ep 90–100). Val drops fast to ep ~10, then only ~0.03 over 90 epochs.
- DIR-Lab over training (old scorer, trend only): ep15 5.17, ep76 5.03, ep88 5.01 — never gets worse.
- **Conclusion:** not underfitting (train at the noise floor), not classic overfitting (test never worsens). A **generalisation gap**: the model learns the general motion in ~10 epochs; later learning is pair-specific and neither helps nor hurts DIR. My first claim ("val is at the noise floor") was wrong — val is ~2× the measured floor.
- Converged: train 0.2490 → 0.2481 over ep 90–100. Global optimum cannot be proven, but train is already at the label floor and ensembles gave no gain (runs land in equally good basins).

**5. Brightness (intensity normalisation) — ruled out.**
- Both sides use the same per-scan µ min-max (training `_load_ct`; scorer v2 `minmax(hu_to_mu)`), confirmed in code.
- `label_quality/brightness_check_all.py` (new): 40 TCIA scans via TCIA3.5's own dataset class vs all 10 DIR cases via the scorer's own path. TCIA lung median 0.063–0.319 (median 0.176), tissue 0.283–0.497 (median 0.405). Every DIR case is darker: lung 0.049–0.105, tissue 0.254–0.367. Flagged: C04–C10; only C01–C03 inside (and those are DIR-Lab's easy small-motion cases anyway). Raw µ max 0.055–0.080 for all, so no single bright voxel sets the scale.
- `label_quality/brightness_match_test.py` (new, no training): straight-line remap of each DIR case so lung → 0.176 and tissue → 0.405, fed straight to the net (bypassing the scorer's re-min-max). Self-check (own path == v2.predict within 0.01 mm) passed; remap hit both targets on every case. Rule fixed in advance (both models > 0.1 mm better and ≥ 7/10).
  - TCIA3.5 ep92: plain 4.181 → 4.568, mirror 3.999 → 4.482, 2/10 better (C08 9.51 → 9.07, C09 4.28 → 3.89, C07 5.59 → 6.40).
  - full160-aug ep40: plain 4.475 → 4.473, mirror 4.402 → 4.376, 5/10 better.
  - **NOT KEPT.** Darker DIR images are not what limits the models. Guess (untested): the ~2× contrast stretch amplifies noise/texture that the crop-trained TCIA3.5 relies on; the whole-view model is robust.

**6. Literature (for next steps).** RMSim (static CT → motion, conditioned on a 1D breathing amplitude, loss = image similarity on the warped image + DVF smoothness, not DVF labels alone); noisy-label regularisation (early-learning then memorisation pattern; SWA, ELR, augmentation); SynthMorph (synthetic deformations). See chat for links.

**Scoreboard of ideas.** Push −0.09 (not kept) · ensembles ~0 · mirror −0.02 to −0.08 · relabel not adopted · brightness not the cause · underfitting no. Every method change < 0.1 mm → the remaining error is about information, not training details: (a) breath depth invisible in one CT (~0.6 mm, mostly C08); (b) the model copies Elastix's own errors.

**Next.** Hybrid loss as a new copy of full160-aug: DVF L1 + image similarity (warp CT_r → CT_g) + smoothness. Expected 0 to −0.2 mm, uncertain. Same fixed rule. TCIA3.5-push v2 keeps running to epoch 100 (comparison at ep 92 and 96–100).

---

# 2026-10-01 11:00 — "Arrows on the input CT" (push-style labels): hypothesis, code, first results

**Hypothesis.** All models fail the same way on deep breathers (C08 ≈ 9 mm, C07 ≈ 6–7 mm in every run, and averaging models does not help). Cause proposed: the labels are *pull* fields on the **target** grid (target(x) = ref(x + u(x)); Elastix fixed = target). The network sees only the input CT (T50 / phase 06) but must output the field on the T00 grid. Near the diaphragm a breathed-in lung point sits up to ~15 mm lower, where the input CT shows liver, so the network is asked to describe lung motion while looking at liver. Error should grow with breath depth.

**Change (no new registrations).** Use the field that sits on the input CT: for input CT_r with phase code r→g, label = file `{scan}_{g}_to_{r}_pair.npy` (Elastix fixed = r) instead of `{scan}_{r}_to_{g}_pair.npy`. Same 100 files per scan, same CTs, same phase codes. For input CT 06: `01_to_06`, `02_to_06`, …, `10_to_06`.

**Predictions (written before results).** C08 and C07 improve most; shallow cases (C01, C02) barely change. Falsified if C07/C08 stay ≈ 9 / 6.5 mm.

**Code (all new files; originals untouched).**
- `TCIA_lite/run_A2_full160_aug_push/scripts/train_aug_push.py` — copy of full160-aug (whole 160³, variety, cosine lr 1e-4→1e-6, 40 epochs, seed 20260929); only the label file and the safety-check direction changed (checks |ref − warp(tgt, v)|). Swap tested with stand-in files (input CT 06, code 06→01, label from `01_to_06`).
- `Grid160/TCIA3.5_push/scripts/train_mae_noaug_push.py` — copy of TCIA3.5 (82 scans, 64³ crops ×16, 100 epochs, Adam 1e-4, seed 20260918); diff vs original = header, names, read-only paths, write guard, cfg only. Built-in `--check-only`: every swapped file exists, shapes match, and new labels align CT_g onto CT_r better than no motion. Stand-in tests: correct swap passes (0.0003 vs 0.017), forgotten swap fails (0.048), missing file fails and names it. **Not started yet** (needs `--check-only` then launch on GPU 0; ≈ 34–36 h, TCIA3.5 took 34 h 17 min).
- `DIR EXPERIMENTS/scripts/score_push_iso2mm_v2.py` — scores push-style models two ways. **STANDARD** (headline): invert v to the usual T00-grid field (fixed-point, 25 iterations; synthetic round trip 0.002 vox) and score with the unchanged verified scorer → comparable with every earlier number. **DIRECT** (secondary): move T50 landmarks, error measured in T00 space.
- Scorer v1 self-test failed by design error, not a bug: on Elastix's own fields STANDARD = 1.956 vs reference 1.957 (exact), DIRECT = 2.116 (+0.159, largest C06 +0.43, C08 +0.24). DIRECT measures the error in the other breathing phase, where the lung has a different volume, so it is not expected to match. v2 self-test requires only STANDARD within 0.05 mm.

**TCIA-lite push, early epochs (STANDARD TRE300, mm).**

| epoch | plain | mirror | C07 | C08 |
|---|---:|---:|---:|---:|
| 5 | 4.110 | 4.086 | 6.14 | 8.86 |
| 6 | 4.320 | 4.335 | 6.45 | 8.85 |
| 7 | 3.993 | 4.006 | 5.91 | 7.92 |
| 8 | 4.306 | 4.281 | 6.29 | 8.54 |
| 9 | 4.283 | 4.265 | 6.06 | 8.54 |
| 10 | 4.444 | 4.431 | 6.58 | 9.18 |
| 11 | 4.544 | 4.494 | 6.81 | 8.69 |

Epoch 7 per case: C01 2.71, C02 2.37, C03 2.90, C04 2.63, C05 3.77, C06 4.55, C07 5.91, C08 7.92, C09 3.55, C10 3.62. Epoch 7 is a single swing and is not a result (it would be picking by DIR).

**Matched-epoch comparison with the old-arrow run (full160-aug, same recipe):** epochs 6–11 mean TRE300 4.31 (push) vs 4.43 (old); mean C08 8.67 vs 9.27. Direction as predicted, smaller than hoped. Same epoch-to-epoch wobble (~0.3 mm) as every run.

**Rules fixed now.**
- TCIA-lite push vs full160-aug: last-5 (36–40) mean, > 0.1 mm better AND ≥ 7/10 cases (full160-aug last-5 = 4.47).
- TCIA3.5-push vs TCIA3.5: same fixed epochs for both, **epoch 92 and last-5 (96–100)**, plain and mirror. Not each run's best-validation epoch, because TCIA3.5's validation is a phase-pair split inside every scan (all patients and phases are in training; see 2026-10-01 22:40), so its validation loss is optimistic. DIR stays fair (DIR patients never in training). TCIA3.5's own last-5 must be computed for this.
- Proposed, not yet adopted: weight averaging of epochs 31–40 for both lite runs, to be fixed before those epochs exist.

**Target arithmetic.** Mean over 10 cases: each 1 mm on C08 moves the mean 0.1 mm. From TCIA3.5 + mirror (4.00, C08 8.94), 3.80 needs −2.0 mm summed, e.g. C08 → ~6.9 alone, or C08 → 7.5 and C07 → 5.7.

# 2026-10-01 07:00 — Ensembles + mirror averaging: no gain (`ensemble_eval_iso2mm.py`)

Groups fixed in the script before results (final epochs by rule). Reference TCIA3.5 ep92 + mirror = 4.00.

| model / group | variant | TRE300 | C08 | vs 4.00 | cases better |
|---|---|---:|---:|---:|---:|
| TCIA3.5 ep92 | mirror | **4.00** | 8.94 | ref | – |
| E_all (TCIA3, TCIA3.5, full160-aug, full160, deep128) | mirror | 4.02 | 9.07 | +0.02 | 4/10 |
| E_full (TCIA3 + TCIA3.5) | mirror | 4.05 | 9.44 | +0.05 | 2/10 |
| E_all | plain | 4.07 | 9.04 | +0.07 | 4/10 |
| E_lite (full160-aug, full160, deep128) | mirror | 4.13 | 8.87 | +0.13 | 3/10 |
| TCIA3.5 ep92 | plain | 4.18 | 9.51 | +0.18 | 2/10 |
| TCIA3 ep100 | plain / mirror | 4.25 / 4.25 | 9.15 / 10.01 | +0.25 | 2/10 |
| SNAP_aug (full160-aug ep36–40) | mirror | 4.40 | 8.95 | +0.40 | 2/10 |

JSON: `.../qc_dir_oracle/ensemble_eval_iso2mm_20261001_065928.json`.
**Reading.** Averaging only removes errors that differ between models; all models share the same errors (C08 short in every model) → the remaining error is systematic (breath depth + shared TCIA→DIR gap), not noise. E_all + mirror 4.02 is a more robust estimate of the same level as the single-epoch 4.00. → motivated the push-label hypothesis above.

# 2026-10-01 06:50 — Full-resolution relabel: built, run, and not adopted

**DIR-Lab evidence that started it** (`DIR EXPERIMENTS/label_quality/`, raw DIR-Lab images, label Elastix settings `My v1.0/configs/elastix_bspline_masked.txt`, TRE300 mm):

| method | mean (10) | C01 | C08 |
|---|---:|---:|---:|
| current labels (Elastix on 2 mm grid) | 1.96 | 1.13 | 3.32 |
| same Elastix, original resolution, direct | **1.46** | 1.07 | 1.96 |
| original resolution, chained T00→T20→T50 | 1.81 | 1.52 | 2.99 |
| deeds (default, original resolution) | 1.37 | 0.99 | 1.55 |

- Full resolution gains ~0.5 mm on DIR-Lab; deeds adds only 0.097 mm (7/10 cases; below the 0.1 bar) → keep Elastix. Chaining through a middle phase is worse on all 10 cases (+0.35 mean) → register every pair directly (7,380 registrations).
- Preparation matters: the same Elastix on processed DIR volumes (yesterday's bench) gave C08 2.96 vs 1.96 on raw images.

**Pipeline** (`DIR EXPERIMENTS/relabel/`): per-unit worker (scan × target phase), scheduler with auto core/RAM sizing (8 cores reserved for GPU loaders), RAM/disk guard, timeouts, retries, resume, atomic writes, per-unit scratch folders; watcher with stale-heartbeat restart and leftover-process cleanup; `check_adapter.py` gate. Tested on synthetic scans with crash/hang injection and hard kills (all outputs valid). Adapter `source_tcia.py` written by the training session (native `volumes/S*/GTVol_XX.mha`, grid from `packed_iso/S*/pack_meta.json`, R3 mapping `transpose(1,0,2)[::-1,::-1]`, channels (dx, −dz, −dy), /2 mm).

**What happened.** The first gate run FAILED 3/6 pairs (size ratio 0.53–0.70 vs old labels over the old-label motion region); the thresholds were then loosened (0.8→0.7, 0.6→0.5) and the full run produced 8,200 files in `synth_g160_r3/pooled_native_f16`. Lesson: gate thresholds must not be changed after seeing a failure.

**Independent check** (`relabel/verify_new_labels.py`, 100 pairs / 20 scans, warp reference CT with each label, compare with target inside the lung):

| | no motion | old labels | new labels |
|---|---:|---:|---:|
| error (lower better) | 0.060 | **0.023** | 0.028 |
| match NCC (higher better) | 0.927 | **0.993** | 0.984 |

New labels better on 0/100 pairs. Inside the lung new and old are the same size (ratio 1.00) and direction (agreement 0.91); the "40 % smaller" was outside the lung.
**Caveat.** This test favours the old labels (they were optimised on exactly these 2 mm images). It does not prove the new ones are less accurate anatomically. But inside the lung the two label sets are so similar that the DIR-Lab 0.5 mm label gain does not carry over to TCIA; expected model gain ≤ ~0.1 mm. **Decision: not used.** My earlier "0.6 mm" framing was label accuracy on DIR-Lab, not model gain, and was overstated.

# 2026-10-01 06:43 — full160-aug (whole lung + variety + cosine lr) final: not kept

Last-5 (36–40): 4.47, 4.48, 4.46, 4.47, 4.48 → **4.47 mm**; C08 last-5 8.83. vs full160 (4.58): −0.11 mm but 4/10 cases better → fails the 7/10 part. Very stable ending (epochs 32–40 all 4.46–4.49); best last-5 of all TCIA-lite runs. Best single epoch 10 (4.32). Validation at epoch 7: train 0.399 vs full160 0.356 (variety slows memorising), val 0.51–0.55 vs 0.52–0.59.
Script: `run_A2_full160_aug/scripts/train_aug.py` (flip with x negation, shift ±8, zoom 0.92–1.08 with DVF×s, brightness/contrast/gamma, noise; consistency check none/flip/shift 0.0032, zoom 0.0026, identity 0.0080).

# 2026-09-30 14:10 — TCIA-lite final evaluation (`final_eval_tcia_lite.py`; JSON `TCIA_lite/final_eval_20260930_140158.json`)

| run | epochs | last-5 TRE300 | C08 | vs A′ | cases better (final) | pattern-only lsq | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| TCIA3 ep100 (ref) | 100 | 4.25 | 9.15 | | | | reference |
| A′ (64³) | 76–80 | **4.74** | 9.22 | | | 4.18 | baseline |
| A′-small (half width, 0.27M) | 76–80 | 4.84 | 9.91 | +0.11 | 5/10 | 4.07 | worse |
| A′-deep128 (+1 level, 128³) | 76–80 | 4.59 | 9.71 | −0.15 | 7/10 | 3.90 | better |
| A′-full160 (whole 160³) | 36–40 | 4.58 | 8.68 | −0.16 | 7/10 | 3.95 | better |

- Seeing more of the lung helps the motion pattern (lsq 4.18 → 3.90–3.95); breath-depth error (~0.6 mm) unchanged. full160 = deep128 without the extra layer, and better on C08 → keep whole-lung view.
- Caveat: A′ ran 80 epochs and drifted up late; part of the margin is A′ getting worse.
- Small model: train/val gap smaller (ep20 0.13 vs 0.17) but val worse (≈0.51 vs 0.49) → learns less; 1.07M is the right size. A′ val flat 0.48–0.51 from epoch 5 to 80 while train fell 0.41→0.26: everything after ~epoch 5 is memorising.
- full160 timeline: epoch 5 = 4.10 (C08 7.47), then memorised back to 4.58; val best at epoch 1.
- A-EPE (stopped at ep 21): epochs 16–20 mean 4.40 vs run A 4.46 → +0.06, under 0.1 → **keep L1**.
- Pattern-only (shape) loss: built (k = L1-optimal per pair, weighted median, clamp [0.25, 4], identity k = 1, no gradient; 9 checks incl. broken-version controls all behave) but the run died at start (no epoch); deprioritised because on training pairs k ≈ 1 (memorised size) and, with no breathing signal in the clinic, a size-free model has no use. Files `run_A2_full160_shape/`.

# 2026-09-30 15:57 — Clinical constraint: only a single 3D planning CT

At treatment the pipeline has **only a single 3D planning CT** (no 4DCT, no CBCT/kV projections, no breathing trace). Consequences: no breathing signal → breath depth must stay "typical breath" (report a shallow–deep range); the Amsterdam Shroud and "with breathing signal" numbers are future work only. Open check: planning CT acquisition (free-breathing / breath-hold / exhale) vs training input (exhale phase); a blurred mean-phase input test was proposed.

# 2026-09-30 15:30 — Amsterdam Shroud on simulated DIR-Lab X-rays (`DIR EXPERIMENTS/shroud_sim/`)

Simulated parallel-beam DRRs of T00/T50 (90 views, state switches every 6 views, 1 % noise), shroud = head-foot gradient summed across detector (Zijp et al. 2004).
- **Breathing timing:** recovered label-free for 97 % of views in all 10 cases (static-edge removal + PCA).
- **Breath depth (mm) vs landmark truth (lowest 20 % landmarks):** v1 edge max: corr 0.24, |err| 7.5; v3 PCA + shift: 0.19, 6.0; v4 top-edge of main peak: **0.51, 4.6 (no bias)**; v5 planning-DRR region match: −0.07, 7.5; v6 column-wise edge from planning-CT lung base: **0.69**, 4.9 (C01–C03, C07 within 1 mm; C09, C10 off by 12–16 mm).
- **Does depth predict the stretch each DIR case needs (k_lsq, TCIA3)?** Real diaphragm travel from landmarks: corr 0.87, leave-one-out k error 0.20 (vs 0.31 guessing). Simulated shroud: corr 0.34–0.63, k error 0.33–0.34 (no better than guessing).
- Conclusion: depth is the right signal if measured well; the simple simulation is not good enough. SPARE projections are not on this machine (only GT volumes in `Data/MonteCarloDatasets`). With only a 3D planning CT in the clinic, shroud work is stopped.

# 2026-09-29 16:00–16:30 — Breath depth is the size error, and one CT cannot show it

- **Network motion size / true size at landmarks** (`viz_network_fields.py`): C01 (3.89 mm true): TCIA3 1.45, A′ ep6 1.10. C08 (14.99 mm): TCIA3 0.42, A′ 0.53 → regression to the mean (every model predicts ~5–8 mm).
- **Scale oracle** (`scale_oracle_test.py`, TCIA3 ep100): plain 4.25 · one shared k (1.19) 4.27 · per-case size match 3.70 · best per-case k (lsq) **3.58**. k_lsq: C01 0.62 … C07 1.54, C08 2.16; mean direction cosine 0.73–0.93. C08 plain 9.15 → 5.39 at k = 2.16. → ~0.67 mm of the 4.25 is breath depth; the rest (~3.6) is pattern.
- **Can one CT predict breath depth?** (`amp_from_ct_check.py`, 82 scans / 20 patients, leave-one-patient-out): guess-the-mean R² −0.06; lung volume −0.10; SI height −0.07; AP −0.11; LR −0.16; SI/AP −0.13; L/R −0.09; base fraction −0.09; dome −0.10; all eight −0.62 to −0.97. Ceiling (each patient's own mean) R² 0.48. → No: about half of depth is a stable patient trait, half varies between sessions; no single-CT feature recovers it.
- TCIA phase extremes: the largest-motion pair is 01↔06 in only 59 % of scans (01↔07 in 29 %); DIR-Lab T00/T50 are true extremes → slight extra pull toward under-predicting deep breaths.

**Overall position (2026-10-01).** Best one-CT result: TCIA3.5 ep92 + mirror 4.00 mm (E_all + mirror 4.02). Loss, size, view, variety, optimiser, labels and ensembles all land in 4.0–4.6. Remaining gap to 3.8 is mostly C08/C07 (the other 9 cases average ~3.45). Active: push-label runs (TCIA-lite on GPU 1; TCIA3.5-push to start on GPU 0).

# 2026-09-29 20:19 — Case 8 alongside the cohort score (A′ vs A′-deep128)

Both runs still training. DIR TRE300 is the mean of the 10 case means. Case 8 is tracked as its own average over the epochs that have been scored.

Same epochs (5, 10, 13), the only ones both runs have:

| Epoch | A′ TRE300 | A′ case 8 | deep128 TRE300 | deep128 case 8 |
|---|---:|---:|---:|---:|
| 5 | 4.80 | 10.38 | 4.57 | 8.84 |
| 10 | 4.37 | 8.70 | 4.35 | 8.17 |
| 13 | 4.71 | 9.61 | 4.72 | 8.88 |
| **Mean** | **4.63** | **9.56** | **4.55** | **8.63** |

Deep128 is lower on case 8 at all three epochs (mean gap 0.94 mm). The 10-case mean is only 0.08 mm lower, under the 0.1 mm bar. Epoch 10 is the 8.2 mm point; the other two deep128 epochs are 8.8 mm, so 8.63 is the case-8 average so far, not 8.2.

A′ over all 55 scored epochs: TRE300 4.65, case 8 9.12 (range 8.00–11.14). From epoch 17 on, case 8 averages 8.86 and mostly sits in 8.5–9.4. Best A′ cohort is still epoch 6 at 4.33 mm (case 8 9.39). Best deep128 so far is epoch 10 at 4.35 mm (case 8 8.17). TCIA3 is 4.25 mm, case 8 9.15.

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
