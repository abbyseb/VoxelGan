# DIR EXPERIMENTS — Results (TRE)

**KPI:** mean TRE (mm), DIR-Lab **75-point** landmarks (and 300-point), T00→T50. Lower is better.

*Snapshot: 2026-10-02. Re-scored with an **exact inverse** for T00→T50 (`scripts/eval_a1_tre_v2.py`, `scripts/eval_a1_a2_tre_v2.py`).
The previous T00→T50 numbers used a first-order inverse (lm00 − disp(lm00) on a T50-grid field) and were too high, mostly on large-motion cases.
Previous version kept as `Results_pre_2026-10-02.md`.*

## Cohort summary (mean over 10 cases)

| Arm | What | TRE₇₅ | TRE₃₀₀ | Note |
|-----|------|------:|-------:|------|
| **A0** Identity | No motion | **8.69** | **8.46** | |
| **A1** Elastix | Oracle registration on the patient's own 4D-CT | **1.55** | **1.49** | was 2.08 / 1.96 (approx. inverse) |
| **A1** VoxelMap | Patient oracle (real 4D → DRR → NoFiLM) | **1.95** | **1.90** | was 2.32 / 2.23; older 3.16 superseded |
| **A2** VoxelMap | SPARE population prior, zero-shot | **7.30** | **7.12** | **provisional** (see below); was 8.97 in the 14 Sep snapshot |
| **A3** VoxelMap | Synthesiser 4D-CT → DRR → VoxelMap | — | — | **to be redone** with the corrected input and the final synthesiser |

T50→T00 (exact in both versions): A1 Elastix 1.64 / 1.57 · A1 VoxelMap 2.02 / 1.97 · A2 VoxelMap 7.49 / 7.30.

## Per-case, T00→T50 exact (TRE₇₅ / TRE₃₀₀, mm)

| Case | A0 id | A1 Elastix | A1 VM | A2 VM |
|-----:|------:|-----------:|------:|------:|
| 1 | 3.91 / 3.89 | 1.14 / 1.06 | 1.31 / 1.25 | 3.85 / 3.78 |
| 2 | 4.65 / 4.34 | 0.92 / 0.95 | 1.37 / 1.34 | 4.64 / 4.36 |
| 3 | 7.25 / 6.94 | 1.22 / 1.23 | 1.39 / 1.35 | 6.79 / 6.62 |
| 4 | 9.69 / 9.83 | 1.49 / 1.44 | 1.53 / 1.48 | 7.90 / 8.14 |
| 5 | 7.41 / 7.48 | 1.83 / 1.65 | 2.04 / 1.91 | 6.16 / 6.08 |
| 6 | 11.77 / 10.89 | 1.57 / 1.61 | 2.66 / 2.82 | 9.27 / 8.53 |
| 7 | 10.71 / 11.03 | 1.89 / 1.93 | 2.32 / 2.53 | 8.67 / 8.88 |
| 8 | 16.00 / 14.99 | 1.98 / 1.67 | 3.23 / 2.81 | 13.78 / 13.00 |
| 9 | 7.16 / 7.92 | 1.82 / 1.69 | 1.88 / 1.74 | 6.63 / 7.24 |
| 10 | 8.33 / 7.30 | 1.63 / 1.66 | 1.76 / 1.76 | 5.33 / 4.63 |
| **mean** | **8.69 / 8.46** | **1.55 / 1.49** | **1.95 / 1.90** | **7.30 / 7.12** |

## Caveats

- **A2 is provisional.** Its checkpoint link points into `arms/Incorrect DRR/…`, the weights file is missing, and no other A2 weights exist. It was probably trained on DRRs later found to be wrong. Saved fields were re-scored; the model cannot be re-run. Retrain before any claim that depends on A3 vs A2 being close.
- **Do not mix scorers.** Arm numbers here are on the native DIR-Lab grid. Synthesiser (mid-CT → motion) numbers come from `eval_dir_tcia3_iso2mm_v2.py` on a 2 mm grid and are TRE300 (e.g. hybrid 4.17, TCIA3.5 + mirror 4.00); not directly comparable with this table.
- Old A3 rows (6.56 partial, ablations) used the squeezed / wrongly normalised input and are withdrawn.

## Synthesiser headline (corrected 8 Oct 2026, two seeds)

*Single planning CT only. 2 mm iso scorer, TRE₃₀₀; POPI via `eval_popi_tcia3_lps.py`. Main model epochs 96–100, small model 36–40. `Grid160/TCIA_lite/analysis_2026-10-03/{seeds_final,stabilise_free,combo_small_big,all82_clean_old}.json`.*

| Model | DIR-Lab (10) | POPI (6) | All 16 | Note |
|---|---:|---:|---:|---|
| **Best single model: TCIA3.5 old loss + mirror, seeds 1+2 mean** | **4.07** | **4.57** | **4.26** | seed 1 4.02 / 4.50, seed 2 4.12 / 4.64 |
| TCIA3.5 hybrid loss + mirror, seeds 1+2 mean | 4.12 | 4.54 | 4.28 | seed 1 3.87 / 4.46, seed 2 4.37 / 4.63 (unstable) |
| TCIA3.5 old loss, plain, seeds mean | 4.21 | 4.82 | 4.44 | |
| TCIA3.5 hybrid, plain, seeds mean | 4.29 | 4.87 | 4.51 | |
| **Pre-specified combination:** small hybrid s1+s2 + TCIA3.5 old s1+s2, mirror | **3.95** | **4.53** | **4.17** | fixed before scoring |
| Small hybrid, 78 scans (82 minus 4 flagged labels), mirror | 4.24 | 4.73 | 4.42 | best small-model POPI |

Hybrid vs old on the main model (16 patients, seed means): mirror +0.02 mm, p = 0.46 → **no improvement; the earlier 3.87 mm was one seed.** Local NCC top-up: −0.02 DIR-Lab, 0.00 POPI (not kept). The "Synthesiser" section below keeps the earlier single-seed numbers for the record.

## Synthesiser: single planning CT → motion (2 mm iso scorer, TRE₃₀₀, T00→T50)

*Earlier single-seed snapshot (superseded by the headline above).*

*Separate scorer from the arm table above (`scripts/rescore_oracle_iso2mm_v3.py`, 2 mm iso grid, `eval_dir_tcia3_iso2mm_v2` geometry). Input = DIR-Lab CT_06 (T50) only. Updated 2026-10-05.*

| Model | TRE₃₀₀ | + mirror | Lung NCC | Lung folding (det J ≤ 0) | Note |
|---|---:|---:|---:|---:|---|
| Identity (no motion) | 8.46 | — | 0.845 | — | |
| TCIA-lite full160-aug (old loss), ep 36–40 | 4.470 | 4.371 (ep 28) | 0.915 (ep 40) | 0 % | baseline for hybrid |
| **TCIA-lite hybrid seed 1**, ep 36–40 | **4.174** | 4.133 (ep 40) | 0.924 (ep 40) | 0.0001 % | passed rule |
| **TCIA-lite hybrid seed 2**, ep 36–40 | **4.163** | 4.145 (ep 40) | **0.925** (ep 40) | **0 %** | passed rule (9/10 cases lower at ep 40) |
| TCIA3.5 (crops, old loss), ep 96–100 | 4.20 | 4.02 | 0.930 (mirror) | — | |
| **TCIA3.5-hybrid, ep 96–100 (final)** | **4.11** | **3.87** | **0.930** (mirror) | 0 % (mirror) | **best model**; plain −0.09 (fails bar), mirror −0.15 (passes) |
| MagFT (TCIA3 fine-tune), ep 26–30 | 4.15 | 4.15 | — | — | re-scored 5 Oct under the rule |
| Ensemble s1 + s2 + TCIA3.5-hyb ep 100 (**exploratory**) | 3.898 | 3.854 | — | — | members picked after seeing results |
| Lite hybrid, image weight 30, ep 36–40 | 4.339 | — | — | — | worse than weight 10 |
| Lite hybrid, deepest-breath labels, ep 36–40 | 4.245 | — | — | — | not kept |
| Lite hybrid s2, all 82 scans, ep 36–40 | 4.495 | — | — | — | worse (C06 5.48) |

Hybrid = L1 DVF + 10 · image match |tgt − warp(ref, pred)| (lung) + 0.1 · smoothness. Rule: last-5 mean > 0.1 mm better than baseline and ≥ 7/10 cases lower.

**Per case, TRE₃₀₀ mean over ep 36–40 (mm):**

| Case | full160-aug | hybrid s1 | hybrid s2 |
|-----:|------:|------:|------:|
| 1 | 2.37 | 2.10 | 2.02 |
| 2 | 3.03 | 2.42 | 2.39 |
| 3 | 3.14 | 2.85 | 3.06 |
| 4 | 4.26 | 3.74 | 3.91 |
| 5 | 4.48 | 3.90 | 4.48 |
| 6 | 4.06 | 4.17 | 4.19 |
| 7 | 6.23 | 6.66 | 6.17 |
| 8 | 8.83 | 7.91 | 7.61 |
| 9 | 4.38 | 4.00 | 3.91 |
| 10 | 3.93 | 3.99 | 3.90 |
| **mean** | **4.47** | **4.17** | **4.16** |

**Image match and plausibility, hybrid seed 2 ep 40 (`Grid160/TCIA_lite/analysis_2026-10-03/`):**
- NCC, real CT_01 vs CT_06 warped by the network, lung mask dilated 10 mm: identity 0.845 → s2 0.925 (old loss 0.915). Whole 160³ box: identity 0.971, **old loss 0.957 (worse than no motion)**, s2 0.979. Per case (lung): C01 0.963, C02 0.931, C03 0.956, C04 0.925, C05 0.935, C06 0.885, C07 0.920, C08 0.873, C09 0.943, C10 0.917.
- Jacobian det of x + u(x): lung folding 0 % in all 10 cases, min +0.20 (per-case min +0.20 to +0.50), mean 0.90 (≈ 10 % lung volume change), 1st–99th pct 0.59–1.16. Whole box 0.06 % folded (outside lung).
- No Elastix upper reference on this grid yet.

**Jacobian, main model (det of x + u(x), DIR-Lab, lung mask +10 mm; `analysis_2026-10-03/jacobian_dirlab_main.json`, 5 Oct):**

| Model (ep 100 unless noted) | Lung folded | Lung min | Lung mean | Box folded |
|---|---:|---:|---:|---:|
| **TCIA3.5-hybrid + mirror, ep 96–100** | **0 % every epoch** | +0.09 to +0.17 | 0.89–0.90 | 0.006–0.017 % |
| TCIA3.5-hybrid plain | 0.0008 % | −0.20 | 0.88 | 0.056 % |
| TCIA3.5 old + mirror | 0 % | +0.15 | 0.90 | 0 % |
| TCIA3.5 old plain | 0.0017 % | −0.20 | 0.90 | 0.0004 % |

Hybrid + mirror ep 100 per case: no lung folding in C01–C10 (min +0.14 to +0.40). Mirror removes the few folded lung voxels; hybrid adds slight folding outside the lung vs old loss.

**POPI and the 16-patient test (main model, ep 96–100, 5 Oct; `analysis_2026-10-03/{popi_scores,spread}.json`):**

| Model | DIR-Lab (10) | POPI (6) | All 16 | 
|---|---:|---:|---:|
| TCIA3.5 old loss | 4.20 ± 2.26 | 4.56 ± 1.78 | 4.34 ± 2.04 |
| TCIA3.5 old loss + mirror | 4.02 ± 2.19 | 4.50 ± 1.89 | 4.20 ± 2.03 |
| TCIA3.5-hybrid | 4.11 ± 2.05 | 4.84 ± 1.64 | 4.38 ± 1.88 |
| **TCIA3.5-hybrid + mirror** | **3.87 ± 1.96** | **4.46 ± 1.76** | **4.09 ± 1.85** |

(mean ± SD over patients; landmark-pooled SD for hybrid + mirror: DIR 3.33, POPI 2.85.) Paired hybrid vs old over 16 patients: plain +0.05 (p 0.90, 9/16), mirror −0.11 (95% CI −0.23 to +0.01, 11/16, Wilcoxon p 0.14) → **not significant**; seed 2 of both losses running (6 Oct).

Sources: `Grid160/TCIA3/DecoderCRB/plots/qc_dir_oracle/rescore_oracle_iso2mm_v3_20261003_055111.json`, `mirror_tta_iso2mm_20261002_215330.json`; `Grid160/TCIA_lite/analysis_2026-10-03/{free_checks,ncc_dirlab,jacobian_dirlab}.json`.

## Sources

- A1: `arms/A1_oracle_dirlab/results/cohort_tre_v2_exact_inverse.json` (old: `cohort_tre_r3_final.json`)
- A2: `arms/A2_generic_spare/results/cohort_tre_v2_exact_inverse.json` (old: `cohort_tre_r3_final.json`)
- Log: `logs/eval_a1_a2_tre_v2.log`
- A0: identity from the same landmark files (`dirlab_tre.py --check` matches DIR-Lab's published 300-pt identity)
