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

## Synthesiser: single planning CT → motion (2 mm iso scorer, TRE₃₀₀, T00→T50)

*Separate scorer from the arm table above (`scripts/rescore_oracle_iso2mm_v3.py`, 2 mm iso grid, `eval_dir_tcia3_iso2mm_v2` geometry). Input = DIR-Lab CT_06 (T50) only. Updated 2026-10-03.*

| Model | TRE₃₀₀ | + mirror | Lung NCC | Lung folding (det J ≤ 0) | Note |
|---|---:|---:|---:|---:|---|
| Identity (no motion) | 8.46 | — | 0.845 | — | |
| TCIA-lite full160-aug (old loss), ep 36–40 | 4.470 | 4.371 (ep 28) | 0.915 (ep 40) | 0 % | baseline for hybrid |
| **TCIA-lite hybrid seed 1**, ep 36–40 | **4.174** | 4.133 (ep 40) | 0.924 (ep 40) | 0.0001 % | passed rule |
| **TCIA-lite hybrid seed 2**, ep 36–40 | **4.163** | 4.145 (ep 40) | **0.925** (ep 40) | **0 %** | passed rule (9/10 cases lower at ep 40) |
| TCIA3.5 (crops, old loss), ep 92 | 4.18 | 4.00 | — | — | |
| TCIA3.5-hybrid, ep 46 (**interim**, run to ep 100) | 4.129 | 4.011 | — | — | vs TCIA3.5 ep 46 4.353 |
| Ensemble s1 + s2 + TCIA3.5-hyb ep 46 (**exploratory**) | 3.916 | **3.909** | — | — | members picked after seeing results |

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

Sources: `Grid160/TCIA3/DecoderCRB/plots/qc_dir_oracle/rescore_oracle_iso2mm_v3_20261003_055111.json`, `mirror_tta_iso2mm_20261002_215330.json`; `Grid160/TCIA_lite/analysis_2026-10-03/{free_checks,ncc_dirlab,jacobian_dirlab}.json`.

## Sources

- A1: `arms/A1_oracle_dirlab/results/cohort_tre_v2_exact_inverse.json` (old: `cohort_tre_r3_final.json`)
- A2: `arms/A2_generic_spare/results/cohort_tre_v2_exact_inverse.json` (old: `cohort_tre_r3_final.json`)
- Log: `logs/eval_a1_a2_tre_v2.log`
- A0: identity from the same landmark files (`dirlab_tre.py --check` matches DIR-Lab's published 300-pt identity)
