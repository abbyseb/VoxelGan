# DIR EXPERIMENTS — Results (TRE)

**KPI:** mean TRE (mm), DIR-Lab **75-point** landmarks, T00→T50.

Lower is better. Identity = no motion (A0).

*Snapshot: 2026-09-14. A3 g160ft Case 3 TRE still pending (train in progress).*

## Cohort summary (mean over cases)

| Arm | What | Mean TRE₇₅ (mm) | Cases |
|-----|------|----------------:|------:|
| **A0** Identity | No motion | **8.69** | 10 |
| **A1** Elastix | Oracle registration on real 4D-CT | **2.08** | 10 |
| **A1** VoxelMap | Patient oracle (real 4D → DRR → NoFiLM) | **3.16** | 10 |
| **A2** VoxelMap | SPARE MC Prior zero-shot on DIR | **8.97** | 10 |
| **A3** VoxelMap (default G160) | Synth DVFs → NoFiLM; SPARE G160 zero-shot | **6.56** | 4 (partial) |

## Per-case TRE₇₅ (mm)

| Case | A0 id | A1 Elastix | A1 VM | A2 VM | A3 default |
|-----:|------:|-----------:|------:|------:|-----------:|
| 1 | 3.91 | 1.24 | 1.33 | 4.07 | 4.18 |
| 2 | 4.65 | 1.09 | 1.38 | 4.94 | 4.81 |
| 3 | 7.25 | 1.30 | 1.86 | 7.36 | 7.55 |
| 4 | 9.69 | 1.89 | 3.10 | 9.86 | 9.70 |
| 5 | 7.41 | 2.08 | 2.32 | 7.67 | — |
| 6 | 11.77 | 2.93 | 3.83 | 12.06 | — |
| 7 | 10.71 | 2.25 | 5.03 | 10.58 | — |
| 8 | 16.00 | 3.76 | 4.91 | 16.13 | — |
| 9 | 7.16 | 2.07 | 4.72 | 8.12 | — |
| 10 | 8.33 | 2.19 | 3.14 | 8.92 | — |
| **mean** | **8.69** | **2.08** | **3.16** | **8.97** | **6.56** (4 cases) |

## A3 diagnostics / ablations (Case 1 unless noted)

| Variant | TRE₇₅ | vs identity | Note |
|---------|------:|------------:|------|
| A3 default C1 | 4.18 | +0.27 | SPARE G160 synth labels |
| A3 hist-match µ C1 | 4.43 | +0.52 | µ domain only; no TRE gain |
| A3 P3 semi-oracle C1 | 2.54 | -1.38 | Synth DRRs + **A1 Elastix labels** (diagnostic) |
| A1 VoxelMap C1 | 1.33 | -2.59 | Oracle upper bound |
| A1 Elastix C1 | 1.24 | -2.67 | |

### G160 DIR fine-tune (C1/C5/C8) → A3 holdout

Synthesizer fine-tuned on DIR Elastix (C1,C5,C8); inference = CT only.

| Check | Result |
|-------|--------|
| cos(synth, Elastix) @160 phase01, holdout C2/C3 | ~0.72 / 0.78 (was ~0.14 / 0.22) |
| A3 + g160ft VoxelMap TRE C3 | **pending** (train ~ep 35/50) |

## Takeaways so far

1. **A1** works (oracle 4D): VM mean ~3.2 mm vs id ~8.7.
2. **A2** SPARE prior ≈ identity (~9.0) — generic prior does not transfer.
3. **A3** SPARE G160 zero-shot ≈ identity / slightly worse (partial C1–C4).
4. µ hist-match does not fix TRE; **direction** of synth motion was wrong.
5. P3 (real labels + synth DRRs) improves C1 (2.54 vs id 3.91) — appearance partly OK; motion teacher was the bottleneck.
6. DIR fine-tune of G160 improves direction on holdouts; A3 TRE with FT ckpt TBD.

## Sources

- A0: `arms/A0_identity/results/summary.json`
- A1: `arms/A1_oracle_dirlab/results/cohort_tre_frozen.json`
- A2: `arms/A2_generic_spare/results/cohort_tre_frozen.json`
- A3: `arms/A3_synth_conditioned/runs/DIR_C0N/tre/tre_summary.json`
- G160 FT: `experiments/G160_DIR_FT_C1C5C8/`

