# Grid128 — archived clinical transfer (128³)

All experiments train on **full SPARE P1–P9** at **128³ lung-bbox** resample (per-patient aniso mm/voxel). Test: zero-shot Varian `CV_*` + Elekta `CE_*` vs Elastix.

**Status: complete — archive only. Do not retrain.** Use [Grid160/](../Grid160/) for iso clinical QC.

## Experiments

| Folder | Encoding | Train | Key result |
|--------|----------|-------|------------|
| [Experiment1](Experiment1/) | linear | 64³ patches | Baseline |
| [Experiment2](Experiment2/) | cyclic (broken) | — | Skip |
| [Experiment3](Experiment3/) | cyclic + **FOV aug** | 64³ patches | **Synth ckpt** for VoxelMap_Experiments |
| [Experiment4](Experiment4/) | — | inference only | mm rescoring: marginal |
| [Experiment5](Experiment5/) | — | inference only | **µ norm**: Elekta cos 0.20→0.51 |
| [Experiment6/Normal](Experiment6/Normal/) | linear | **full 128³** | vs E1: Elekta minmax ↑ |
| [Experiment6/Cyclic](Experiment6/Cyclic/) | cyclic | **full 128³** | ≈ tie with E6 Normal |

## Did cyclic encoding help?

**Fair comparison (same setup): E6 Normal (linear) vs E6 Cyclic — BothCRB, full 128³, µ norm**

| Cohort | E6 Normal cos | E6 Cyclic cos | Δ |
|--------|---------------|---------------|---|
| Varian (mean P1–P5) | **0.584** | 0.571 | −0.013 |
| Elekta (mean P1–P5) | **0.478** | 0.454 | −0.024 |

Per-scan (Both µ): mean Δ cos **−0.019** (range −0.066 … +0.016). **No meaningful gain.**

**Confounded comparison:** E3 cyclic + FOV aug beat E1 linear on Elekta minmax — but FOV augmentation changed too, so you cannot attribute that to cyclic alone.

**vs iso baseline:** Iso-E2 full clinical QC (Both µ): Varian **0.592**, Elekta **0.484** — similar to E6 Normal without retraining on 128³.

### Takeaway

- Cyclic `(cos, sin)` phase encoding: **optional, not a priority** on this data at 128³.
- Bigger wins on Elekta: **µ intensity norm (E5)** >> cyclic encoding.
- For controlled ventilator / periodic breathing, cyclic may still be worth testing **with amplitude conditioning on iso grid** (`CRBExperiments/E3`).

## Shared assets

- [scripts/](scripts/) — `prepare_clinical_dvf_library.py`, surveys, E4/E5 QC helpers
- [plots/](plots/) — Varian/Elekta CT surveys
- [logs/](logs/) — queue / prep logs

## References

- E6 comparison table: [Experiment6/Normal/plots/qc_summary/compare_e1_e6.tsv](Experiment6/Normal/plots/qc_summary/compare_e1_e6.tsv)
- Parent overview: [../README.md](../README.md)
