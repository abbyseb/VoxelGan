# ClinicalExperiments

Zero-shot transfer checks: **SPARE-trained CRB → Varian / Elekta clinical 4DCT** (DVF vs Elastix).

Organised by **inference grid**:

| Folder | Grid | Status |
|--------|------|--------|
| **[Grid128/](Grid128/)** | Lung-bbox → **128³** (aniso mm/voxel) | **Archive** — E1–E6 complete |
| **[Grid160/](Grid160/)** | **2 mm, 160³** iso (320 mm FOV) | **Active** — 9-model matrix (6 done + 3 FOV trains) |

**End-to-end product metric:** [`../../VoxelMap_Experiments/`](../../VoxelMap_Experiments/) (single CT → synth 4D → projections).

**Motion-model training (iso):** [`../IsoExperiments/`](../IsoExperiments/) + [`../CRBExperiments/`](../CRBExperiments/).

## Layout

```
ClinicalExperiments/
  Grid128/          # all completed 128³ experiments (E1–E6)
  Grid160/          # next: clinical QC @ 2 mm 160³
  Experiment1…6 →   # symlinks → Grid128/ (backward compat)
  scripts/ plots/ logs/ → Grid128/
```

## Grid128 summary (done — do not retrain)

| Exp | Phase | Notes |
|-----|-------|-------|
| E1 | linear | Baseline full SPARE |
| E2 | cyclic (intended) | Broken sys.path — skip |
| E3 | cyclic + **FOV/CBCT aug** | Best 128³ train recipe; used by `VoxelMap_Experiments` synth arm |
| E4 | — | Inference-only mm rescoring (marginal) |
| E5 | — | Inference-only **µ norm** (large Elekta gain) |
| E6 Normal | linear, full 128³ | vs E1: mixed; Elekta minmax ↑ |
| E6 Cyclic | cyclic, full 128³ | **≈ same as E6 Normal** (see Grid128/README) |

**Lessons to port to Grid160 / iso training:** E3 FOV aug, E5 µ norm, amplitude conditioning (`CRBExperiments/`).

## Grid160 (next)

Clinical zero-shot QC with models trained on `data_iso` — see [Grid160/README.md](Grid160/README.md).

Legacy paths `Experiment1` … `scripts/` still resolve via symlinks.
