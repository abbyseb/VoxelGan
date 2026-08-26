# ClinicalExperiments — Experiment 4

**Inference-only grid-scale / millimetre rescoring** of the Experiment 3 BothCRB
checkpoint (cyclic + FOV aug). **No retrain.**

## What changed vs Experiment 3

| | Experiment 3 | Experiment 4 |
|--|--------------|---------------|
| Weights | Trained BothCRB | **Same** (symlink) |
| Input CT norm | min–max | **Same** |
| DVF units at score | packed **voxels** | also **mm** and SPARE-scale **corrected** voxels |
| Training | FOV/CBCT aug | none |

Packing is `lung-bbox (1 mm native) → 128³`. Elastix/network DVFs live in packed-voxel
units. Per-axis spacing is `bbox_size_zyx / 128`. E4 tests whether converting to mm
(or rescaling pred by `sp_SPARE / sp_patient`) closes the Elekta gap.

### Modes

1. **voxel** — same metric space as E3 QC (baseline)  
2. **mm** — pred & GT × per-axis pack spacing  
3. **corrected** — `pred *= (sp_SPARE / sp_patient)` per axis, score vs GT in patient voxels  

## Verdict

**Scale-only does not fix Elekta.** mm scoring: Elekta cos **0.197 → 0.215**.  
Varian almost unchanged. Prefer E5 (intensity norm) for the real lift.

## Status

- [x] mm / corrected rescoring on CV_P1–P5 + CE_P1–P5  
- [x] Metrics JSON/TSV  
- [x] Visual QC panels = E3 Both voxel predictions (same ckpt; E4 does not change images)

## Layout

```
Experiment4/
  data -> ../Experiment1/data
  BothCRB/weights/  -> E3 fov-aug generator (symlink)
  BothCRB/plots/
    e4_summary.json
    e4_metrics.tsv
    qc_varian/   (symlink → qc_varian_voxel_baseline)
    qc_elekta/   (symlink → qc_elekta_voxel_baseline)
  scripts/e4_mm_rescoring.py
```

## Results (cohort means)

| Vendor | Mode | L1/zero | cos | beat |
|--------|------|---------|-----|------|
| Varian | voxel | 0.890 | 0.563 | 84% |
| Varian | mm | 0.898 | 0.543 | 84% |
| Varian | corrected | 0.886 | 0.561 | 85% |
| Elekta | voxel | 0.990 | 0.197 | 59% |
| Elekta | mm | 0.990 | **0.215** | 60% |
| Elekta | corrected | 0.996 | 0.187 | 57% |

## Commands

```bash
cd PopulationStudy/ClinicalExperiments
python Experiment4/scripts/e4_mm_rescoring.py --gpu 1
# writes BothCRB/plots/e4_summary.json (also under ../plots/e4_mm_rescoring/)
```

## QC browse

- Metrics: [`BothCRB/plots/e4_summary.json`](BothCRB/plots/e4_summary.json), [`e4_metrics.tsv`](BothCRB/plots/e4_metrics.tsv)  
- Panels (voxel / E3 preds): `BothCRB/plots/qc_varian/P*/CV_*`, `BothCRB/plots/qc_elekta/P*/CE_*`
