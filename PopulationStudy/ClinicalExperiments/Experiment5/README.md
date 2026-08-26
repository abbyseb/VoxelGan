# ClinicalExperiments — Experiment 5

**Inference-only robust CT intensity normalization** on the Experiment 3 BothCRB
checkpoint. **No retrain. Not clinical fine-tuning** — still zero-shot.

## What changed vs Experiment 3

| | Experiment 3 | Experiment 5 |
|--|--------------|---------------|
| Weights | Trained BothCRB | **Same** (symlink) |
| CT → network | global **min–max** | **percentile clip** or **µ air/water** anchors |
| FOV aug | used in train | unchanged (same weights) |
| Clinical labels | unused | **unused** |

E3 QC did `(ct - min) / (max - min)`. Outliers on Elekta pushed the whole volume into an
OOD brightness range. E5 replaces that scaling at **test time only**.

### Norms swept

| Name | Rule |
|------|------|
| `minmax` | baseline (E3) |
| `p0.5_p99.5` … `p5_p95` | clip to percentiles, then → [0,1] |
| **`mu_air_water`** | air µ→0, water µ→0.02, clip, scale (**best**) |

## Verdict

**Success.** Elekta mean cos **0.197 → 0.505** (`mu_air_water`), beat **59% → 82%**.  
Hard patients (P1–P3) rose the most. Varian held / slightly improved.  
→ Intensity scaling was the main Elekta failure mode (not mm/voxel — see E4).

## Status

- [x] Norm sweep QC metrics (all CV + CE)  
- [x] Full 100-pair visual QC panels for **µ air/water** (Varian + Elekta P1–P5)

## Layout

```
Experiment5/
  data -> ../Experiment1/data
  BothCRB/weights/  -> E3 fov-aug generator (symlink)
  BothCRB/plots/
    e5_summary.json          # full sweep
    e5_metrics.tsv
    qc_varian/P*/CV_*/       # µ air/water panels + summary.json
    qc_elekta/P*/CE_*/       # µ air/water panels + summary.json
  scripts/e5_intensity_norm_qc.py
  scripts/e5_write_qc_panels.py
```

## Results

### Cohort means

| Vendor | Norm | L1/zero | cos | beat |
|--------|------|---------|-----|------|
| Varian | minmax | 0.890 | 0.563 | 84% |
| Varian | µ air/water | 0.866 | 0.577 | 86% |
| Elekta | minmax | 0.990 | 0.197 | 59% |
| Elekta | p0.5–p99.5 | 0.911 | 0.423 | 80% |
| Elekta | **µ air/water** | **0.892** | **0.505** | **82%** |

### Elekta per patient (minmax → µ)

| Scan | cos | beat |
|------|-----|------|
| CE_P1 | 0.136 → **0.589** | 40% → **93%** |
| CE_P2 | 0.130 → **0.477** | 47% → **81%** |
| CE_P3 | 0.056 → **0.452** | 52% → **86%** |
| CE_P4 | 0.346 → **0.455** | 77% → 68% |
| CE_P5 | 0.315 → **0.551** | 80% → 82% |

## Commands

```bash
cd PopulationStudy/ClinicalExperiments
# sweep metrics only
python Experiment5/scripts/e5_intensity_norm_qc.py --gpu 1
# write µ air/water panels
python Experiment5/scripts/e5_write_qc_panels.py --gpu 1
```

## QC browse

- Sweep metrics: [`BothCRB/plots/e5_summary.json`](BothCRB/plots/e5_summary.json)  
- Panels: `BothCRB/plots/qc_elekta/P1/CE_P1_V_01/*.png` (etc.)
