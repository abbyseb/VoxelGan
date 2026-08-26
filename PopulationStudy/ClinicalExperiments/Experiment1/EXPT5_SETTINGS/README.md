# Experiment 1 · EXPT5_SETTINGS

**Same Experiment 1 weights** (linear phase, full SPARE, no FOV aug), but QC uses
**Experiment 5 CT intensity settings**: µ air/water anchors instead of min–max.

This is **inference-only** — not a retrain, not clinical fine-tuning.

## What changed vs default E1 QC

| | Default E1 QC | EXPT5_SETTINGS |
|--|---------------|----------------|
| Weights | Enc / Dec / Both | **Same** |
| Phase | linear `cond_dim=2` | **Same** |
| CT norm | min–max | **µ air→0, water→0.02** (E5 best) |

## Layout

```
EXPT5_SETTINGS/
  README.md
  summary_all.json
  metrics.tsv
  EncoderCRB/plots/qc_varian|qc_elekta/P*/C*/
  DecoderCRB/plots/qc_varian|qc_elekta/P*/C*/
  BothCRB/plots/qc_varian|qc_elekta/P*/C*/
  scripts/qc_e5_settings.py
```

## Status

- [ ] QC Enc/Dec/Both × Varian P1–P5 × Elekta P1–P5 (running / pending)

## Commands

```bash
cd PopulationStudy/ClinicalExperiments/Experiment1
PYTHONPATH=. PYTHONUNBUFFERED=1 python EXPT5_SETTINGS/scripts/qc_e5_settings.py --gpu 1
```

Compare to default E1 QC under `../{Encoder,Decoder,Both}CRB/plots/qc_*` and to
`../../Experiment5/` (E3 Both + same µ norm).
