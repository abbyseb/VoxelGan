# G160-E2 — inference norm sweep (no retrain)

Sweep CT normalization at QC time on all available checkpoints
(**A0**, **A0h**, **A1** × Enc/Dec/Both; A0h Both missing):

| Norm | Notes |
|------|-------|
| `minmax` | baseline |
| `p0.5_p99.5`, `p1_p99`, `p2_p98`, `p5_p95` | percentile clip |
| `mu_air_water` | expect best on Elekta (Grid128 E5) |

**Metrics only** — no PNG panels.

## Run

```bash
cd PopulationStudy/ClinicalExperiments/Grid160/Experiment2
bash scripts/run_norm_sweep.sh          # GPU=0 by default
# or:
PYTHONPATH=. python scripts/qc_norm_sweep.py --gpu 0
```

## Outputs

```
plots/qc_norm_sweep/
  summary_*.json     # per-scan metrics for every arm/arch/norm
  metrics_*.tsv      # flat table
  cohort_*.tsv       # vendor means
logs/qc_norm_sweep.log
```

## Checkpoints

| Arm | Source |
|-----|--------|
| A0 / A0h | `../Experiment0/{Arch}CRB/weights/` |
| A1 | `../Experiment1/{Arch}CRB/weights/` |
