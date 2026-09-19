# DIR-Lab Tier-0 benchmark — harness validation

**Status:** PASS (2026-08-31)

## What was fixed

Landmark TRE applied the Elastix DVF with the **wrong sign**. ITK / `grid_sample` warps use
`output(p) = moving(p + u(p))`, so a physical landmark at `p` in the moving (T00) frame maps to
**`p − u(p)`** in the fixed (T50) frame — not `p + u(p)`.

Files changed:
- `benchmark/tre_utils.py` — `eval_tre_from_disp`
- `benchmark/verify_landmark_chain.py` — new sanity script
- `benchmark/eval_dirlab_standard.py` — cohort report + harness PASS/FAIL

## Sanity check (P1_DIR)

```bash
python benchmark/verify_landmark_chain.py --patient P1_DIR
```

| Check | Result |
|-------|--------|
| Landmark round-trip native↔packed | ~0 mm error |
| Identity TRE | 3.89 mm |
| Elastix TRE | 3.13 mm (beats identity) |
| Image warp residual (µ) | Elastix < identity |

## Cohort results (`packed` / spare_axes)

Run: `eval_dirlab_standard.py --methods identity,elastix,e6_both_mu,...`

| Method | Mean TRE (mm) | vs identity | Beat identity |
|--------|---------------|-------------|---------------|
| identity | **8.46** | — | — |
| **elastix** | **3.20** | −5.26 | **10/10** |
| e6_both_mu | 13.23 | +4.76 | 0/10 |
| e6_both_minmax | 12.07 | +3.61 | 0/10 |
| e3_both_mu | 13.12 | +4.66 | 0/10 |

**Harness:** Elastix beats identity on **10/10** patients — benchmark is valid.

E6 zero-shot does **not** beat identity on TRE (expected until domain/amplitude fixes).

## Commands

```bash
cd PopulationStudy/DIR-Experiments

# Per-patient sanity
python benchmark/verify_landmark_chain.py --patient P1_DIR

# Full benchmark (reuse cached Elastix DVFs)
VENV=.../LEARN-GUI-Python/.venv/bin/python
$VENV benchmark/eval_dirlab_standard.py \
  --methods identity,elastix,e6_both_mu,e6_both_minmax,e3_both_mu \
  --gpu 0 --skip-elastix
```

Outputs: `benchmark/results/tre_benchmark_packed.tsv`, `.json`
