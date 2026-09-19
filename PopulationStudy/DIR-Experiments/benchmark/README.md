# DIR-Lab benchmark (Phase 0 + 1)

Paper-style **TRE (mm)** on 300 landmarks, **T00→T50**, 10 cases.

## What you need to provide

| Item | Status | Notes |
|------|--------|-------|
| DIR data | **Done** | `DIR-Experiments/data/` (7.6 GB) |
| Packed volumes | **Done** | `Experiment1/packed/` (spare_axes) |
| E6 / E3 weights | **Done** | ClinicalExperiments checkpoints |
| LEARN-GUI venv | **Needed to run** | `.../LEARN-GUI-Python/.venv/bin/python` (itk + lungmask) |
| **Free GPU** | **Your call** | ~1 GPU for Elastix + CRB inference (~1–2 h full sweep) |
| Disk | ~2 GB extra | Elastix cache per case under `packed/P*/benchmark/` |

**Nothing to download or label.** Optional: tell me which GPU to use (`--gpu 0` vs `1`) and if long jobs should avoid GPU 0 (VoxelMap eval).

## Phase 0 — run benchmark

```bash
cd PopulationStudy/DIR-Experiments

# Tier-0 sanity (one patient): landmark round-trip + Elastix vs identity
python benchmark/verify_landmark_chain.py --patient P1_DIR

VENV=/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python

# Quick: identity + Elastix + E6 Both µ
$VENV benchmark/eval_dirlab_standard.py \
  --methods identity,elastix,e6_both_mu \
  --gpu 1
```

**Harness check:** cohort table must show **Elastix mean TRE < identity** on most patients.
Landmark displacement uses `pred = ref − u` (ITK warp / grid_sample convention).

**Output:** `benchmark/results/tre_benchmark_<pack>.tsv`  
Columns: `native_mm` (paper-style) and `packed_vox` (internal).

## Phase 1 — axial pack + re-benchmark

```bash
cd PopulationStudy/DIR-Experiments/Experiment1
$VENV scripts/pack_dirlab.py --layout axial --redo   # → packed_axial/

cd ..
$VENV benchmark/eval_dirlab_standard.py \
  --packed-root Experiment1/packed_axial \
  --methods identity,elastix,e6_both_mu \
  --gpu 1
```

Compare `packed` vs `packed_axial` cohort means in the two TSV files.

## Methods registry

| Key | Model |
|-----|--------|
| `identity` | No registration |
| `elastix` | Masked B-spline (same param as clinical) |
| `e6_both_mu` | E6 Normal Both + µ norm |
| `e6_both_minmax` | E6 Both + minmax |
| `e3_both_mu` | E3 cyclic FOV Both + µ |
| `e5_both_mu` | E3 Both + E5 µ (same weights as e3) |

## Success criteria (Phase 0)

- Elastix **beats identity** on most cases (~1–3 mm mean target)
- At least one CRB method **beats identity** on cohort mean TRE
