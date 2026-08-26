# ClinicalExperiments — Experiment 2

**Full SPARE MC train → zero-shot Clinical Varian test**, same as Experiment 1, but with
**cyclic phase encoding**:

`[cos 2πθ_ref, sin 2πθ_ref, cos 2πθ_tgt, sin 2πθ_tgt]` (`cond_dim=4`)

No amplitude conditioning. Same seed / patient pool / val split as E1.

## vs Experiment 1

| | E1 | E2 |
|--|----|----|
| Phase codes | linear `(θ/9, θ/9)` | cyclic cos/sin |
| `cond_dim` | 2 | 4 |
| Train data | SPARE P1–P9 | **same** (symlinked pooled) |
| Clinical GT | Elastix Varian | **same** (symlinked) |
| Weights | `*_mse_full_spare_*` | `*_mse_cyclic_full_spare_*` |

## Status

- [x] Scaffold + cyclic Encoder/Decoder/Both nets
- [x] Reuse E1 pooled + clinical `data/` via symlink
- [ ] Train Encoder / Decoder *(running on GPU 1)* / Both
- [ ] Zero-shot QC P1–P5

## Commands

```bash
cd PopulationStudy/ClinicalExperiments/Experiment2
python scripts/build_pooled_dataset.py   # refresh symlinks to E1 data

# Train (same recipe as E1)
PYTHONPATH=. PYTHONUNBUFFERED=1 python scripts/train_mse.py --arch encoder --gpu 0 --epochs 100
PYTHONPATH=. PYTHONUNBUFFERED=1 python scripts/train_mse.py --arch decoder --gpu 1 --epochs 100
# PYTHONPATH=. PYTHONUNBUFFERED=1 python scripts/train_mse.py --arch both --gpu 0 --epochs 100

# QC → plots/qc_varian/P{n}/CV_P{n}_V_01/
PYTHONPATH=. python scripts/qc_varian.py --arch decoder --scan CV_P1_V_01 --all --gpu 1
```
