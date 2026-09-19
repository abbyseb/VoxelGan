# ClinicalExperiments — Experiment 6

**Full SPARE train at 128³** (vs E1–E5 train crops at 64³), zero-shot clinical test later.

Two sections:

| Folder | Phase | Train size | Status |
|--------|-------|------------|--------|
| [Normal/](Normal/) | **linear** `cond_dim=2` | **128³** full volume | training Enc/Dec/Both |
| [Cyclic/](Cyclic/) | cyclic `cond_dim=4` | **128³** | scaffold only (start later) |

Same data split / seed as Experiment1. No FOV aug. CT norm in dataset = min–max (same as E1 train).

## Why

Ablate whether full-volume 128³ training (matched to QC) beats 64³ patch training.

## Commands (Normal)

```bash
cd PopulationStudy/ClinicalExperiments/Experiment6/Normal
PYTHONPATH=. PYTHONUNBUFFERED=1 python scripts/train_mse.py --arch encoder --gpu 1 --epochs 100
PYTHONPATH=. PYTHONUNBUFFERED=1 python scripts/train_mse.py --arch decoder --gpu 1 --epochs 100
PYTHONPATH=. PYTHONUNBUFFERED=1 python scripts/train_mse.py --arch both --gpu 1 --epochs 100
```
