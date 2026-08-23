# PopulationStudy — Experiment 2

Leave-**patient**-out population study with Dan 2.0 CRB generators, **MSE only** (no discriminator, no FiLM).

## Split (fixed, user-specified)

| Role | Patients | Pairs |
|------|----------|-------|
| **Train** | P1, P3, P4, P5 | 400 (360 train + 40 val @ 10%) |
| **Hold-out test** | P7, P9 | 200 |

See **`seed.json`** — seed `20260818` for val pair split and all RNGs.

Experiment 1 used a random 6/9 draw (train P1,P2,P6,P7,P8,P9; test P3,P4,P5). Experiment 2 is a **different** hold-out to probe generalization on P7/P9 while training on P1/P3/P4/P5.

## What is happening

Same pipeline as Experiment 1:
- Pooled patient-prefixed symlinks (`P01_01_to_02_pair.npy` → that patient's CTs/mask only)
- Three architectures: **EncoderCRB**, **DecoderCRB**, **BothCRB**
- Lung-masked MSE vs Elastix, lr `1e-4`, 100 epochs, 64³ patches (16 train / 8 val)
- QC on hold-out with warped CT panel (2×4 layout)

## Build pooled data

```bash
cd PopulationStudy/InitialExperiments/Experiment2
python scripts/build_pooled_dataset.py
```

## Train

```bash
cd PopulationStudy/InitialExperiments/Experiment2
PYTHONPATH=. PYTHONUNBUFFERED=1 python EncoderCRB/train_crb_enc_mse.py --gpu 0
PYTHONPATH=. PYTHONUNBUFFERED=1 python DecoderCRB/train_crb_dec_mse.py --gpu 1
PYTHONPATH=. PYTHONUNBUFFERED=1 python BothCRB/train_crb_both_mse.py --gpu 0
```

Weights: `crb_{enc,dec,both}_mse_pop_e2_generator.pth`

## QC (hold-out P7, P9)

```bash
PYTHONPATH=. python scripts/qc_pairs.py \
  --arch decoder \
  --ckpt DecoderCRB/weights/crb_dec_mse_pop_e2_generator.pth \
  --out_dir DecoderCRB/plots/qc_holdout_final
```

## Layout

Same self-contained tree as Experiment 1 (`networks/`, `losses/`, `utilities/`, `scripts/`, `data/pooled/`).
