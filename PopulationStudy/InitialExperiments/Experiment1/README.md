# PopulationStudy — Experiment 1

Leave-**patient**-out population study with Dan 2.0 CRB generators, **MSE only** (no discriminator, no FiLM).

## What is happening

1. Pool DVFs from **6 train patients** (100 pairs each, **identity included** → **600** pairs).
2. Hold out **3 patients** (P3, P4, P5 → **300** pairs) for final QC only.
3. Train three architectures side-by-side on the same pooled data:
   - **EncoderCRB** — CRB on encoder + bottleneck (`networks/generator_crb.py`)
   - **DecoderCRB** — CRB on decoder (`networks/generator_crb_dec.py`)
   - **BothCRB** — CRB on encoder + bottleneck + decoder (`networks/generator_crb_both.py`)
4. Objective: lung-masked MSE vs Elastix DVF (`losses/losses.py`). Same hyperparameters as Dan 2.0: 100 epochs, lr `1e-4`, 64³ patches (16 train / 8 val), Adam.
5. Validation uses a **10% pair hold-out from train patients only**. P3–P5 never enter train or val.
6. QC on hold-out: original target CT (not warp-as-target) and `|target − warp(ref, pred)| · lung`.

This folder is **self-contained**: network code, loss, warp, dataset, and train/QC scripts live here (copied/adapted from `Dan2.0/`). Unmodified Dan 2.0 train scripts are also under `scripts/dan20_copies/`.

## Seeds (see `seed.json`)

| Item | Value |
|------|-------|
| Patient split seed | **`20260817`** |
| Train patients | P1, P2, P6, P7, P8, P9 |
| Hold-out test | P3, P4, P5 |
| Val pair split seed | **`20260817`** (10% per train patient) |
| torch / numpy / python / CUDA seed | **`20260817`** |

Patient split method: `random.Random(20260817).sample(P1..P9, k=6)`.

## Matching rule (no cross-patient mixing)

Every file is patient-prefixed. A pair only ever loads that patient’s CTs and mask:

```
P01_01_to_02_pair.npy  →  P01_CT_01.npy , P01_CT_02.npy , P01_Mask_Lung.npy
```

The DataLoader shuffles **indices**; `__getitem__` never crosses patients. There is **no patient-ID embedding** — conditioning is phase codes only (`t_ref`, `t_tgt`), same as Dan 2.0.

Pooled trees (symlinks into `../data/P*/all/`):

```
data/pooled/train/   P01_*, P02_*, P06_*, P07_*, P08_*, P09_*
data/pooled/test/    P03_*, P04_*, P05_*
data/pooled/manifest.json
```

## Build pooled data

```bash
cd PopulationStudy/InitialExperiments/Experiment1
python scripts/build_pooled_dataset.py
```

## Train

```bash
cd PopulationStudy/InitialExperiments/Experiment1
# Encoder + Decoder in parallel on two GPUs
PYTHONPATH=. PYTHONUNBUFFERED=1 python EncoderCRB/train_crb_enc_mse.py --gpu 0 \
  2>&1 | tee EncoderCRB/plots/train_crb_enc_mse.log
PYTHONPATH=. PYTHONUNBUFFERED=1 python DecoderCRB/train_crb_dec_mse.py --gpu 1 \
  2>&1 | tee DecoderCRB/plots/train_crb_dec_mse.log
# After one finishes:
PYTHONPATH=. PYTHONUNBUFFERED=1 python BothCRB/train_crb_both_mse.py --gpu 0 \
  2>&1 | tee BothCRB/plots/train_crb_both_mse.log
```

Or: `PYTHONPATH=. python scripts/train_mse.py --arch encoder --gpu 0`

## QC (hold-out patients)

```bash
PYTHONPATH=. python scripts/qc_pairs.py \
  --arch encoder \
  --ckpt EncoderCRB/weights/crb_enc_mse_pop_e1_generator.pth \
  --out_dir EncoderCRB/plots/qc_holdout
```

QC panel (2×4):

| | Col 1 | Col 2 | Col 3 | Col 4 |
|--|-------|-------|-------|-------|
| Row 1 | ref CT | target CT (original) | **warp(ref, pred)** | \|target − warp\| · lung |
| Row 2 | \|Elastix\| | \|pred\| | \|pred − Elastix\| · lung | — |

## Layout

```
InitialExperiments/Experiment1/
  seed.json
  README.md
  networks/          # UNetCRB, UNetCRBDecoder, UNetCRBBoth (copies from Dan2.0)
  losses/            # DVFMSELoss, DVFLoss
  utilities/         # dataset, warp, view_config, seed_utils
  scripts/           # build_pooled_dataset, train_mse, qc_pairs, dan20_copies/
  EncoderCRB/ DecoderCRB/ BothCRB/   # train wrappers, weights/, plots/
  data/pooled/       # patient-prefixed symlinks + manifest.json
  configs/           # dvf_view_config.json
```
