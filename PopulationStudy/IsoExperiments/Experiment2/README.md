# IsoExperiments Experiment 2 — physically comparable CRB (Run 0)

Train **Encoder**, **Decoder**, and **Both** CRB on the P0-A common isotropic grid so labels and metrics are in **mm** and comparable across patients.

| | |
|--|--|
| **Grid** | `PopulationStudy/data_iso/` — **2.0 mm** isotropic, **160³** |
| **Split** | **E1 split:** train P1,P2,P6–P9 · hold-out P3–P5 · **Full:** all P1–P9 |
| **Target** | Raw Elastix DVF in **iso-voxels** (1 vx = 2 mm) |
| **Loss** | Lung-masked MSE (iso-vox²); QC reports L1 in **mm** |
| **Arch** | `cond = [θ_ref, θ_tgt]` linear, no amplitude conditioning |

## Why this experiment

Without a shared mm frame, per-patient anisotropic 128³ crops make MSE and L1 **not physically comparable** — a “better” loss can be grid noise. This is the baseline before amplitude conditioning (see `CRBExperiments/`).

## Train

```bash
cd PopulationStudy/IsoExperiments/Experiment2
python scripts/build_pooled_dataset.py

# one architecture
PYTHONPATH=. PYTHONUNBUFFERED=1 python EncoderCRB/train_crb_enc_mse.py --gpu 0 --epochs 100
PYTHONPATH=. PYTHONUNBUFFERED=1 python DecoderCRB/train_crb_dec_mse.py --gpu 0 --epochs 100
PYTHONPATH=. PYTHONUNBUFFERED=1 python BothCRB/train_crb_both_mse.py --gpu 0 --epochs 100

# or all three sequentially
bash scripts/run_train_all.sh
```

Checkpoints: `{Encoder,Decoder,Both}CRB/weights/crb_*_mse_iso_e2_generator.pth` (E1 split)

### Full P1–P9

```bash
python scripts/build_pooled_dataset.py --full
PYTHONPATH=. PYTHONUNBUFFERED=1 python EncoderCRB/train_crb_enc_mse.py --gpu 0 --epochs 100 --full
PYTHONPATH=. PYTHONUNBUFFERED=1 python DecoderCRB/train_crb_dec_mse.py --gpu 0 --epochs 100 --full
PYTHONPATH=. PYTHONUNBUFFERED=1 python BothCRB/train_crb_both_mse.py --gpu 0 --epochs 100 --full
# or
bash scripts/run_train_full.sh
```

Checkpoints: `crb_*_mse_iso_e2_full_generator.pth`

## Hold-out QC (mm)

```bash
PYTHONPATH=. python scripts/qc_pairs.py \
  --arch both \
  --ckpt BothCRB/weights/crb_both_mse_iso_e2_generator.pth \
  --out_dir BothCRB/plots/qc_holdout \
  --gpu 0 --panels extreme
```

## DIR iso repack

For zero-shot DIR eval on the same grid, see [`../../DIR-Experiments/Experiment2/`](../DIR-Experiments/Experiment2/).
