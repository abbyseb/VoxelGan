# PopulationStudy — Experiment 4

Pretrained **anatomy encoder** on public chest CT (LIDC-IDRI), then freeze and splice into the CRB DVF nets. Same SPARE outer split as Experiment 2 / 3.

E1–E3 only ever see 4–6 SPARE identities. This experiment changes the **representation**: lung features from N=40 LIDC IDs, then a small DVF head on SPARE.

## Split

| Stage | Data | Role |
|-------|------|------|
| **Pretrain** | LIDC 40-patient subset (seed `20260819`) | masked CT autoencoder |
| LIDC val | 20% of those 40 (`lidc_val_split_seed`) | recon QC only |
| **SPARE train** | P1, P3, P4, P5 | DVF MSE (frozen encoder) |
| **SPARE hold-out** | P7, P9 | same QC as E2/E3 |

No patient-ID embedding. LIDC has no DVF / no phases.

## 128³ pack (same recipe as SPARE DVF library)

R231 mask bbox + 8 vx → resample **128³**. Intensities stay HU on disk; loaders **min-max per volume**, identical to `InitialExperiments/Experiment2/utilities/dataset.py`.

```bash
cd PopulationStudy/InitialExperiments/Experiment4
python scripts/pack_lidc_128.py
```

Output: `data/lidc128/<PatientID>/{CT,Mask_Lung}.npy` + `manifest.json`.

## How the encoder is used

### 1. Pretrain on LIDC (no SPARE, no DVF)

`AnatomyAutoencoder` = `AnatomyEncoder` + throwaway `AnatomyDecoder`.

- Input: one 128³ CT, 1 channel, min-max [0,1].
- Target: same CT.
- Loss: **lung-masked MSE** (`Mask_Lung.npy`). Table / chest wall do not dominate.
- Channel map matches Decoder-CRB: `1→16→32→32→64→64` (`ResidualBlock` + `AvgPool`).
- After pretrain: **keep encoder weights only**. Decoder is discarded.

This step is “learn what a lung looks like” across 40 scanners/patients.

### 2. SPARE DVF — Decoder-CRB (encoder **swap**)

Decoder-CRB’s encoder is already unconditioned residual blocks — same class as `AnatomyEncoder`.

- Load `AnatomyEncoder.state_dict()` into `UNetCRBDecoder.enc1 / down2…down5`.
- **Freeze** those layers (or later: tiny LR).
- Train only CRB decoder + `out_conv` on SPARE pairs (phase codes `[t_ref, t_tgt]`, lung-masked DVF MSE).
- Skips `s1…s4` still feed the decoder; they now carry LIDC-pretrained anatomy instead of random filters.

SPARE CTs are already 128³ under `PopulationStudy/data/P*/all/`. Same min-max loader. No LIDC files enter the DVF DataLoader.

### 3. SPARE DVF — Encoder-CRB and Both-CRB (frozen **side-branch**, code only)

**Not started in E4.** Wrappers exist (`EncoderCRB/train_crb_enc_mse.py`, `BothCRB/train_crb_both_mse.py`) but should not be launched unless asked.

Those encoders are **phase-CRB** end-to-end. You cannot drop `AnatomyEncoder` on top of `ConditionalResidualBlock(1,16)` — weights do not match.

- Keep the existing phase-CRB UNet.
- Run frozen `AnatomyEncoder` on the same reference CT in parallel.
- Pool the bottleneck (GAP) → `anatomy_vec` (dim D=32).
- Widen CRB `cond` from 2 → `2+D`: `[t_ref, t_tgt, anatomy…]`. First `Linear` becomes `Linear(2+D, 32)`.
- DVF training: phase path + decoder learn to use a stable anatomy code; encoder CRB still sees phase.

## Train (Decoder-CRB only)

```bash
cd PopulationStudy/InitialExperiments/Experiment4
python scripts/build_pooled_dataset.py
PYTHONPATH=. PYTHONUNBUFFERED=1 python scripts/train_ae.py --gpu 0
PYTHONPATH=. PYTHONUNBUFFERED=1 python DecoderCRB/train_crb_dec_mse.py --gpu 0
```

Encoder / Both (do **not** run):

```bash
# PYTHONPATH=. python EncoderCRB/train_crb_enc_mse.py --gpu 0
# PYTHONPATH=. python BothCRB/train_crb_both_mse.py --gpu 0
```

## Status

- [x] LIDC 40 CT + R231 masks
- [x] 128³ pack (`data/lidc128/`, 32 train / 8 val)
- [ ] AE train
- [ ] Decoder-CRB splice + SPARE train
- [ ] Encoder/Both side-branch **code ready, not started**
- [ ] P7/P9 QC vs E2/E3
