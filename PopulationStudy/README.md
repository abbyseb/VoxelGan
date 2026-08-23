# PopulationStudy — multi-patient SPARE MC

Train one model on DVFs from multiple patients; leave some patients out for test.

## Raw GT (this phase)

Symlinks to `GTVol_01..10.mha` + `Mask_*.mha` under `raw/P1`…`raw/P9`:

| ID | Case |
|----|------|
| P1 | `Data/.../MC_T_P1_NS` |
| P2 | `Data/.../MC_T_P2_SC` |
| P3–P8 | SpareDVFs Validation `MC_V_P*_NS_01` |
| P9 | `MC_V_P9_SC_01` |

## CT survey

Full GT is **450×220×450**. Fixed index **z=54** is **above the lungs** (empty) on every patient — use **lung mid-Z** plots for anatomy screening.

```bash
cd PopulationStudy
python scripts/viz_patient_cts.py --slice 54
```

| Plot | Meaning |
|------|---------|
| `plots/ct_survey/P0k_slice54.png` | Fixed z=54 (often empty) |
| `plots/ct_survey/P0k_lung_mid.png` | Mid-lung axial (use this) |
| `plots/ct_survey/all_patients_lung_mid.png` | Side-by-side compare |

**P6:** lung Z≈[174,334], mid **z=254** (`P6_lung_mid.png`).

## Mask padding check

```bash
cd PopulationStudy
# static survey PNGs → plots/mask_pad/
python scripts/viz_mask_padding.py --survey --pads 8,16

# PyQtGraph GUI (scroll slices, switch patients, live pad)
python scripts/viz_mask_padding.py --gui --patient P6 --init_pad 8
```

**GUI:** patient combo or `←`/`→` · mouse wheel / Z slider / ImageView timeline · pad slider or `[`/`]` · rotation combo or `R` · `Space` = lung mid-Z · **Save padded mask** or `Ctrl+S`.

Saved masks used for Elastix: `PaddedLungMasks/Mask_Lung_pad*_P*.mha`.

## DVF library (Elastix)

Masked B-spline registration (same params as My v1.0) using **padded** lung masks.

```bash
cd PopulationStudy
# all patients (one process each; skip existing pairs)
PYTHONUNBUFFERED=1 python scripts/prepare_dvf_library.py --patients P1 --skip_existing
```

Layout:

```
data/P1/all/   CT_01..10.npy  Mask_Lung.npy  01_to_01_pair.npy … 10_to_10_pair.npy
data/P1/train/  leave-out 5&9 bookkeeping
data/P1/val/
```

100 directed pairs per patient (10 identity zeros). 1 voxel = 1 mm after 128³ resample.

## Train / hold-out (random 6 of 9)

Drawn with seed `20260817` (`random.sample` of P1–P9).

| Role | Patients |
|------|----------|
| **Train** | P1, P2, P6, P7, P8, P9 |
| **Hold-out test** | P3, P4, P5 |

## Experiment 1

Self-contained leave-patient-out CRB MSE study (Encoder / Decoder / Both). See **`Experiment1/README.md`** and **`Experiment1/seed.json`**.

- Train: P1, P2, P6, P7, P8, P9 · Hold-out: P3, P4, P5

## Results / limitations

Leave-patient-out is **weaker than Dan 2.0 same-patient leave-phase-out**. E1 Decoder is useful on large-motion P3/P4; E2 mostly ties a zero DVF because P7/P9 barely move. Full write-up: **[`results.md`](results.md)**.

## Experiment 3

Episodic / zero-shot **training** (leave-one-train-patient-out), same outer split as E2. See **`Experiment3/README.md`**.

- Train pool: P1, P3, P4, P5 · inner default zero-out: **P1** (or rotate)
- True hold-out: P7, P9

## Experiment 4

LIDC-IDRI anatomy autoencoder → freeze encoder → splice into CRB DVF nets. Same SPARE hold-out as E2/E3 (P7, P9). See **`Experiment4/README.md`**.

```bash
cd PopulationStudy/Experiment4
python scripts/pack_lidc_128.py
```


## Experiment 2

Fixed split — see **`Experiment2/README.md`** and **`Experiment2/seed.json`**.

- **Train:** P1, P3, P4, P5 (400 pairs)
- **Hold-out test:** P7, P9 (200 pairs)

```bash
cd PopulationStudy/Experiment2
python scripts/build_pooled_dataset.py
PYTHONPATH=. python EncoderCRB/train_crb_enc_mse.py --gpu 0
```
