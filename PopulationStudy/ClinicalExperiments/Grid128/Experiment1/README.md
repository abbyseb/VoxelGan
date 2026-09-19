# ClinicalExperiments — Experiment 1

**Full SPARE MC train → zero-shot Clinical Varian test.** No amplitude conditioning.

## Train (SPARE MC)

| Item | Value |
|------|-------|
| Patients | **P1–P9** (all) |
| Pairs | 900 directed (100 per patient, identity included) |
| Val | 10% pair split from those 900 (seed TBD in `seed.json`) |
| Grid | **128³** lung-bbox resample (same as InitialExperiments E1/E2) |
| Data root | `PopulationStudy/data/P*/all/` |
| Architectures | EncoderCRB, DecoderCRB, BothCRB |

Copy/adapt train scripts from [`../../InitialExperiments/Experiment1/`](../../InitialExperiments/Experiment1/)
with pooled **train = all patients**, no hold-out patient in train/val.

## Test (Clinical Varian / Elekta)

| Item | Value |
|------|-------|
| Data | `PopulationStudy/varian/` (Varian) · SpareDVFs `ClinicalElektaDatasets` (Elekta) |
| First case | `CV_P1_V_01` / `CE_P1_V_01` (then P2–P5) |
| GT | Elastix on `Evaluation/.../GTVol_*` (must be built — see parent README) |
| Inference | Frozen ckpt from full SPARE train; same loader norm as E1 |

Hold-out patients do **not** exist in this design — the domain shift is SPARE sim → clinical CBCT/FDK.

## Clinical DVF prep

```bash
cd PopulationStudy/ClinicalExperiments
PYTHONUNBUFFERED=1 /path/to/LEARN-GUI/.venv/bin/python scripts/prepare_clinical_dvf_library.py \\
  --scan CV_P1_V_01 --skip_existing
# same for Elekta:
#   --scan CE_P1_V_01
```

Output: `Experiment1/data/{CV|CE}_P*_V_01/all/` · Elastix: `My v1.0/configs/elastix_bspline_masked.txt`

## QC layout

**Varian** panels:

`{Encoder|Decoder|Both}CRB/plots/qc_varian/P{n}/CV_P{n}_V_01/`

**Elekta** panels:

`{Encoder|Decoder|Both}CRB/plots/qc_elekta/P{n}/CE_P{n}_V_01/`

(`summary.json`, `metrics.tsv`, and 100 pair PNGs per scan.)

## EXPT5_SETTINGS (E1 weights + E5 µ intensity norm)

Same Enc/Dec/Both checkpoints, but CT scaled with Experiment 5 **µ air/water**
instead of min–max. Inference-only comparison.

See [`EXPT5_SETTINGS/README.md`](EXPT5_SETTINGS/README.md) · panels under
`EXPT5_SETTINGS/{Encoder,Decoder,Both}CRB/plots/qc_{varian,elekta}/`.

```bash
PYTHONPATH=. python EXPT5_SETTINGS/scripts/qc_e5_settings.py --gpu 1
```

## Status

- [x] Clinical Elastix + 128³ pack for Varian `CV_P1…P5_V_01`
- [x] Clinical Elastix + QC for Elekta `CE_P1…P5_V_01`
- [x] Pooled full-SPARE dataset + `seed.json`
- [x] Train Encoder / Decoder / Both
- [x] Zero-shot QC Varian + Elekta (default min–max)
- [ ] EXPT5_SETTINGS QC Enc/Dec/Both × Varian+Elekta *(running)*

## Commands

```bash
cd PopulationStudy/ClinicalExperiments/Experiment1
# Varian QC
PYTHONPATH=. python scripts/qc_varian.py --arch encoder --scan CV_P1_V_01 --all --gpu 1

# Elekta: prep Elastix + QC all patients P1–P5
cd .. && GPU=1 bash Experiment1/scripts/prepare_and_qc_elekta.sh
# log: Experiment1/logs/prepare_and_qc_elekta.log
```
