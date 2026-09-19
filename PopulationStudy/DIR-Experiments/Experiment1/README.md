# DIR-Experiments — Experiment 1

**Inference-only.** Take **ClinicalExperiments Experiment 6 Normal** (128³ linear CRB)
weights and QC them on DIR-Lab `P1_DIR`…`P10_DIR`. **No retrain.**

## Goal

Establish a DIR-Lab baseline for the best full-volume SPARE model (E6 Normal), with
**TRE in mm** as the primary metric (landmarks), plus visual warp panels.

## Model

| Arch | Checkpoint (symlink) |
|------|----------------------|
| Encoder | `weights/encoder_e6_normal.pth` → E6 Normal Enc |
| Decoder | `weights/decoder_e6_normal.pth` → E6 Normal Dec |
| Both | `weights/both_e6_normal.pth` → E6 Normal Both |

Nets: InitialExperiments linear CRB (`cond_dim=2`), same as Clinical E6 Normal / E1.

## Why prep is required before QC

DIR `data/` is **not** network-ready:

| Issue | DIR native | E6 expects |
|-------|------------|------------|
| Size | 256²×94 … 512²×136 | **128³** |
| Intensity | HU (−1024…) | SPARE-like **µ** (~−0.02…0.03) |
| Lung mask | missing | needed for crop + (optional) Elastix |
| GT DVF | none | optional; landmarks preferred |

Landmarks are voxel indices in **native** space — any pack must save the
bbox/resample transform so TRE can be computed in **mm** on the original grid
(or consistently in packed space then scaled by spacing).

## Plan (ordered)

### 0. Data (done)
- [x] Copy `/home/abhishek/4DCT-LUNG-DIRLAB/processed` → `../data/`
- [x] Symlink E6 Normal Enc/Dec/Both weights

### 1. Pack pipeline (`scripts/pack_dirlab.py`)
Per patient:
1. **Lung mask** on `GTVol_01` (e.g. `lungmask` R231 — same idea as LIDC prep)
2. Pad mask (ClinicalExperiments EDT pad recipe)
3. Crop to lung bbox → resample CT + mask → **128³**
4. **HU → µ**: `(HU + 1024) / 1000 * 0.02` style (match SPARE / clinical pack)
5. Write `Experiment1/packed/P*_DIR/all/CT_01…10.npy`, `Mask_Lung.npy`, `pack_meta.json`
6. Map landmarks through the same crop/resample → packed voxel coords + keep native
   coords + spacing for TRE in mm

Skip Elastix for E1 v1 (landmarks only). Add later if we want L1/cos vs Elastix.

### 2. QC script (`scripts/qc_e6_dirlab.py`)
- **Clinical-style:** 100 directed pairs per patient, L1/cos vs Elastix GT, 7 panel PNGs
- Same panel pairs as E6: `01_to_02`, `01_to_06`, `01_to_10`, `06_to_01`, …
- Requires `prepare_dir_dvf_library.py` first (`*_pair.npy` in `packed/.../all/`)
- Outputs: `plots/qc_{arch}/{norm}/P*_DIR/` — `summary.json`, `metrics.tsv`, `*.png`

### 2b. Elastix GT (`scripts/prepare_dir_dvf_library.py`)
- Masked B-spline on packed 128³ µ volumes (same param as clinical)
- 10×10 pairs → `packed/P*_DIR/all/XX_to_YY_pair.npy`

### 3. Report
- Per-patient + cohort TRE table (Both × µ first; then Enc/Dec; minmax vs µ)
- Compare to published DIR-Lab baselines if useful (context only)

## Out of scope for E1
- Training / fine-tuning on DIR
- Cyclic E6, E3 FOV-aug, E5-as-weights (those can be Experiment2+)
- Full 10×10 Elastix DVF library (optional Experiment1b)

## Commands (once pack exists)

```bash
cd PopulationStudy/DIR-Experiments/Experiment1
# pack (to implement)
PYTHONPATH=. python scripts/pack_dirlab.py --gpu 0
# QC (to implement)
PYTHONPATH=. python scripts/qc_e6_dirlab.py --arch both --norm mu --gpu 1
```

## Status

- [x] Folder + weight symlinks + data copy
- [x] Pack script (`scripts/pack_dirlab.py`) + QC (`scripts/qc_e6_dirlab.py`)
- [ ] Pack run (lungmask → 128³ µ)
- [ ] QC + TRE
- [ ] Cohort summary
