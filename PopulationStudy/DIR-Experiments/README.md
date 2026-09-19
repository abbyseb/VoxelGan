# DIR-Experiments

Zero-shot / transfer evaluation of SPARE-trained CRB DVF models on **DIR-Lab 4DCT**
(10 cases: `P1_DIR` … `P10_DIR`).

Source processed pack: `/home/abhishek/4DCT-LUNG-DIRLAB/processed`  
Local copy: [`data/`](data/) (`GTVol_01…10.mha` + landmarks + `metadata.json`).

## Why DIR-Lab

ClinicalExperiments QC uses Elastix DVFs as proxy GT. DIR-Lab adds **expert landmarks**,
so we can report **TRE in mm** — the registration community’s standard — without needing
clinical labels in training.

## Data notes (as copied)

| | |
|--|--|
| Patients | 10 (`P1_DIR`…`P10_DIR`) |
| Phases | 10 · `GTVol_01`=T00 … `GTVol_06`=T50 … `GTVol_10`=T90 |
| Intensity | HU (offset −1024) — **not** SPARE µ |
| Grid | Native (256²×94 … 512²×136); **not** 128³ |
| Landmarks | 300 pts T00↔T50; 75 pts T00…T50 (voxel indices, 0-based) |
| Lung mask | **Not present** — must generate before pack |

## Experiments

| Folder | Status | Summary |
|--------|--------|---------|
| [Experiment1](Experiment1/) | planned | **No retrain** · E6 Normal (128³ linear) weights · DIR QC + TRE |

E1 is inference-only (same spirit as Clinical E5): reuse ClinicalExperiments E6 Normal
checkpoints; adapt DIR CT into the SPARE-like inference space.

## Roadmap (high level)

1. **Pack** native DIR → 128³ SPARE-like tensors (mask → crop → resample → HU→µ), keep
   landmark transforms.
2. **Tier-0 benchmark** — fix TRE harness (`benchmark/verify_landmark_chain.py`), then
   `eval_dirlab_standard.py` (identity / Elastix / CRB).
3. **Experiment1 QC** — E6 Enc/Dec/Both, minmax + µ norms, 100-pair Elastix L1/cos + panels.
4. Later: Cyclic E6; iso-grid SPARE train; amplitude conditioning.
