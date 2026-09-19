# DIR-Experiments — Experiment 2 (iso grid)

Repack DIR-Lab onto the **same P0-A iso frame** as `PopulationStudy/data_iso` (2 mm, 160³) so TRE and DVF metrics are in **mm** and comparable to [`IsoExperiments/Experiment2`](../../IsoExperiments/Experiment2/) training.

## Pipeline

1. **`pack_dirlab_iso.py`** — lungmask → centroid-centred 160³ @ 2 mm, HU CT, landmarks in iso indices
2. **`prepare_dir_dvf_library_iso.py`** — Elastix 10×10 pairs; DVF stored as **iso-voxels** (÷ 2 mm)
3. QC / TRE eval (TODO: wire `qc_e6_dirlab.py` to `packed_iso` + Iso-E2 weights)

## Commands

```bash
cd PopulationStudy/DIR-Experiments/Experiment2
VENV=/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python

$VENV scripts/pack_dirlab_iso.py
$VENV scripts/prepare_dir_dvf_library_iso.py --patients P1_DIR,...,P10_DIR --skip_existing
```

Output: `packed_iso/P*_DIR/all/` (same layout as `data_iso/P*/all/`).

## Model weights

Use checkpoints from `IsoExperiments/Experiment2/{Encoder,Decoder,Both}CRB/weights/crb_*_mse_iso_e2_generator.pth`.
