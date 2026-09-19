# VoxelMap_Experiments

Compare **Elastix VoxelMap** vs **4D Motion Synthesizer VoxelMap** on clinical scans.

## Run layout (per scan)

```
runs/CV_P3_V_01/
  elastix/                 # Arm A — GT volumes + Elastix DVFs
  synth_g160_a1_dec/       # Arm B — G160-A1 Decoder + µ (current best)
  synth_e3_both/           # legacy — Grid128 E3 Both + p1_p99
```

Compat symlinks (old flat names):

```
runs/elastix_CV_P3_V_01   → CV_P3_V_01/elastix
runs/synth_CV_P3_V_01     → CV_P3_V_01/synth_e3_both
```

| Arm folder | Volumes | Motion source | Norm |
|------------|---------|---------------|------|
| `elastix` | GT `GTVol_01…10` | Elastix (`sub_CT_06` → others) | — |
| `synth_g160_a1_dec` | Only `GTVol_06` → synth 10 | **G160-A1 Decoder** (linear, FOV) | **µ air/water** |
| `synth_e3_both` | Only `GTVol_06` → synth 10 | Grid128-E3 Both (cyclic) | p1_p99 |

Each arm folder contains:

```
<arm>/
  CV_P3_V_01/train/          # CT_*.mha, DRR bins, DVFs, Proj/
  ModelTraining/train|test/
  checkpoints_nofilm/
  logs/
  synth_meta.json            # synth arms only
```

## Prepare

```bash
cd VoxelMap_Experiments

# Arm A — Elastix (already prepared under runs/CV_P3_V_01/elastix)
python scripts/prepare_elastix_arm.py --scan-id CV_P3_V_01 --gpu 0

# Arm B — G160-A1 Decoder + µ (default --synth-id)
python scripts/prepare_synth_arm.py --scan-id CV_P3_V_01 --synth-id synth_g160_a1_dec --gpu 1

# Legacy E3 Both (optional rebuild)
python scripts/prepare_synth_arm.py --scan-id CV_P3_V_01 --synth-id synth_e3_both --gpu 1
```

## Train / eval NoFiLM VoxelMap

```bash
# --arm synth  aliases to synth_g160_a1_dec
python scripts/train_nofilm_arm.py --arm elastix --scan-id CV_P3_V_01 --gpu 0
python scripts/train_nofilm_arm.py --arm synth_g160_a1_dec --scan-id CV_P3_V_01 --gpu 0

python scripts/eval_arm.py --arm elastix --scan-id CV_P3_V_01
python scripts/eval_arm.py --arm synth_g160_a1_dec --scan-id CV_P3_V_01
```

## LEARN-GUI

```bash
python scripts/setup_learn_gui_run.py --arm elastix --link-learn-gui
python scripts/setup_learn_gui_run.py --arm synth_g160_a1_dec --link-learn-gui
```

## Data

```
data/staged/P3/CV_P3_V_01/     # SPARE symlinks (GTVol + Proj + masks)
data/spare_root/               # bridges SpareDVFs → stage layout
```
