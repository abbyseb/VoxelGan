# G160-A1 — FOV aug, linear, 160³ iso (full P1–P9)

| | |
|--|--|
| **ID** | G160-A1 |
| **Phase** | linear (`cond_dim=2`) |
| **Grid** | `data_iso` via symlink → Iso-E2 `data/pooled_full/` |
| **FOV aug** | ¼ normal · ¼ half-FOV · ¼ CBCT noise · ¼ both (Grid128 E3) |
| **Val** | normal only |
| **Arch** | Encoder, Decoder, Both (**3 trains**) |

## Checkpoints

```
{Encoder,Decoder,Both}CRB/weights/crb_*_mse_iso_g160_fov_full_generator.pth
```

## Train

```bash
cd PopulationStudy/ClinicalExperiments/Grid160/Experiment1
bash scripts/run_train_all.sh              # all 3 arch
# or single:
PYTHONPATH=. python scripts/train_mse.py --arch decoder --full --gpu 0
```

Logs: `logs/train_{encoder,decoder,both}_g160_fov_full.log`

## Compare

| Set | FOV | Checkpoints |
|-----|-----|-------------|
| G160-A0 | No | `IsoExperiments/Experiment2/.../crb_*_iso_e2_full` |
| G160-A1 | Yes | this folder |

After train: norm sweep in [Experiment2/](../Experiment2/) + clinical QC.
