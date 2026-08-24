# CRBExperiments — Experiment 1

**First ablation:** Decoder-CRB + **oracle amplitude**, on the **P0-A common isotropic grid**.

| | |
|--|--|
| **Grid** | `data_iso/`: **2.0 mm** isotropic, **160³** (320 mm FOV), lung-centroid centred |
| Split | Train P1,P2,P6–P9 · hold-out **P3,P4,P5** |
| Arch | Decoder-CRB, `cond_dim=3`: `[θ_ref, θ_tgt, log(r_p)]` |
| `A_p` | q90 lung ‖u‖ in **mm** on extreme pair (unpadded mask) |
| Targets | Shape `û = u_vox / A_vox`; inference `u = shape · A_vox` |

```bash
cd PopulationStudy
python scripts/regrid_common_iso.py          # once → data_iso/
cd CRBExperiments/Experiment1
python scripts/compute_oracle_amplitude.py
python scripts/build_pooled_dataset.py
PYTHONPATH=. PYTHONUNBUFFERED=1 python DecoderCRB/train_crb_dec_amp_mse.py --gpu 0
```
