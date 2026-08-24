# Experiment 4 — inhale/exhale `δ_tgt`

Same as Experiment 3 (cyclic + oracle amp), plus **δ_tgt** ∈ {+1, −1} for the target phase.

| | |
|--|--|
| `δ = +1` | inhaling (phases 02 → peak, usually 06/07) |
| `δ = −1` | exhaling (peak → 10 and phase 01 = EE) |
| `cond_dim` | 6 |

Limb table: [`../phase_limbs.json`](../phase_limbs.json) (from `01→phase` motion peak on `data_iso`).

```bash
cd PopulationStudy/CRBExperiments/Experiment4
PYTHONPATH=. PYTHONUNBUFFERED=1 python DecoderCRB/train_crb_dec_amp_mse.py --gpu 0 --epochs 100
```
