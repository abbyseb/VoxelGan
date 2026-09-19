# Grid160 — linear CRB @ 2 mm 160³ iso

**Phase:** linear (`cond_dim=2`) · **Grid:** `data_iso` (2 mm, 160³) · **Train norm:** per-volume minmax  
**Norm sweep at QC:** minmax + percentiles + µ (inference-only, like Grid128 E5)

## 9-model train matrix (3 conditions × 3 architectures)

| ID | FOV aug | Split | Enc | Dec | Both |
|----|---------|-------|-----|-----|------|
| **G160-A0** | No | full P1–P9 | ✅ | ✅ | ✅ |
| **G160-A0h** | No | E1 hold-out | ✅ | ✅ | ✅ |
| **G160-A1** | Yes (E3 recipe) | full P1–P9 | ✅ | ✅ | ✅ |

- **A0 / A0h** = [`IsoExperiments/Experiment2/`](../../IsoExperiments/Experiment2/) — checkpoints `crb_*_mse_iso_e2_full` / `crb_*_mse_iso_e2`  
- **A1** = [Experiment1/](Experiment1/) — checkpoints `crb_*_mse_iso_g160_fov_full`  
- **QC / norm sweep** = [Experiment2/](Experiment2/) (planned, metrics only)

> Original A1=Decoder-only + A2=Both-only with FOV are merged into **one FOV row × all 3 arch** (3 trains, not 6).

## Experiments

| Folder | Role |
|--------|------|
| [Experiment0/](Experiment0/) | Pointer + QC for **G160-A0 / A0h** (Iso-E2, done) |
| [Experiment1/](Experiment1/) | **Train G160-A1** — FOV aug, Enc/Dec/Both |
| [Experiment2/](Experiment2/) | Norm sweep QC on A0/A0h/A1 checkpoints (metrics only) |

## Train commands (A1 only — 3 runs)

```bash
cd PopulationStudy/ClinicalExperiments/Grid160/Experiment1
bash scripts/run_train_all.sh          # enc → dec → both, full P1–P9
# or one arch:
PYTHONPATH=. python scripts/train_mse.py --arch decoder --full --gpu 0
```

## Success criteria

1. Hold-out SPARE (P3–P5) L1/mm not worse than A0h by much  
2. Clinical zero-shot (Varian/Elekta) cos ↑ vs A0, especially Elekta  
3. Best checkpoint + µ norm → `VoxelMap_Experiments` synth arm

## Related

- Amplitude (next): [`../../CRBExperiments/`](../../CRBExperiments/)  
- Archived 128³: [`../Grid128/`](../Grid128/)
