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

## Train / hold-out (TBD after visual review)

| Role | Patients |
|------|----------|
| Train (proposed) | P1–P5, P7 |
| Hold-out test (proposed) | P6, P8, P9 |

Update this table after inspecting slice-54 montages.
