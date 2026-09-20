# SPARE DVF characteristics — iso mm (fair vs TCIA)

Source: `PopulationStudy/data_iso` P1–P9 — **160³ @ 2 mm**, lung-centroid FOV.  
DVF stored as iso-grid voxels → **mm = voxel × 2**.

This is the apples-to-apples companion to TCIA  
`TCIA_4D-Lung_dvf_characteristics/` (also mm @ 2 mm).

| Metric (lung-masked ‖u‖) | SPARE iso (n=9) | TCIA (n=82) |
|--------------------------|----------------:|------------:|
| **01→06 mean** | **4.20 mm** | **6.25 mm** |
| median / min–max | 3.94 / 2.68–6.84 | 6.28 / 2.1–12.4 |
| mean all non-id pairs | 2.11 mm | 2.98 mm |

TCIA / SPARE mean 01→06 ≈ **1.5×** (physical mm, same grid recipe).

Files: `summary.json`, `metrics_per_patient.tsv`, `motion_ranking_01_to_06_mm.png`.

*Note:* old `DVFCharacteristics/` (128³ bbox voxels) is ranking-only — prefer this folder for mm.
