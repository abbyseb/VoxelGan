# A1 — Oracle VoxelMap (patient DIR-Lab 4D-CT) — **R3 redo**

Fresh arm after incorrect-orbit archive. Recipe:

1. Stage native `P{N}_DIR` → LEARN d2m → **R3 reorient** (SI on Y, AP+SI flip, isocentre)
2. Keep-HU downsample 128³
3. DRR: `Geometry.xml` (= `Geometry_SPARE.xml`, OffsetY −2) + MC/Varian half-fan
4. Lung-masked Elastix BSpline (grid 16) on R3 `sub_CT`
5. NoFiLM VoxelMap; TRE via `eval_a1_tre.py --r3`

Archived wrong-orbit runs: [`../Incorrect DRR/A1_oracle_dirlab/`](../Incorrect%20DRR/A1_oracle_dirlab/)

```bash
cd "DIR EXPERIMENTS"
# Elastix-only smoke
python scripts/prepare_a1_case.py --case 1 --gpu 1 --skip-drr --skip-prep
python scripts/eval_a1_tre.py --case 1 --r3
# Full prepare
python scripts/prepare_a1_case.py --case 1 --gpu 1
```
