# A0 — Identity (static)

No motion: landmarks stay at source phase. **Lower bound** on TRE.

## Hard rule

Primary KPI = **75-point** Sampled4D, T00→T50 (not 300-pt).

## Status

- [x] Identity TRE for cases 1–10 (**75-pt** T00→T50)
- [x] Pack QA: `dirlab_tre.py --check` still verifies published 300-pt means (not an arm score)

## Results

| | |
|--|--|
| Cohort mean TRE (**75-pt**) | **8.69 mm** (see `results/summary.json`) |
| Case 1 identity (75-pt) | **3.91 mm** |
| Outputs | [`results/identity_tre.tsv`](results/identity_tre.tsv), [`results/summary.json`](results/summary.json) |

```bash
cd "DIR EXPERIMENTS"
python3 scripts/dirlab_tre.py --case 1 --set 75   # identity row only if no --dvf/--pred
python3 scripts/dirlab_tre.py --check             # pack QA
```
