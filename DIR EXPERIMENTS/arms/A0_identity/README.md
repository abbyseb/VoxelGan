# A0 — Identity (static)

No motion: landmarks stay at source phase. **Lower bound** on TRE.

## Status

- [x] Identity TRE for cases 1–10 (300-pt T00→T50 + 75-pt T00→T50)
- [x] Matches DIR-Lab published 300-pt means (`dirlab_tre.py --check`)

## Results

| | |
|--|--|
| Cohort mean TRE (300-pt) | **8.46 mm** (see `results/summary.json`) |
| Outputs | [`results/identity_tre.tsv`](results/identity_tre.tsv), [`results/summary.json`](results/summary.json) |

```bash
cd "DIR EXPERIMENTS"
python3 scripts/dirlab_tre.py --check
# or regenerate tables:
# python3 -c '...'  # see logs / re-run arm script if added
```
