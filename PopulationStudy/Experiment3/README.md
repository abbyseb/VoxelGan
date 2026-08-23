# PopulationStudy — Experiment 3

**Train under the zero-shot condition**, not only evaluate under it. Same outer split as Experiment 2.

## Split (same as Experiment 2)

| Role | Patients | Pairs |
|------|----------|-------|
| **Train pool** | P1, P3, P4, P5 | 400 (360 train + 40 val @ 10%) |
| **True hold-out test** | P7, P9 | 200 — never in train or val |

Seeds: `seed.json` (`20260819` for val/RNG). Patient assignment copied from E2.

**P1 is the default inner zero-out** among the train pool: when the episode is “fixed P1”, support = P3/P4/P5 and query = P1. The alternative is **rotate** P1→P3→P4→P5 as query each cycle (still never using P7/P9).

## Why (vs E1 / E2)

E1/E2 **jointly** fit all train patients, then hope P7/P9 work. E2 showed that hope is weak (hold-out ≈ zero DVF).

E3 makes **cross-patient transfer** part of the training signal:

Each episode:

1. Pick a **query** patient from the train pool (P1 by default, or rotate).
2. **Support** batch = the other train patients (anatomy the model is allowed to fit this step).
3. **Query** batch = the excluded patient (mini zero-shot test).
4. Loss = `L_support + λ L_query` (both lung-masked MSE vs Elastix).
5. Backprop that total so doing well on the *unseen-this-step* patient is rewarded.

True hold-out remains **P7, P9** — same QC as E2.

This is **first-order episodic** training (one graph, mixed loss). It is **not** full MAML (inner SGD on support, then query loss through the adapted weights) unless we add that next.

There is still **no patient-ID embedding**. “Exclude from the anatomy-conditioning batch” means those patients’ CTs/DVFs are not in the **support** batch that step; query CTs still go through the same UNet (phase codes only).

## Layout

Self-contained copy of E2 (`networks/`, `losses/`, `utilities/`, `scripts/`). Joint E2-style loop still exists as `scripts/train_mse.py`. **E3 default is** `scripts/train_meta.py`.

```
Experiment3/
  seed.json
  README.md
  scripts/train_meta.py     # episodic leave-one-train-patient-out
  scripts/train_mse.py      # joint baseline (E2 clone)
  EncoderCRB/ DecoderCRB/ BothCRB/
  data/pooled/              # same prefix rule as E1/E2
```

## Build pooled data

```bash
cd PopulationStudy/Experiment3
python scripts/build_pooled_dataset.py
```

## Train (not started — waiting on next-step calls)

```bash
cd PopulationStudy/Experiment3
PYTHONPATH=. PYTHONUNBUFFERED=1 python scripts/train_meta.py \
  --arch encoder --gpu 0 --query_schedule rotate --lambda_query 2.5
```

Weights: `crb_{enc,dec,both}_mse_pop_e3_generator.pth`

## QC (P7, P9)

Same 2×4 panel as E2 (`scripts/qc_pairs.py`).
