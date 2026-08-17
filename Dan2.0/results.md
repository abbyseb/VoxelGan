# Dan 2.0 results (MSE only, no D)

P1 leave-phase-out. Metric = lung-masked mean |pred − Elastix| L1 on 128³ (1 voxel = 1 mm). Lower is better.

## Hold-out L1

| Model | 5 & 9 | 3 & 6 | 3, 6, 8 |
|-------|-------|-------|---------|
| **Decoder FiLM** | **0.196** | **0.219** | 0.249 |
| Decoder CRB | 0.204 | 0.241 | 0.282 |
| Decoder CRB + Bot | 0.203 | 0.226 | **0.248** |
| Both CRB | 0.217 | 0.228 | 0.260 |
| Encoder CRB | 0.230 | 0.255 | 0.310 |

Decoder CRB + Bot = plain encoder, **CRB bottleneck + decoder**.

## Directed L1 (from the 5 & 9 run)

| Model | Directed L1 |
|-------|-------------|
| Decoder CRB + Bot | 0.179 |
| Decoder CRB | 0.185 |
| Decoder FiLM | 0.187 |
| Both CRB | 0.203 |
| Encoder CRB | 0.219 |

**Best hold-out:** Decoder FiLM on 5&9 and 3&6; Decoder CRB + Bot edges FiLM on 3,6,8 (0.248 vs 0.249). **Worst:** Encoder CRB.

## Negative Jacobian % (folding)

`det(I+∇u) < 0` via finite differences (same as `scripts/qc_leave_phase_out.py`). Mean over 90 directed pairs.

| Model | Dir full % | Dir lung % | Leave-out lung % |
|-------|------------|------------|------------------|
| Encoder CRB (all splits) | ≤0.006 | **0.000** | **0.000** |
| Decoder CRB (all splits) | ≤0.002 | **0.000** | **0.000** |
| Both CRB (all splits) | ≤0.001 | **0.000** | **0.000** |
| Decoder FiLM (all splits) | **0.000** | **0.000** | **0.000** |
| Decoder CRB + Bot (5&9, 3&6) | ≤0.024 | **0.000** | **0.000** |

Lung-masked folding is ~0% everywhere. Tiny full-volume % is outside the lung. Full table: [`jacobian_summary.md`](jacobian_summary.md).
