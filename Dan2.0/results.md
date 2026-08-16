# Dan 2.0 results (MSE only, no D)

P1 leave-phase-out. Metric = lung-masked mean |pred − Elastix| L1 on 128³ (1 voxel = 1 mm). Lower is better.

## Hold-out L1

| Model | 5 & 9 | 3 & 6 | 3, 6, 8 |
|-------|-------|-------|---------|
| **Decoder FiLM** | **0.196** | **0.219** | **0.249** |
| Decoder CRB | 0.204 | 0.241 | 0.282 |
| Both CRB | 0.217 | 0.228 | 0.260 |
| Encoder CRB | 0.230 | 0.255 | 0.310 |

## Directed L1 (from the 5 & 9 run)

| Model | Directed L1 |
|-------|-------------|
| Decoder CRB | 0.185 |
| Decoder FiLM | 0.187 |
| Both CRB | 0.203 |
| Encoder CRB | 0.219 |

**Best:** Decoder FiLM. **Worst:** Encoder CRB.
