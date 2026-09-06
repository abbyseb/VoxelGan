# G160-A0 / A0h — no FOV (done)

Checkpoints **copied and renamed** from Iso-E2 into `{Encoder,Decoder,Both}CRB/weights/`:

| Arm | Split | Filename pattern |
|-----|-------|------------------|
| **G160-A0** | full P1–P9 | `crb_*_mse_iso_g160_a0_full_generator.pth` |
| **G160-A0h** | E1 hold-out | `crb_*_mse_iso_g160_a0h_generator.pth` |

**Note:** `both` A0h was never trained in Iso-E2 (encoder + decoder A0h only).

```bash
bash scripts/setup_checkpoints.sh   # refresh copies from Iso-E2
PYTHONPATH=. python scripts/qc_clinical.py --arms a0_full,a0h --norm both --no-panels --gpu 0
```

Summaries: `plots/qc_summary/`
