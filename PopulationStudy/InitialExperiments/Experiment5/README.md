# PopulationStudy — Experiment 5

Same **LIDC anatomy encoder** as Experiment 4 (shared `InitialExperiments/Experiment4/AnatomyAE/weights/anatomy_encoder.pth`). **SPARE split from Experiment 1.**

| Stage | Data |
|-------|------|
| Anatomy (shared E4) | LIDC 40-patient R231 pretrain |
| **Train** | P1, P2, P6, P7, P8, P9 (600 pairs) |
| **Hold-out** | P3, P4, P5 (300 pairs) |

## Train order

1. **Decoder-CRB** — frozen encoder swap (running first)
2. Encoder-CRB / Both-CRB — side-branch (after Decoder)

```bash
cd PopulationStudy/InitialExperiments/Experiment5
python scripts/build_pooled_dataset.py
PYTHONPATH=. PYTHONUNBUFFERED=1 python DecoderCRB/train_crb_dec_mse.py --gpu 0
# then:
PYTHONPATH=. PYTHONUNBUFFERED=1 python EncoderCRB/train_crb_enc_mse.py --gpu 0
PYTHONPATH=. PYTHONUNBUFFERED=1 python BothCRB/train_crb_both_mse.py --gpu 1
```

Compare hold-out QC to **E1** (no anatomy pretrain) and **E4** (same anatomy, E2 split).
