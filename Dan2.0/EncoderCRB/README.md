# Dan 2.0 — Encoder CRB (MSE only)

Baseline: **CRBs in the encoder + bottleneck**, plain decoder, direct 3-ch DVF, **MSE only, no D**. Generator = `UNetCRB` (same as original Dan 2.0).

| | EncoderCRB | DecoderCRB | BothCRB |
|--|------------|------------|---------|
| Encoder | **CRB** | plain | CRB |
| Bottleneck | **CRB** | plain | CRB |
| Decoder | plain | CRB | CRB |

## Results (already trained + QC)

| Leave-out | Directed L1 | Hold-out L1 |
|-----------|-------------|-------------|
| **5 & 9** | 0.219 | 0.230 |
| **3 & 6** | 0.218 | 0.255 |
| **3, 6, 8** | 0.263 | 0.310 |

Weights/QC under `LeaveOut_*/` (symlinked to original `Dan2.0/` and top-level `LeaveOut_*` runs).

## Re-train / QC

```bash
cd Dan2.0/EncoderCRB
PYTHONPATH=.. PYTHONUNBUFFERED=1 python train_crb_enc_mse.py --held_out 5,9 --run_dir LeaveOut_5_9
PYTHONPATH=.. python scripts/qc_crb_enc_pairs.py \
  --ckpt LeaveOut_5_9/weights/crb_enc_mse_lo_05_09_generator.pth \
  --out_dir LeaveOut_5_9/plots/qc_test_mse --held_out 5,9
```
