# Dan 2.0 — Decoder CRB + Bottleneck CRB (MSE only)

Variant of [`DecoderCRB`](../DecoderCRB/): plain encoder, **CRB at bottleneck and decoder**.

| | DecoderCRB | **DecoderCRBBot** | BothCRB |
|--|------------|-------------------|---------|
| Encoder | plain | plain | CRB |
| Bottleneck | plain | **CRB** | CRB |
| Decoder | CRB | **CRB** | CRB |

```bash
cd Dan2.0/DecoderCRBBot
PYTHONPATH=.. PYTHONUNBUFFERED=1 python train_crb_dec_bot_mse.py --held_out 5,9 --run_dir LeaveOut_5_9
PYTHONPATH=.. python scripts/qc_crb_dec_bot_pairs.py \
  --ckpt LeaveOut_5_9/weights/crb_dec_bot_mse_lo_05_09_generator.pth \
  --out_dir LeaveOut_5_9/plots/qc_test_mse --held_out 5,9
```
