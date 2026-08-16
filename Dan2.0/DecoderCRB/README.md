# Dan 2.0 — Decoder CRB (MSE only)

Same Dan 2.0 recipe (**MSE only, no D**), but CRBs are on the **decoder**, not the encoder.

| | Dan2.0 CRB | DecoderCRB |
|--|------------|------------|
| Encoder | CRB | **plain residual** (anatomy only) |
| Bottleneck | CRB | **plain** |
| Decoder | plain conv | **CRB** after skip concat |
| Output | direct 3-ch DVF | same (no SVF) |
| Loss | MSE | MSE |

## Layout

```text
Dan2.0/DecoderCRB/
  LeaveOut_5_9/
  LeaveOut_3_6/
  LeaveOut_3_6_8/
```

```bash
cd Dan2.0/DecoderCRB
PYTHONPATH=.. PYTHONUNBUFFERED=1 python train_crb_dec_mse.py --held_out 5,9 --run_dir LeaveOut_5_9
PYTHONPATH=.. python scripts/qc_crb_dec_pairs.py \
  --ckpt LeaveOut_5_9/weights/crb_dec_mse_lo_05_09_generator.pth \
  --out_dir LeaveOut_5_9/plots/qc_test_mse --held_out 5,9
```
