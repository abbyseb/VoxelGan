# Dan 2.0 — Both-side CRB (MSE only)

**Next** after Encoder CRB / Decoder CRB: **CRB on encoder + bottleneck + decoder** (`UNetCRBBoth`), direct 3-ch DVF, **MSE only, no D**.

| | EncoderCRB | DecoderCRB | BothCRB |
|--|------------|------------|---------|
| Encoder | CRB | plain | **CRB** |
| Bottleneck | CRB | plain | **CRB** |
| Decoder | plain | CRB | **CRB** |
| Output | direct DVF | direct DVF | direct DVF |

Leave-outs: **5&9**, **3&6**, **3,6,8**.

```bash
cd Dan2.0/BothCRB
PYTHONPATH=.. PYTHONUNBUFFERED=1 python train_crb_both_mse.py --held_out 5,9 --run_dir LeaveOut_5_9
PYTHONPATH=.. python scripts/qc_crb_both_pairs.py \
  --ckpt LeaveOut_5_9/weights/crb_both_mse_lo_05_09_generator.pth \
  --out_dir LeaveOut_5_9/plots/qc_test_mse --held_out 5,9
```
