# LeaveOut_3_6_8 — Decoder FiLM MSE

Hold-out SPARE phases **3, 6 & 8**. UNetFiLM, MSE only, no D.

```bash
cd Dan2.0/DecoderFiLM
PYTHONPATH=../.. PYTHONUNBUFFERED=1 python train_film_mse.py --held_out 3,6,8 --run_dir LeaveOut_3_6_8
PYTHONPATH=../.. python scripts/qc_film_pairs.py \
  --ckpt LeaveOut_3_6_8/weights/film_mse_lo_03_06_08_generator.pth \
  --out_dir LeaveOut_3_6_8/plots/qc_test_mse --held_out 3,6,8
```
