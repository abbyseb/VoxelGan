# LeaveOut_3_6 — Decoder FiLM MSE

Hold-out SPARE phases **3 & 6** (interp gap + PE). UNetFiLM, MSE only, no D.

```bash
cd Dan2.0/DecoderFiLM
PYTHONPATH=../.. PYTHONUNBUFFERED=1 python train_film_mse.py --held_out 3,6 --run_dir LeaveOut_3_6
PYTHONPATH=../.. python scripts/qc_film_pairs.py \
  --ckpt LeaveOut_3_6/weights/film_mse_lo_03_06_generator.pth \
  --out_dir LeaveOut_3_6/plots/qc_test_mse --held_out 3,6
```
