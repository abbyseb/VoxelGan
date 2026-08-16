# LeaveOut_5_9 — Decoder FiLM MSE

Hold-out SPARE phases **5 & 9**. UNetFiLM, MSE only, no D.

```bash
cd Dan2.0/DecoderFiLM
PYTHONPATH=../.. PYTHONUNBUFFERED=1 python train_film_mse.py --held_out 5,9 --run_dir LeaveOut_5_9
PYTHONPATH=../.. python scripts/qc_film_pairs.py \
  --ckpt LeaveOut_5_9/weights/film_mse_lo_05_09_generator.pth \
  --out_dir LeaveOut_5_9/plots/qc_test_mse --held_out 5,9
```
