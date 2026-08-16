# Dan 2.0 — Decoder FiLM (MSE only)

Same **Dan 2.0 recipe** (no discriminator, lung-masked **MSE vs Elastix DVF only**), but generator = **UNetFiLM** (Phase-MLP FiLM on bottleneck + decoder + SVF), not UNetCRB.

## Layout

```text
Dan2.0/DecoderFiLM/
  train_film_mse.py
  scripts/qc_film_pairs.py
  LeaveOut_5_9/     # hold-out phases 5 & 9
  LeaveOut_3_6/     # hold-out 3 & 6
  LeaveOut_3_6_8/   # hold-out 3, 6 & 8
```

## vs Dan 2.0 CRB

| | Dan2.0 (CRB) | DecoderFiLM |
|--|--------------|-------------|
| G | UNetCRB (direct DVF) | **UNetFiLM + SVF** |
| D | none | none |
| Loss | MSE | MSE |
| Splits | 5&9 (+ top-level LeaveOut_3_6 / _3_6_8) | same three here |

## Train / QC

From `Dan2.0/DecoderFiLM/`:

```bash
PYTHONPATH=../.. PYTHONUNBUFFERED=1 python train_film_mse.py --held_out 5,9 --run_dir LeaveOut_5_9
PYTHONPATH=../.. PYTHONUNBUFFERED=1 python train_film_mse.py --held_out 3,6 --run_dir LeaveOut_3_6
PYTHONPATH=../.. PYTHONUNBUFFERED=1 python train_film_mse.py --held_out 3,6,8 --run_dir LeaveOut_3_6_8

PYTHONPATH=../.. python scripts/qc_film_pairs.py \
  --ckpt LeaveOut_5_9/weights/film_mse_lo_05_09_generator.pth \
  --out_dir LeaveOut_5_9/plots/qc_test_mse --held_out 5,9
```

Logs: `LeaveOut_*/plots/train_film_mse.log`. Weights gitignored.
