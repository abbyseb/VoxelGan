# LeaveOut_3_6_8

Dan 2.0 recipe (UNetCRB, lung-masked **MSE only**, no D), hold-out SPARE phases **3, 6, 8**.

Harder val: PE (6) + two other gaps (3, 8).

```bash
cd LeaveOut_3_6_8
PYTHONUNBUFFERED=1 python train_crb_mse.py
PYTHONPATH=../Dan2.0 python scripts/qc_crb_pairs.py
```

Shares `utilities/` / `configs/` with `Dan2.0/` via symlink.
