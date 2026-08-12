# LeaveOut_3_6

Dan 2.0 recipe (UNetCRB, lung-masked **MSE only**, no D), hold-out SPARE phases **3 & 6**.

- **3** ≈ mid-cycle gap → interpolation
- **6** = PE (if 06 is PE) → hard extreme

```bash
cd LeaveOut_3_6
PYTHONUNBUFFERED=1 python train_crb_mse.py
PYTHONPATH=../Dan2.0 python scripts/qc_crb_pairs.py
```

Shares `utilities/` / `configs/` with `Dan2.0/` via symlink.
