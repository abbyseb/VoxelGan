"""X1 train (fair train-eval) vs val LNCC loss and the gap; old LNCC-only model in grey.  python3 plot_xray1_trainval.py OUT.png"""
import json, sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
L = Path(__file__).resolve().parents[1]
hx = json.loads((L / "run_A2_full160_aug_xray1/logs/history.json").read_text())
hu = json.loads((L / "run_A2_full160_aug_unsup_lncc/logs/history.json").read_text())
g = lambda h, k: [r[k] for r in h]; ex = g(hx, "epoch"); eu = g(hu, "epoch"); xl = (0.5, max(12.5, ex[-1] + 0.5))
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
a = ax[0]
a.plot(ex, g(hx, "train_eval_lncc_loss"), "g-s", label="X1 train (fair: no aug, end of epoch)")
a.plot(ex, g(hx, "val_lncc_loss"), "b-o", label="X1 val (4 unseen patients)")
a.plot(eu, g(hu, "train_lncc_loss"), ":", color="grey", label="old model train (epoch average, aug)")
a.plot(eu, g(hu, "val_lncc_loss"), "-", color="grey", label="old model val")
a.set_xlim(*xl); a.set_title("Train vs val LNCC loss ↓"); a.legend()
a = ax[1]
a.bar(ex, np.array(g(hx, "val_lncc_loss")) - np.array(g(hx, "train_eval_lncc_loss")), color="C2")
a.axhline(0, color="k", lw=.8); a.set_xlim(*xl); a.set_title("X1 gap: val − train  (0 = curves together)")
for a in ax: a.set_xlabel("epoch"); a.grid(alpha=.3)
fig.suptitle(f"X1 epoch {ex[-1]}/40"); fig.tight_layout(); fig.savefig(sys.argv[1], dpi=105)
for r in hx: print(r["epoch"], round(r["train_eval_lncc_loss"], 4), round(r["val_lncc_loss"], 4), round(r["val_lncc_loss"] - r["train_eval_lncc_loss"], 4))
