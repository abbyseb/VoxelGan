"""X1 (CT + 1 AP DRR) vs LNCC-only progress figure. Reads the run histories and the two TRE watcher JSONs.
  python3 plot_xray1.py OUT.png"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L = Path(__file__).resolve().parents[1]; A = Path(__file__).resolve().parent
hx = json.loads((L / "run_A2_full160_aug_xray1/logs/history.json").read_text())
hu = json.loads((L / "run_A2_full160_aug_unsup_lncc/logs/history.json").read_text())
tx = json.loads((A / "xray1_tre.json").read_text())["xray1"] if (A / "xray1_tre.json").exists() else {}
tu = json.loads((A / "unsup_lncc_tre.json").read_text())["unsup_lncc"]
ex = [r["epoch"] for r in hx]; eu = [r["epoch"] for r in hu]
g = lambda h, k: [r[k] for r in h]
fig, ax = plt.subplots(2, 2, figsize=(13, 9))

a = ax[0, 0]
ep = sorted(tu, key=int); a.plot([int(e) for e in ep], [np.mean(tu[e]["dir_plain"]) for e in ep], "-", color="grey", label="LNCC-only (no X-ray)")
for mode, c, s in (("real", "C0", "o-"), ("zero", "C1", "s--"), ("shuffled", "C3", "^--")):
    ep = sorted([e for e in tx if mode in tx[e]], key=int)
    a.plot([int(e) for e in ep], [np.mean(tx[e][mode]["dir_plain"]) for e in ep], s, ms=4, color=c, label=f"X1, {mode} X-ray")
a.set_title("DIR-Lab TRE300 (plain) ↓"); a.set_ylabel("mm"); a.legend()

a = ax[0, 1]
a.plot(eu, g(hu, "val_lncc_loss"), "-", color="grey", label="LNCC-only val")
a.plot(ex, g(hx, "train_eval_lncc_loss"), "g-s", ms=4, label="X1 train-eval (no aug, end of epoch)")
a.plot(ex, g(hx, "val_lncc_loss"), "C0-o", ms=4, label="X1 val, real X-ray")
a.plot(ex, g(hx, "val_lncc_loss_shuffled_xray"), "C3--^", ms=4, label="X1 val, wrong X-ray")
a.set_title("LNCC loss ↓"); a.legend()

a = ax[1, 0]
a.plot(ex, np.array(g(hx, "val_lncc_loss")) - np.array(g(hx, "train_eval_lncc_loss")), "C2-o", ms=4, label="X1: val − train-eval (fair gap)")
a.plot(eu, np.array(g(hu, "val_lncc_loss")) - np.array(g(hu, "train_lncc_loss")), "-", color="grey",
       label="LNCC-only: val − train (old, epoch-average train)")
a.axhline(0, color="k", lw=0.8); a.set_title("Train/val gap → 0 is the goal"); a.legend()

a = ax[1, 1]
a.plot(eu, g(hu, "val_fold_pct"), "-", color="grey", label="LNCC-only")
a.plot(ex, g(hx, "val_fold_pct"), "C0-o", ms=4, label="X1")
a.set_title("TCIA val folding (% det J ≤ 0)"); a.legend()

for a in ax.flat:
    a.set_xlabel("epoch"); a.grid(alpha=.3)
    if ex: a.set_xlim(0.5, max(ex[-1] + 0.5, 10.5))
fig.suptitle(f"X1 (CT + 1 AP X-ray, from scratch) vs LNCC-only · epoch {ex[-1] if ex else 0}/40"); fig.tight_layout()
fig.savefig(sys.argv[1], dpi=105)
