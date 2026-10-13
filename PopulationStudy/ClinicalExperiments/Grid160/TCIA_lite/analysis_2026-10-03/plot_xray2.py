"""X2 vs X1 vs LNCC-only: DIR-Lab TRE (X2 real/blank/wrong X-ray), train vs val, gap.  python3 plot_xray2.py OUT.png"""
import json, sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
L = Path(__file__).resolve().parents[1]; A = Path(__file__).resolve().parent
J = lambda p: json.loads(p.read_text()) if p.exists() else None
h2 = J(L / "run_A2_full160_aug_xray2/logs/history.json") or []
h1 = J(L / "run_A2_full160_aug_xray1/logs/history.json")
t2 = (J(A / "xray2_tre.json") or {"xray2": {}})["xray2"]; t1 = J(A / "xray1_tre.json")["xray1"]; tu = J(A / "unsup_lncc_tre.json")["unsup_lncc"]
g = lambda h, k: [r[k] for r in h]
fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
a = ax[0]
for t, c, l in ((tu, "lightgrey", "LNCC-only"),):
    ep = sorted(t, key=int); a.plot([int(e) for e in ep], [np.mean(t[e]["dir_plain"]) for e in ep], "-", color=c, label=l)
ep = sorted(t1, key=int); a.plot([int(e) for e in ep], [np.mean(t1[e]["real"]["dir_plain"]) for e in ep], "-", color="grey", label="X1 real X-ray")
for mode, c, s in (("real", "C0", "o-"), ("zero", "C1", "s--"), ("shuffled", "C3", "^--")):
    ep = sorted([e for e in t2 if mode in t2[e]], key=int)
    a.plot([int(e) for e in ep], [np.mean(t2[e][mode]["dir_plain"]) for e in ep], s, ms=4, color=c, label=f"X2 {mode} X-ray")
a.axhline(3.5, color="k", ls=":", lw=1); a.set_yscale("log"); a.set_ylim(2.5, 14)
a.set_yticks([2.5, 3, 3.5, 4, 5, 6, 8, 10, 14]); a.set_yticklabels(["2.5", "3", "3.5", "4", "5", "6", "8", "10", "14"])
a.set_title("DIR-Lab TRE300 ↓ (log scale; dotted = 3.5 mm goal)"); a.set_ylabel("mm"); a.legend(fontsize=8)
a = ax[1]
a.plot(g(h1, "epoch"), g(h1, "train_eval_lncc_loss"), ":", color="grey", label="X1 train")
a.plot(g(h1, "epoch"), g(h1, "val_lncc_loss"), "-", color="grey", label="X1 val")
if h2:
    a.plot(g(h2, "epoch"), g(h2, "train_eval_lncc_loss"), "g-s", ms=4, label="X2 train (fair)")
    a.plot(g(h2, "epoch"), g(h2, "val_lncc_loss"), "b-o", ms=4, label="X2 val")
a.set_title("Train vs val LNCC loss ↓"); a.legend(fontsize=8)
a = ax[2]
a.plot(g(h1, "epoch"), np.array(g(h1, "val_lncc_loss")) - np.array(g(h1, "train_eval_lncc_loss")), "-", color="grey", label="X1 gap")
if h2:
    a.plot(g(h2, "epoch"), np.array(g(h2, "val_lncc_loss")) - np.array(g(h2, "train_eval_lncc_loss")), "C2-o", ms=4, label="X2 gap")
a.axhline(0, color="k", lw=.8); a.set_title("Gap: val − train (0 = curves together)"); a.legend(fontsize=8)
for a in ax: a.set_xlabel("epoch"); a.grid(alpha=.3); a.set_xlim(0.5, 40.5)
fig.suptitle(f"X2 (X-ray + 'what moved' + breath code) · epoch {h2[-1]['epoch'] if h2 else 0}/40"); fig.tight_layout(); fig.savefig(sys.argv[1], dpi=105)
