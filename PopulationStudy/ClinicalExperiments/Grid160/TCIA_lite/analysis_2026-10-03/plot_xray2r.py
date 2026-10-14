"""X2R (X2 trained with realistic X-rays) progress: DIR-Lab TRE with clinic-like / perfect / blank / wrong X-ray, against
X2 (perfect X-rays) and X1 with clinic-like X-rays (3.94 mm, realism_clinic_test.json); train vs val; gap.
  python3 plot_xray2r.py OUT.png"""
import json, sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
L = Path(__file__).resolve().parents[1]; A = Path(__file__).resolve().parent
J = lambda p: json.loads(p.read_text()) if p.exists() else None
hr = J(L / "run_A2_full160_aug_xray2r/logs/history.json") or []
h2 = J(L / "run_A2_full160_aug_xray2/logs/history.json")
tr = (J(A / "xray2r_tre.json") or {"xray2r": {}})["xray2r"]; t2 = J(A / "xray2_tre.json")["xray2"]
x1c = np.mean(J(A / "realism_clinic_test.json")["tre300"]["X1"]["clinic"])
x2c = np.mean(J(A / "realism_clinic_test.json")["tre300"]["X2"]["clinic"])
g = lambda h, k: [r[k] for r in h]
fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
a = ax[0]
ep = sorted(t2, key=int); a.plot([int(e) for e in ep], [np.mean(t2[e]["real"]["dir_plain"]) for e in ep], "-", color="grey", label="X2, perfect X-ray")
a.axhline(x1c, color="purple", ls="--", lw=1.2, label=f"X1 with realistic X-ray ({x1c:.2f})")
a.axhline(x2c, color="brown", ls="--", lw=1.2, label=f"X2 with realistic X-ray ({x2c:.2f})")
for mode, c, s, lab in (("clinic", "C0", "o-", "X2R, realistic X-ray"), ("real", "C2", "s-", "X2R, perfect X-ray"),
                        ("zero", "C1", "s--", "X2R, blank"), ("shuffled", "C3", "^--", "X2R, wrong X-ray")):
    ep = sorted([e for e in tr if mode in tr[e]], key=int)
    a.plot([int(e) for e in ep], [np.mean(tr[e][mode]["dir_plain"]) for e in ep], s, ms=4, color=c, label=lab)
a.axhline(3.5, color="k", ls=":", lw=1); a.set_yscale("log"); a.set_ylim(2.5, 14)
a.set_yticks([2.5, 3, 3.5, 4, 5, 6, 8, 10, 14]); a.set_yticklabels(["2.5", "3", "3.5", "4", "5", "6", "8", "10", "14"])
a.set_title("DIR-Lab TRE300 ↓ (log; dotted = 3.5 mm goal)"); a.set_ylabel("mm"); a.legend(fontsize=7.5)
a = ax[1]
a.plot(g(h2, "epoch"), g(h2, "train_eval_lncc_loss"), ":", color="grey", label="X2 train (perfect)")
a.plot(g(h2, "epoch"), g(h2, "val_lncc_loss"), "-", color="grey", label="X2 val (perfect)")
if hr:
    a.plot(g(hr, "epoch"), g(hr, "train_eval_lncc_loss"), "g-s", ms=4, label="X2R train (realistic, fair)")
    a.plot(g(hr, "epoch"), g(hr, "val_lncc_loss"), "b-o", ms=4, label="X2R val (realistic)")
    a.plot(g(hr, "epoch"), g(hr, "val_lncc_loss_perfect_xray"), "c--", label="X2R val (perfect)")
a.set_title("Train vs val LNCC loss ↓"); a.legend(fontsize=8)
a = ax[2]
a.plot(g(h2, "epoch"), np.array(g(h2, "val_lncc_loss")) - np.array(g(h2, "train_eval_lncc_loss")), "-", color="grey", label="X2 gap")
if hr:
    a.plot(g(hr, "epoch"), np.array(g(hr, "val_lncc_loss")) - np.array(g(hr, "train_eval_lncc_loss")), "C2-o", ms=4, label="X2R gap")
a.axhline(0, color="k", lw=.8); a.set_title("Gap: val − train (0 = together)"); a.legend(fontsize=8)
for a in ax: a.set_xlabel("epoch"); a.grid(alpha=.3); a.set_xlim(0.5, 40.5)
fig.suptitle(f"X2R (trained with realistic X-rays) · epoch {hr[-1]['epoch'] if hr else 0}/40"); fig.tight_layout(); fig.savefig(sys.argv[1], dpi=105)
