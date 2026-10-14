"""X3S (front + side, separate channels, realistic) progress vs X2R: DIR-Lab TRE per input scenario, train vs val, gap.
  python3 plot_xray3s.py OUT.png"""
import json, sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
L = Path(__file__).resolve().parents[1]; A = Path(__file__).resolve().parent
J = lambda p: json.loads(p.read_text()) if p.exists() else None
hs = J(L / "run_A2_full160_aug_xray3s/logs/history.json") or []
hr = J(L / "run_A2_full160_aug_xray2r/logs/history.json") or []
ts = (J(A / "xray3s_tre.json") or {"xray3s": {}})["xray3s"]; tr = (J(A / "xray2r_tre.json") or {"xray2r": {}})["xray2r"]
x1c = np.mean(J(A / "realism_clinic_test.json")["tre300"]["X1"]["clinic"])
g = lambda h, k: [r[k] for r in h]
fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
a = ax[0]
ep = sorted([e for e in tr if "clinic" in tr[e]], key=int)
a.plot([int(e) for e in ep], [np.mean(tr[e]["clinic"]["dir_plain"]) for e in ep], "-", color="grey", lw=2, label="X2R (front only), realistic")
a.axhline(x1c, color="purple", ls="--", lw=1.2, label=f"X1, realistic ({x1c:.2f})")
for mode, c, s, lab in (("clinic", "C0", "o-", "X3S front+side, realistic"), ("real", "C2", "s-", "X3S front+side, perfect"),
                        ("front_only", "C9", "d:", "X3S front only (side blanked)"), ("side_only", "C4", "v:", "X3S side only (front blanked)"),
                        ("zero", "C1", "s--", "X3S no X-ray"), ("shuffled", "C3", "^--", "X3S wrong X-ray")):
    ep = sorted([e for e in ts if mode in ts[e]], key=int)
    a.plot([int(e) for e in ep], [np.mean(ts[e][mode]["dir_plain"]) for e in ep], s, ms=4, color=c, label=lab)
a.axhline(3.5, color="k", ls=":", lw=1); a.set_yscale("log"); a.set_ylim(2.5, 10)
a.set_yticks([2.5, 3, 3.5, 4, 5, 6, 8, 10]); a.set_yticklabels(["2.5", "3", "3.5", "4", "5", "6", "8", "10"])
a.set_title("DIR-Lab TRE300 ↓ (log; dotted = 3.5 mm)"); a.set_ylabel("mm"); a.legend(fontsize=7.5)
a = ax[1]
if hr:
    a.plot(g(hr, "epoch"), g(hr, "train_eval_lncc_loss"), ":", color="grey", label="X2R train")
    a.plot(g(hr, "epoch"), g(hr, "val_lncc_loss"), "-", color="grey", label="X2R val")
if hs:
    a.plot(g(hs, "epoch"), g(hs, "train_eval_lncc_loss"), "g-s", ms=4, label="X3S train (realistic, fair)")
    a.plot(g(hs, "epoch"), g(hs, "val_lncc_loss"), "b-o", ms=4, label="X3S val (realistic)")
a.set_title("Train vs val LNCC loss ↓"); a.legend(fontsize=8)
a = ax[2]
if hr: a.plot(g(hr, "epoch"), np.array(g(hr, "val_lncc_loss")) - np.array(g(hr, "train_eval_lncc_loss")), "-", color="grey", label="X2R gap")
if hs: a.plot(g(hs, "epoch"), np.array(g(hs, "val_lncc_loss")) - np.array(g(hs, "train_eval_lncc_loss")), "C2-o", ms=4, label="X3S gap")
a.axhline(0, color="k", lw=.8); a.set_title("Gap: val − train (0 = together)"); a.legend(fontsize=8)
for a in ax: a.set_xlabel("epoch"); a.grid(alpha=.3); a.set_xlim(0.5, 40.5)
fig.suptitle(f"X3S (front + side X-rays, separate channels) · epoch {hs[-1]['epoch'] if hs else 0}/40"); fig.tight_layout(); fig.savefig(sys.argv[1], dpi=105)
