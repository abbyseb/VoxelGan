"""Look-and-correct feasibility (analysis only). Does the 2-D 'still wrong' image R_v = std(X-ray_v) - std(DRR_v(warp(CT, u1)))
carry the motion the first model (X3S) missed, also with REALISTIC X-rays?
For each DIR-Lab case (front + side, X3S latest):
  1. residual energy E(u) = mean_v mean(R_v^2) in the lung shadow, for
       u1           : X3S prediction
       u_star       : near-perfect field (X3S + per-case optimisation with the full target CT; TRE ~1.3 mm)
       zero motion
     with perfect and with realistic X-rays. 'nuisance' = E(u_star) (what is left when the motion is right: scatter/noise etc).
     signal-to-nuisance = (E(u1) - E(u_star)) / E(u_star).
  2. amplitude sweep: E(s * u1) for s in 0.7..1.6. Where is the minimum? The true best scale (from landmarks) is known; if the
     residual's minimum tracks it across cases, the residual 'knows' how much more motion is needed.
  python3 lookcorrect_feasibility.py"""
import json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
A = Path(__file__).resolve().parent
src = (A / "diagnose_x3s.py").read_text(); src = src[:src.index("rows, ceil =")]
sys.argv = [sys.argv[0]]; exec(src)
S = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.45, 1.6]


def residual_energy(ref, u, Xt):
    with torch.no_grad():
        Dw = X._std2d(X.dt.drr(X.warp(ref, u), ANG))
        R = Xt - Dw
        m = (X._std2d(X.dt.drr(ref, ANG)) < 0.5).float()          # rough lung/body shadow mask (darker than average)
        return float(((R ** 2) * m).sum() / m.sum())


out = []
for c in range(1, 11):
    ref = torch.from_numpy(mm(DIR_IN[c])[None, None]).to(dev); tgt = torch.from_numpy(mm(DIR_TG[c])[None, None]).to(dev)
    u1n = pred(g, DIR_IN[c], DIR_TG[c], 5, 0, "clinic"); u1 = torch.from_numpy(np.ascontiguousarray(np.moveaxis(u1n, -1, 0))[None]).to(dev)
    us = torch.from_numpy(np.ascontiguousarray(np.moveaxis(optimise(c, u1, "full_ct"), -1, 0))[None]).to(dev)
    with torch.no_grad():
        Xp = X._std2d(X.dt.drr(tgt, ANG))
        Xr = X._std2d(realistic_target(DIR_TG[c], int(float(np.abs(DIR_TG[c]).sum()) * 1000) % (2 ** 31)))
    l0, l5 = lms(c); t = (l5 - l0) * Sp; p = v2.sample_u(u1n, l0) * Sp
    s_true = float((p * t).sum() / (p * p).sum())
    row = {"case": c, "tre_u1": tre_of(u1n, c), "tre_ustar": tre_of(np.moveaxis(us[0].cpu().numpy(), 0, -1), c), "s_true": s_true}
    for name, Xt in (("perfect", Xp), ("realistic", Xr)):
        e1, es, e0 = residual_energy(ref, u1, Xt), residual_energy(ref, us, Xt), residual_energy(ref, 0 * u1, Xt)
        sweep = [residual_energy(ref, s * u1, Xt) for s in S]
        row[name] = {"E_u1": e1, "E_ustar": es, "E_zero": e0, "signal_to_nuisance": (e1 - es) / es,
                     "s_argmin": S[int(np.argmin(sweep))], "sweep": sweep}
    out.append(row)
    print(f"C{c:02d} TRE u1 {row['tre_u1']:.2f} u* {row['tre_ustar']:.2f} | true best scale {s_true:.2f} | "
          + " | ".join(f"{k}: E u1 {row[k]['E_u1']:.3f} u* {row[k]['E_ustar']:.3f} zero {row[k]['E_zero']:.3f} "
                       f"S/N {row[k]['signal_to_nuisance']:.2f} argmin-s {row[k]['s_argmin']:.2f}" for k in ("perfect", "realistic")), flush=True)
summ = {}
for k in ("perfect", "realistic"):
    st_ = np.array([r["s_true"] for r in out]); sa = np.array([r[k]["s_argmin"] for r in out])
    summ[k] = {"mean_signal_to_nuisance": float(np.mean([r[k]["signal_to_nuisance"] for r in out])),
               "corr_argmin_vs_true_scale": float(np.corrcoef(sa, st_)[0, 1]),
               "mean_abs_scale_error": float(np.abs(sa - st_).mean())}
(A / "lookcorrect_feasibility.json").write_text(json.dumps({"epoch": ep, "summary": summ, "cases": out}, indent=1))
print(json.dumps({"epoch": ep, **summ}, indent=1))
