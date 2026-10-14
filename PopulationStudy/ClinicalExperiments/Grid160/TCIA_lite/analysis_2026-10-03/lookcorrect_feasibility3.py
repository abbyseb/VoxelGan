"""Part 3: which realism component hides the look-and-correct signal? lncc2d and gradient metrics, X-ray variants:
physics (polychromatic + bone only), +noise, +scatter (SPR 0.3), +blur, rails_cables (physics + rails/cables only), clinic (all)."""
import json
from pathlib import Path
import numpy as np, torch
A = Path(__file__).resolve().parent
src = (A / "lookcorrect_feasibility2.py").read_text(); src = src[:src.index("KINDS = [")]
exec(src)
rx.LEVELS.update({"physics": dict(spr=0.0, n0=None, sigma_e=0.0, blur=0.0, shift=0.0),
                  "noise": dict(spr=0.0, n0=1e5, sigma_e=10.0, blur=0.0, shift=0.0),
                  "scatter": dict(spr=0.3, n0=None, sigma_e=0.0, blur=0.0, shift=0.0),
                  "blur": dict(spr=0.0, n0=None, sigma_e=0.0, blur=0.6, shift=0.0)})


def variant(tgt_mu, v, seed):
    hu = torch.from_numpy(((tgt_mu / v2.MU_WATER) * 1000.0 - 1000.0)[None, None].astype(np.float32)).to(dev)
    soft, bone = rx.materials(hu); outs = []
    for a in ANG:
        ls, lb = X.dt.drr(soft, [a]) / 10.0, X.dt.drr(bone, [a]) / 10.0; gen = torch.Generator().manual_seed(seed + int(a))
        if v == "rails_cables":
            c0 = dict(rx.CLINIC); rx.CLINIC.update(spr=(0.0, 0.0), n0_log10=(9.0, 9.0), sigma_e=(0.0, 0.0), blur=(0.0, 0.0), shift=(0, 0), gamma=(1.0, 1.0), p_rails=1.0, p_cables=1.0)
            outs.append(rx.clinic_projection(ls, lb, gen)); rx.CLINIC.clear(); rx.CLINIC.update(c0)
        elif v == "clinic":
            outs.append(rx.clinic_projection(ls, lb, gen))
        else:
            outs.append(rx.from_line_integrals(ls, lb, v, gen))
    return torch.cat(outs, 1)


VARS = ["physics", "noise", "scatter", "blur", "rails_cables", "clinic"]; KINDS = ["lncc2d", "gradient"]
res = {f"{k}/{v}": {"sn": [], "sa": []} for k in KINDS for v in VARS}; strue = []
for c in range(1, 11):
    ref = torch.from_numpy(mm(DIR_IN[c])[None, None]).to(dev)
    u1n = pred(g, DIR_IN[c], DIR_TG[c], 5, 0, "clinic"); u1 = torch.from_numpy(np.ascontiguousarray(np.moveaxis(u1n, -1, 0))[None]).to(dev)
    us = torch.from_numpy(np.ascontiguousarray(np.moveaxis(optimise(c, u1, "full_ct"), -1, 0))[None]).to(dev)
    m = (X._std2d(X.dt.drr(ref, ANG)) < 0.5).float()
    l0, l5 = lms(c); t = (l5 - l0) * Sp; p = v2.sample_u(u1n, l0) * Sp; strue.append(float((p * t).sum() / (p * p).sum()))
    for v in VARS:
        with torch.no_grad(): Xt = variant(DIR_TG[c], v, 100 * c)
        for k in KINDS:
            e1, es = energy(ref, u1, Xt, k, m), energy(ref, us, Xt, k, m)
            res[f"{k}/{v}"]["sn"].append((e1 - es) / es); res[f"{k}/{v}"]["sa"].append(S[int(np.argmin([energy(ref, s * u1, Xt, k, m) for s in S]))])
    print(f"C{c:02d} done", flush=True)
st_ = np.array(strue)
summ = {k: {"mean_S/N": round(float(np.mean(v["sn"])), 3), "corr_argmin_vs_true": round(float(np.corrcoef(v["sa"], st_)[0, 1]) if np.std(v["sa"]) > 0 else float("nan"), 3)} for k, v in res.items()}
(A / "lookcorrect_feasibility3.json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ, indent=1))
