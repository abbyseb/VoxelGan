"""Look-and-correct feasibility part 2: compare X-ray and DRR(warped CT) in scatter-robust ways (scatter is smooth, motion
shows as edges). Same tests as part 1 (signal-to-nuisance, does the residual's best amplitude track the true one), per metric:
  highpass : std(X - G8 * X) residual (Gaussian sigma 8 px removed)
  gradient : residual of standardised Sobel gradients (x, y)
  lncc2d   : 1 - local NCC (9 x 9) between X and DRR (a dissimilarity map, averaged)
  python3 lookcorrect_feasibility2.py"""
import json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
A = Path(__file__).resolve().parent
src = (A / "lookcorrect_feasibility.py").read_text(); src = src[:src.index("out = []")]
exec(src)


def gblur(x, s):
    return rx._gauss2d(x.reshape(-1, 1, *x.shape[-2:]), s).reshape(x.shape)


def feats(P, kind):
    if kind == "highpass":
        return X._std2d(P - gblur(P, 8.0))
    if kind == "gradient":
        k = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=P.device)
        q = P.reshape(-1, 1, *P.shape[-2:])
        gx = F.conv2d(F.pad(q, (1, 1, 1, 1), mode="replicate"), k.view(1, 1, 3, 3)); gy = F.conv2d(F.pad(q, (1, 1, 1, 1), mode="replicate"), k.t().contiguous().view(1, 1, 3, 3))
        return X._std2d(torch.cat([gx, gy], 1).reshape(P.shape[0], -1, *P.shape[-2:]))
    return P


def dissim(Xt, Dw, kind):
    if kind == "lncc2d":
        q = lambda z: z.reshape(-1, 1, *z.shape[-2:])
        a, b = q(Xt), q(Dw); o = torch.ones(1, 1, 9, 9, device=a.device) / 81
        mu = lambda z: F.conv2d(z, o, padding=4)
        ma, mb = mu(a), mu(b)
        cc = (mu(a * b) - ma * mb) / torch.sqrt((mu(a * a) - ma * ma).clamp_min(1e-6) * (mu(b * b) - mb * mb).clamp_min(1e-6))
        return (1 - cc).reshape(Xt.shape)
    return (feats(Xt, kind) - feats(Dw, kind)) ** 2


def energy(ref, u, Xraw, kind, m):
    with torch.no_grad():
        Dw = X.dt.drr(X.warp(ref, u), ANG)
        d = dissim(Xraw, Dw, kind)
        mm_ = m.expand_as(d) if d.shape == m.shape else m.repeat(1, d.shape[1] // m.shape[1], 1, 1)
        return float((d * mm_).sum() / mm_.sum())


KINDS = ["highpass", "gradient", "lncc2d"]
out = []
for c in range(1, 11):
    ref = torch.from_numpy(mm(DIR_IN[c])[None, None]).to(dev); tgt = torch.from_numpy(mm(DIR_TG[c])[None, None]).to(dev)
    u1n = pred(g, DIR_IN[c], DIR_TG[c], 5, 0, "clinic"); u1 = torch.from_numpy(np.ascontiguousarray(np.moveaxis(u1n, -1, 0))[None]).to(dev)
    us = torch.from_numpy(np.ascontiguousarray(np.moveaxis(optimise(c, u1, "full_ct"), -1, 0))[None]).to(dev)
    with torch.no_grad():
        Xp = X.dt.drr(tgt, ANG)
        Xr = realistic_target(DIR_TG[c], int(float(np.abs(DIR_TG[c]).sum()) * 1000) % (2 ** 31))
        m = (X._std2d(X.dt.drr(ref, ANG)) < 0.5).float()
    l0, l5 = lms(c); t = (l5 - l0) * Sp; p = v2.sample_u(u1n, l0) * Sp
    row = {"case": c, "s_true": float((p * t).sum() / (p * p).sum())}
    for kind in KINDS:
        for name, Xt in (("perfect", Xp), ("realistic", Xr)):
            e1, es = energy(ref, u1, Xt, kind, m), energy(ref, us, Xt, kind, m)
            sweep = [energy(ref, s * u1, Xt, kind, m) for s in S]
            row[f"{kind}/{name}"] = {"S/N": (e1 - es) / es, "s_argmin": S[int(np.argmin(sweep))]}
    out.append(row)
    print(f"C{c:02d} true scale {row['s_true']:.2f} | " + " | ".join(f"{k}: S/N {v['S/N']:.3f} s* {v['s_argmin']:.2f}" for k, v in row.items() if "/" in k and "realistic" in k), flush=True)
summ = {}
for kind in KINDS:
    for name in ("perfect", "realistic"):
        k = f"{kind}/{name}"; st_ = np.array([r["s_true"] for r in out]); sa = np.array([r[k]["s_argmin"] for r in out])
        summ[k] = {"mean_S/N": round(float(np.mean([r[k]["S/N"] for r in out])), 3),
                   "corr_argmin_vs_true": round(float(np.corrcoef(sa, st_)[0, 1]) if sa.std() > 0 else float("nan"), 3),
                   "mean_abs_scale_err": round(float(np.abs(sa - st_).mean()), 3)}
(A / "lookcorrect_feasibility2.json").write_text(json.dumps({"summary": summ, "cases": out}, indent=1))
print(json.dumps(summ, indent=1))
