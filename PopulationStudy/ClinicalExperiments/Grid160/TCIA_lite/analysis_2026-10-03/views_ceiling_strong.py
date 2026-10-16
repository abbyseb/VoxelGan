"""Stronger version of views_ceiling.py: local 2-D NCC (9x9 windows) instead of global NCC, bending 0.1, 300 steps, residual 80^3.
2 vs 8 views, starting from X4. If 8 views still do not beat 2 views, extra angles do not carry usable extra motion information
at this detector resolution (128^2, 4 mm pixels).  python3 views_ceiling_strong.py"""
import json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
A = Path(__file__).resolve().parent
src = (A / "views_ceiling.py").read_text(); src = src[:src.index("res = {")]
exec(src)


def lncc2d(a, b, k=9):
    q = lambda z: z.reshape(-1, 1, *z.shape[-2:]); a, b = q(a), q(b); o = torch.ones(1, 1, k, k, device=a.device) / k ** 2
    mu = lambda z: F.conv2d(z, o, padding=k // 2); ma, mb = mu(a), mu(b)
    cc = (mu(a * b) - ma * mb) / torch.sqrt((mu(a * a) - ma ** 2).clamp_min(1e-6) * (mu(b * b) - mb ** 2).clamp_min(1e-6))
    return cc.mean()


def fit_strong(ref, tgt, u0, angs, steps=300, lam=0.1):
    with torch.no_grad():
        Dt = std(dt.drr(tgt, angs))
    r = torch.zeros(1, 3, 80, 80, 80, device=dev, requires_grad=True); opt = torch.optim.Adam([r], lr=0.03)
    for _ in range(steps):
        u = u0 + F.interpolate(r, size=u0.shape[2:], mode="trilinear", align_corners=True)
        loss = 1 - lncc2d(std(dt.drr(X.warp(ref, u), angs)), Dt) + lam * bending(r)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return np.moveaxis((u0 + F.interpolate(r, size=u0.shape[2:], mode="trilinear", align_corners=True))[0].cpu().numpy(), 0, -1)


res = {"start_X4": [], "2_views_strong": [], "8_views_strong": []}
for c in range(1, 11):
    ref = torch.from_numpy(mm(DIR_IN[c])[None, None]).to(dev); tgt = torch.from_numpy(mm(DIR_TG[c])[None, None]).to(dev)
    with torch.no_grad():
        u = ns["pred"](g, DIR_IN[c], DIR_TG[c], 5, 0, "clinic")
    res["start_X4"].append(tre(u, c)); u0 = torch.from_numpy(np.ascontiguousarray(np.moveaxis(u, -1, 0)))[None].to(dev)
    res["2_views_strong"].append(tre(fit_strong(ref, tgt, u0, VIEWS[2]), c))
    res["8_views_strong"].append(tre(fit_strong(ref, tgt, u0, VIEWS[8]), c))
    print(f"C{c:02d} " + " ".join(f"{k} {vv[-1]:.2f}" for k, vv in res.items()), flush=True)
summ = {k: round(float(np.mean(v)), 3) for k, v in res.items()}
(A / "views_ceiling_strong.json").write_text(json.dumps({"summary": summ, "cases": res}, indent=1)); print(json.dumps(summ, indent=1))
