"""Per-patient fine-tuning at treatment (user-approved test, 17 Oct 2026): start from X4 (realistic front+side input), then fit a
smooth residual (80^3, bending 0.1, Adam 300 steps, local 2-D NCC 9x9 -- as views_ceiling_strong.py) so DRRs of the warped
planning CT match the LIVE X-rays. Live X-rays here are REALISTIC (clinic-like, seed per volume and view) and passed through
the X-ray cleaner; renders of the planning CT are clean software DRRs. Uses only the live X-rays + planning CT (no target CT,
no landmarks, no training on the patient). V = 2 (0, 90) and 8 (every 22.5 deg).  python3 fit_realistic.py"""
import json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
A = Path(__file__).resolve().parent
src = (A / "views_ceiling_strong.py").read_text(); src = src[:src.index("res = {\"start_X4\": [], \"2_views_strong\"")]
exec(src)
G = X.cleaner(dev); rx = X.rx


def live(tgt_mu, angs):
    hu = torch.from_numpy(((tgt_mu / v2.MU_WATER) * 1000 - 1000)[None, None].astype(np.float32)).to(dev)
    soft, bone = rx.materials(hu); seed = int(float(np.abs(tgt_mu).sum()) * 1000) % (2 ** 31)
    out = [G(std(rx.clinic_projection(dt.drr(soft, [a]) / 10.0, dt.drr(bone, [a]) / 10.0, torch.Generator().manual_seed(seed + int(a * 10))))) for a in angs]
    return std(torch.cat(out, 1))


def fit_live(ref, Dt, u0, angs, steps=300, lam=0.1):
    r = torch.zeros(1, 3, 80, 80, 80, device=dev, requires_grad=True); opt = torch.optim.Adam([r], lr=0.03)
    for _ in range(steps):
        u = u0 + F.interpolate(r, size=u0.shape[2:], mode="trilinear", align_corners=True)
        loss = 1 - lncc2d(std(dt.drr(X.warp(ref, u), angs)), Dt) + lam * bending(r)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return np.moveaxis((u0 + F.interpolate(r, size=u0.shape[2:], mode="trilinear", align_corners=True))[0].cpu().numpy(), 0, -1)


res = {"start_X4": [], "fit_2_realistic": [], "fit_8_realistic": []}
for c in range(1, 11):
    ref = torch.from_numpy(mm(DIR_IN[c])[None, None]).to(dev)
    with torch.no_grad():
        u = ns["pred"](g, DIR_IN[c], DIR_TG[c], 5, 0, "clinic")
    res["start_X4"].append(tre(u, c)); u0 = torch.from_numpy(np.ascontiguousarray(np.moveaxis(u, -1, 0)))[None].to(dev)
    for v in (2, 8):
        with torch.no_grad(): Dt = live(DIR_TG[c], VIEWS[v])
        res[f"fit_{v}_realistic"].append(tre(fit_live(ref, Dt, u0, VIEWS[v]), c))
    print(f"C{c:02d} " + " ".join(f"{k} {vv[-1]:.2f}" for k, vv in res.items()), flush=True)
summ = {k: round(float(np.mean(v)), 3) for k, v in res.items()}
(A / "fit_realistic.json").write_text(json.dumps({"summary": summ, "cases": res}, indent=1)); print(json.dumps(summ, indent=1))
