"""How much 3-D motion information do MORE X-ray views carry? (analysis only, ceiling test)
Start from X4's prediction (realistic front+side input), then optimise a smooth residual (40^3, bending 0.5, Adam 150 steps,
as diagnose_x3s.optimise) so that DRRs of the warped planning CT match PERFECT DRRs of the target (T00) at V views spread
over 180 deg: V = 2 (0, 90), 4 (0, 45, 90, 135), 8 (every 22.5 deg). Global NCC of standardised images, mean over views.
If more views drive TRE well below the 2-view result, extra angles carry the missing (depth / internal) motion.
  python3 views_ceiling.py"""
import json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
A = Path(__file__).resolve().parent
src = (A / "score_x4.py").read_text(); src = src[:src.index("out = json.loads")]
ns = {"__file__": str(A / "score_x4.py")}; exec(src, ns)
X, v2, packs, DIR_IN, DIR_TG, dev, lm = ns["X1"], ns["v2"], ns["packs"], ns["DIR_IN"], ns["DIR_TG"], ns["dev"], ns["landmarks_300"]
g = ns["load"](ns["L"] / "run_A2_full160_aug_x4/DecoderCRB/checkpoints/epoch_040.pt")
mm, std, dt = v2.minmax, X._std2d, X.dt
VIEWS = {2: [0.0, 90.0], 4: [0.0, 45.0, 90.0, 135.0], 8: [22.5 * i for i in range(8)]}


def bending(r):
    d2 = lambda x, ax: x.narrow(ax, 2, x.shape[ax] - 2) - 2 * x.narrow(ax, 1, x.shape[ax] - 2) + x.narrow(ax, 0, x.shape[ax] - 2)
    return sum(d2(r, a).pow(2).mean() for a in (2, 3, 4))


def fit(ref, tgt, u0, angs, steps=150, lam=0.5):
    with torch.no_grad():
        Dt = std(dt.drr(tgt, angs))
    r = torch.zeros(1, 3, 40, 40, 40, device=dev, requires_grad=True); opt = torch.optim.Adam([r], lr=0.05)
    for _ in range(steps):
        u = u0 + F.interpolate(r, size=u0.shape[2:], mode="trilinear", align_corners=True)
        Dw = std(dt.drr(X.warp(ref, u), angs))
        loss = 1 - (Dw * Dt).mean((2, 3)).mean() + lam * bending(r)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return np.moveaxis((u0 + F.interpolate(r, size=u0.shape[2:], mode="trilinear", align_corners=True))[0].cpu().numpy(), 0, -1)


def tre(u, c):
    pk = packs[c]; l0 = v2.official_to_iso(lm(c, "T00"), pk); l5 = v2.official_to_iso(lm(c, "T50"), pk)
    return float(np.linalg.norm((l0 + v2.sample_u(u, l0) - l5) * v2.SPACING, axis=1).mean())


res = {"start_X4": []} | {f"{v}_views": [] for v in VIEWS}
for c in range(1, 11):
    ref = torch.from_numpy(mm(DIR_IN[c])[None, None]).to(dev); tgt = torch.from_numpy(mm(DIR_TG[c])[None, None]).to(dev)
    with torch.no_grad():
        u = ns["pred"](g, DIR_IN[c], DIR_TG[c], 5, 0, "clinic")
    res["start_X4"].append(tre(u, c))
    u0 = torch.from_numpy(np.ascontiguousarray(np.moveaxis(u, -1, 0)))[None].to(dev)
    for v, angs in VIEWS.items():
        res[f"{v}_views"].append(tre(fit(ref, tgt, u0, angs), c))
    print(f"C{c:02d} " + " ".join(f"{k} {vv[-1]:.2f}" for k, vv in res.items()), flush=True)
summ = {k: round(float(np.mean(v)), 3) for k, v in res.items()}
(A / "views_ceiling.json").write_text(json.dumps({"summary": summ, "cases": res}, indent=1)); print(json.dumps(summ, indent=1))
