"""Stage 0 (diagnostic): is the remaining error mostly BREATH SIZE? Rescale X4's predicted field by the breath size measured on the
live X-ray. Rule fixed BEFORE looking at results, no landmark information and no tuning on DIR-Lab:
  s_xray = vertical shift (front view) of the lower-thorax region between DRR(planning CT) and the live (cleaned realistic) X-ray
  s_pred = the same measurement between DRR(planning CT) and DRR(planning CT warped by X4's prediction)
  k = clip(s_xray / s_pred, 0.7, 1.8);   u_scaled = k * u_X4
Shift = argmax over vertical sub-pixel shifts of the NCC of the rows 50-122 (lower lung / diaphragm band), columns 16-112.
Reported: TRE300 before / after per case, k, and (labelled as ORACLE, diagnostic only) the landmark-best scale from before.
  python3 stage0_xray_amplitude.py"""
import json
from pathlib import Path
import numpy as np, torch
A = Path(__file__).resolve().parent
src = (A / "score_x4.py").read_text(); src = src[:src.index("out = json.loads")]
ns = {"__file__": str(A / "score_x4.py")}; exec(src, ns)
X, v2, packs, DIR_IN, DIR_TG, dev, lm = ns["X1"], ns["v2"], ns["packs"], ns["DIR_IN"], ns["DIR_TG"], ns["dev"], ns["landmarks_300"]
g = ns["load"](ns["L"] / "run_A2_full160_aug_x4/DecoderCRB/checkpoints/epoch_040.pt")
dt, std, mm = X.dt, X._std2d, v2.minmax
G = X.cleaner(dev)
ROWS, COLS = slice(50, 122), slice(16, 112)


def vshift(a, b):
    """sub-pixel vertical shift (px, + = b content lower) maximising NCC of region (a, b: 2-D numpy)."""
    best, arg = -9.0, 0.0
    ra = a[ROWS, COLS]; ra = (ra - ra.mean()) / (ra.std() + 1e-8)
    yy = np.arange(a.shape[0], dtype=float)
    for s in np.arange(-20, 20.01, 0.25):
        bs = np.stack([np.interp(yy + s, yy, b[:, j]) for j in range(b.shape[1])], 1)[ROWS, COLS]
        bs = (bs - bs.mean()) / (bs.std() + 1e-8); c = float((ra * bs).mean())
        if c > best: best, arg = c, s
    return arg


def tre(u, c):
    pk = packs[c]; l0 = v2.official_to_iso(lm(c, "T00"), pk); l5 = v2.official_to_iso(lm(c, "T50"), pk)
    return float(np.linalg.norm((l0 + v2.sample_u(u, l0) - l5) * v2.SPACING, axis=1).mean())


oracle = {r["case"]: r["best_scalar"] for r in json.loads((A / "diagnose_x3s.json").read_text())["cases"]}   # X3S, diagnostic only
rows = []
with torch.no_grad():
    for c in range(1, 11):
        x = torch.from_numpy(mm(DIR_IN[c])[None, None]).to(dev)
        u = ns["pred"](g, DIR_IN[c], DIR_TG[c], 5, 0, "clinic")                      # X4 field (Z,Y,X,3), realistic X-rays
        seed = int(float(np.abs(DIR_TG[c]).sum()) * 1000) % (2 ** 31)
        xr = std(G(std(ns["realistic_target"](DIR_TG[c], seed)[:, :1])))[0, 0].cpu().numpy()   # live front X-ray, cleaned
        dp = std(dt.drr(x, [0.0]))[0, 0].cpu().numpy()                                       # planning-CT DRR
        ut = torch.from_numpy(np.ascontiguousarray(np.moveaxis(u, -1, 0)))[None].to(dev)
        dw = std(dt.drr(X.warp(x, ut), [0.0]))[0, 0].cpu().numpy()                          # DRR of the predicted state
        sx, sp = vshift(dp, xr), vshift(dp, dw)
        k = float(np.clip(sx / sp, 0.7, 1.8)) if abs(sp) > 0.25 else 1.0
        rows.append({"case": c, "s_xray_px": sx, "s_pred_px": sp, "k": k, "tre_x4": tre(u, c), "tre_scaled": tre(k * u, c),
                     "oracle_scale_X3S_diagnostic_only": oracle[c]})
        r = rows[-1]
        print(f"C{c:02d} shift X-ray {sx:+.2f}px pred {sp:+.2f}px -> k {k:.2f} (oracle {oracle[c]:.2f}) | TRE {r['tre_x4']:.2f} -> {r['tre_scaled']:.2f}", flush=True)
summ = {"tre_x4": float(np.mean([r["tre_x4"] for r in rows])), "tre_scaled": float(np.mean([r["tre_scaled"] for r in rows])),
        "corr_k_vs_oracle": float(np.corrcoef([r["k"] for r in rows], [r["oracle_scale_X3S_diagnostic_only"] for r in rows])[0, 1])}
(A / "stage0_xray_amplitude.json").write_text(json.dumps({"summary": summ, "cases": rows}, indent=1)); print(json.dumps(summ, indent=1))
