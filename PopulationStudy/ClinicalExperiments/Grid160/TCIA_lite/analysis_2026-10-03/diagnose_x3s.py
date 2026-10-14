"""Where does X3S's ~3.6 mm DIR-Lab TRE come from, and what is the ceiling? (analysis only, nothing here is a method)
A. Landmark error decomposition of X3S (front + side, realistic X-rays): true motion t = l50 - l00 (mm), predicted p = u(l00);
   amplitude gain along t, angular error, per-axis RMS (LR, SI, AP), error vs |t|, TRE after the best per-case scalar
   (oracle breath-depth) and after the best per-case scalar per axis.
B. Ceilings by per-case optimisation of a smooth residual on top of the X3S field (Adam, coarse 40^3 grid, bending penalty):
   B1 'full target CT'   : 3-D lung LNCC with the real T00 CT (what this field model can reach if it saw the target)
   B2 'two perfect X-rays': 2-D NCC between DRRs of the warped CT and perfect front/side DRRs of T00 (information in 2 views)
  python3 diagnose_x3s.py [checkpoint]"""
import json, sys, importlib.util
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
A = Path(__file__).resolve().parent
src = (A / "score_xray3s.py").read_text(); src = src[:src.index("out = json.loads")]
__file__ = str(A / "score_xray3s.py"); exec(src)
X = X1
ck = Path(sys.argv[1]) if len(sys.argv) > 1 else L / "run_A2_full160_aug_xray3s/DecoderCRB/checkpoints/latest.pt"
st = torch.load(str(ck), map_location="cpu", weights_only=False); ep = st.get("epoch_done")
g = X.XrayUNet2(im_size=160, n_phases=10); g.load_state_dict(st["generator"]); g = g.to(dev).eval()
for p_ in g.parameters(): p_.requires_grad_(False)


def lms(c):
    pk = packs[c]; l0 = v2.official_to_iso(landmarks_300(c, "T00"), pk); l5 = v2.official_to_iso(landmarks_300(c, "T50"), pk)
    return l0, l5


def tre_of(u_zyx3, c):
    l0, l5 = lms(c); return float(np.linalg.norm((l0 + v2.sample_u(u_zyx3, l0) - l5) * Sp, axis=1).mean())


def bending(r):
    d2 = lambda x, ax: x.narrow(ax, 2, x.shape[ax] - 2) - 2 * x.narrow(ax, 1, x.shape[ax] - 2) + x.narrow(ax, 0, x.shape[ax] - 2)
    return sum(d2(r, a).pow(2).mean() for a in (2, 3, 4))


def optimise(c, u0, mode, steps=150, lam=0.5):
    """u0 (1,3,D,H,W) voxels. Residual on 40^3, upsampled. Returns optimised (Z,Y,X,3) field."""
    ref = torch.from_numpy(mm(DIR_IN[c])[None, None]).to(dev); tgt = torch.from_numpy(mm(DIR_TG[c])[None, None]).to(dev)
    lung = torch.from_numpy(packs[c]["lung_iso"].astype(np.float32))[None, None].to(dev)
    if mode == "xray2":
        with torch.no_grad():
            Dt = X._std2d(X.dt.drr(tgt, ANG))
    r = torch.zeros(1, 3, 40, 40, 40, device=dev, requires_grad=True); opt = torch.optim.Adam([r], lr=0.05)
    for _ in range(steps):
        u = u0 + F.interpolate(r, size=u0.shape[2:], mode="trilinear", align_corners=True)
        if mode == "full_ct":
            loss = X.lncc_term(ref, tgt, u, lung)
        else:
            w = X.warp(ref, u); Dw = X._std2d(X.dt.drr(w, ANG))
            loss = 1 - (Dw * Dt).mean((2, 3)).mean()                          # 2-D NCC of standardised images, both views
        loss = loss + lam * bending(r)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        u = u0 + F.interpolate(r, size=u0.shape[2:], mode="trilinear", align_corners=True)
    return np.moveaxis(u[0].cpu().numpy(), 0, -1)


rows, ceil = [], {"x3s": [], "full_ct": [], "xray2_perfect": [], "zero_motion": []}
for c in range(1, 11):
    u = pred(g, DIR_IN[c], DIR_TG[c], 5, 0, "clinic")                     # (Z,Y,X,3) voxels, channels (LR, SI, AP)
    l0, l5 = lms(c); t = (l5 - l0) * Sp; p = v2.sample_u(u, l0) * Sp      # landmark arrays are (x,y,z) = (LR,SI,AP)
    e = p - t; tn = np.linalg.norm(t, axis=1)
    gain = float((p * t).sum() / (t * t).sum())
    cosang = (p * t).sum(1) / (np.linalg.norm(p, axis=1) * tn + 1e-9)
    s = float((p * t).sum() / (p * p).sum()); s_ax = (p * t).sum(0) / ((p * p).sum(0) + 1e-9)
    big = tn > np.median(tn)
    rows.append({"case": c, "tre": float(np.linalg.norm(e, axis=1).mean()), "mean_true_motion": float(tn.mean()),
                 "mean_pred_motion": float(np.linalg.norm(p, axis=1).mean()), "amplitude_gain": gain,
                 "median_angle_deg": float(np.degrees(np.arccos(np.clip(np.median(cosang), -1, 1)))),
                 "rms_err_LR_SI_AP": np.sqrt((e ** 2).mean(0)).round(3).tolist(),
                 "tre_small_motion_half": float(np.linalg.norm(e[~big], axis=1).mean()),
                 "tre_large_motion_half": float(np.linalg.norm(e[big], axis=1).mean()),
                 "tre_best_scalar": float(np.linalg.norm(s * p - t, axis=1).mean()), "best_scalar": s,
                 "tre_best_axis_scalars": float(np.linalg.norm(s_ax * p - t, axis=1).mean())})
    u0 = torch.from_numpy(np.ascontiguousarray(np.moveaxis(u, -1, 0))[None]).to(dev)
    ceil["x3s"].append(rows[-1]["tre"]); ceil["zero_motion"].append(float(np.linalg.norm(t, axis=1).mean()))
    ceil["full_ct"].append(tre_of(optimise(c, u0, "full_ct"), c))
    ceil["xray2_perfect"].append(tre_of(optimise(c, u0, "xray2"), c))
    r_ = rows[-1]
    print(f"C{c:02d} TRE {r_['tre']:.2f} | true {r_['mean_true_motion']:.1f} pred {r_['mean_pred_motion']:.1f} gain {r_['amplitude_gain']:.2f} "
          f"angle {r_['median_angle_deg']:.0f}° | RMS LR/SI/AP {r_['rms_err_LR_SI_AP']} | best-scalar {r_['tre_best_scalar']:.2f} "
          f"| ceil full-CT {ceil['full_ct'][-1]:.2f} 2-perfect-X-rays {ceil['xray2_perfect'][-1]:.2f}", flush=True)
summ = {k: float(np.mean(v)) for k, v in ceil.items()}
summ.update({k: float(np.mean([r[k] for r in rows])) for k in ("amplitude_gain", "median_angle_deg", "tre_small_motion_half",
                                                                 "tre_large_motion_half", "tre_best_scalar", "tre_best_axis_scalars")})
summ["rms_err_LR_SI_AP"] = np.mean([r["rms_err_LR_SI_AP"] for r in rows], 0).round(3).tolist()
(A / "diagnose_x3s.json").write_text(json.dumps({"epoch": ep, "summary": summ, "cases": rows, "ceilings": ceil}, indent=1))
print(json.dumps({"epoch": ep, **summ}, indent=1))
