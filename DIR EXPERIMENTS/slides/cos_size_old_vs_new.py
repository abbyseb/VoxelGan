import sys, numpy as np, torch
sys.path.insert(0, "slides"); sys.path.insert(0, "scripts")
import make_scorer_arrows_fig as F, eval_dir_tcia3_iso2mm_v2 as V2
dev = torch.device("cuda:0"); g = V2.load_generator(dev)
co, cn, ro, rn = [], [], [], []
for c in range(1, 11):
    F.CASE = c; p = V2.pack_case(c); m = p["lung_iso"]
    el = V2.elastix_on_iso(p) * 2; new = V2.predict(g, dev, p["mu"], False) * 2; old = F.old_field_mm(g, dev, p)
    n = lambda u: np.linalg.norm(u[m], axis=1).mean()
    co.append(F.lung_cos(old, el, m)); cn.append(F.lung_cos(new, el, m)); ro.append(n(old)/n(el)); rn.append(n(new)/n(el))
    print(f"C{c:02d} cos old {co[-1]:.2f} new {cn[-1]:.2f} | size old {ro[-1]:.2f} new {rn[-1]:.2f}", flush=True)
print(f"MEAN cos old {np.mean(co):.2f} new {np.mean(cn):.2f} | size old {np.mean(ro):.2f} new {np.mean(rn):.2f}")
