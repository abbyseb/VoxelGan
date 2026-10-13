"""X1 / X2 DIR-Lab TRE300 with the 'clinic-like' randomised X-ray (realistic_xray.clinic_projection): mild random scatter,
light noise, slight blur/shift/gamma, sometimes couch rails and cables. 3 random X-rays per case (seeds 0-2), same draws
for X1 and X2. Also 'clean' for reference. No retraining.  python3 realism_clinic_test.py [X2 checkpoint]"""
import json, sys
import numpy as np, torch
src = open(__file__.replace("realism_clinic_test.py", "realism_test.py")).read(); src = src[:src.index("F = lambda d")]
exec(src)
F = lambda d: np.moveaxis(d[0].cpu().numpy(), 0, -1).astype(np.float32)
res = {m: {"clean": [], "clinic": []} for m in ("X1", "X2")}
with torch.no_grad():
    for c in range(1, 11):
        pk = v2.pack_case(c); ref = T(mm(pk["mu"])); hu_t = T(t00_hu(pk))
        pr = X2._std2d(dt.drr(ref, [0.0])); rp, tp = torch.tensor([5], device=dev), torch.tensor([0], device=dev)
        soft, bone = rx.materials(hu_t); ls, lb = dt.drr(soft, [0.0]) / 10.0, dt.drr(bone, [0.0]) / 10.0
        draws = [("clean", dt.drr(T(mm(v2.hu_to_mu(hu_t[0, 0].cpu().numpy()))), [0.0]))]
        draws += [("clinic", rx.clinic_projection(ls, lb, torch.Generator().manual_seed(100 * c + s))) for s in range(3)]
        for name, p in draws:
            pt = X2._std2d(p)
            res["X1"][name].append(tre(F(g1(ref, rp, tp, dt.backproject(pt, [0.0], ref.shape[2:]))), c, pk))
            d = pt - pr
            vol = torch.cat([dt.backproject(pt, [0.0], ref.shape[2:]), dt.backproject(d, [0.0], ref.shape[2:])], 1)
            res["X2"][name].append(tre(F(g2(ref, rp, tp, vol, d)), c, pk))
        print(f"C{c:02d} clinic X1 {np.mean(res['X1']['clinic'][-3:]):.2f} X2 {np.mean(res['X2']['clinic'][-3:]):.2f}", flush=True)
A = Path(__file__).resolve().parent
(A / "realism_clinic_test.json").write_text(json.dumps({"x2_epoch": s2.get("epoch_done"), "tre300": res}, indent=1))
print(f"\nDIR-Lab TRE300 (mm), 10 cases · X2 epoch {s2.get('epoch_done')}")
for name in ("clean", "clinic"):
    print(f"{name:8s} X1 {np.mean(res['X1'][name]):.2f}  X2 {np.mean(res['X2'][name]):.2f}")
