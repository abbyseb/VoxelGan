"""X2-BSpline (l20, epoch 40) vs dense X2 (epoch 40) on DIR-Lab with perfect / realistic (clinic) / realistic+cleaner front
X-rays; same realistic X-rays (seed per volume) for both models. TRE300 plain + mirror, folding.  python3 realistic_xray2_bspline.py"""
import json, os, sys, importlib.util
from pathlib import Path
import numpy as np, torch
A = Path(__file__).resolve().parent; L = A.parent
os.environ["BSP_LEVEL"] = "20"
src = (A / "score_xray2_bspline.py").read_text(); src = src[:src.index("out = json.loads")]
ns = {"__file__": str(A / "score_xray2_bspline.py")}; exec(src, ns)
XB, v2, packs, DIR_IN, DIR_TG, dev = ns["X1"], ns["v2"], ns["packs"], ns["DIR_IN"], ns["DIR_TG"], ns["dev"]
gb = ns["load"](L / "run_A2_full160_aug_xray2_bspline/DecoderCRB_l20/checkpoints/epoch_040.pt")
s2 = importlib.util.spec_from_file_location("x2", L / "run_A2_full160_aug_xray2/scripts/train_xray2.py"); X2 = importlib.util.module_from_spec(s2); s2.loader.exec_module(X2)
gd = X2.XrayUNet2(im_size=160, n_phases=10); gd.load_state_dict(torch.load(str(L / "run_A2_full160_aug_xray2/DecoderCRB/checkpoints/epoch_040.pt"), map_location="cpu", weights_only=False)["generator"]); gd = gd.to(dev).eval()
sys.path.insert(0, str(L / "xray")); sys.path.insert(0, str(L / "xray_cleaner"))
import realistic_xray as rx, train_cleaner as TC
G = TC.UNet2D().to(dev); G.load_state_dict(torch.load(str(L / "xray_cleaner/run/ckpt/best_G.pt"), map_location="cpu", weights_only=False)["G"]); G.eval()
dt, std, mm = X2.dt, X2._std2d, v2.minmax
from dirlab_tre import landmarks_300


def front(tgt_mu, kind):
    t = torch.from_numpy(mm(tgt_mu)[None, None]).to(dev)
    if kind == "perfect":
        return std(dt.drr(t, [0.0]))
    hu = torch.from_numpy(((tgt_mu / v2.MU_WATER) * 1000 - 1000)[None, None].astype(np.float32)).to(dev)
    soft, bone = rx.materials(hu); seed = int(float(np.abs(tgt_mu).sum()) * 1000) % (2 ** 31)
    p = std(rx.clinic_projection(dt.drr(soft, [0.0]) / 10.0, dt.drr(bone, [0.0]) / 10.0, torch.Generator().manual_seed(seed)))
    return std(G(p)) if kind == "cleaned" else p


def field(g, mu, tgt_mu, kind):
    x = torch.from_numpy(mm(mu)[None, None]).to(dev)
    pt = front(tgt_mu, kind); pr = std(dt.drr(x, [0.0])); d = pt - pr
    vol = torch.cat([dt.backproject(pt, [0.0], x.shape[2:]), dt.backproject(d, [0.0], x.shape[2:])], 1)
    u = g(x, torch.tensor([5], device=dev), torch.tensor([0], device=dev), vol, d)
    return np.moveaxis(u[0].cpu().numpy(), 0, -1).astype(np.float32)


def both(g, mu, tgt_mu, kind):
    a = field(g, mu, tgt_mu, kind); f = lambda v: np.ascontiguousarray(v[:, :, ::-1])
    b = field(g, f(mu), f(tgt_mu), kind); b = np.ascontiguousarray(b[:, :, ::-1, :]); b[..., 0] *= -1
    return a, 0.5 * (a + b)


res = {}
with torch.no_grad():
    for c in range(1, 11):
        pk = packs[c]; l0 = v2.official_to_iso(landmarks_300(c, "T00"), pk); l5 = v2.official_to_iso(landmarks_300(c, "T50"), pk)
        for mname, g in (("X2_bspline", gb), ("X2_dense", gd)):
            for kind in ("perfect", "realistic", "cleaned"):
                a, m = both(g, DIR_IN[c], DIR_TG[c], kind)
                for i, var in ((a, "plain"), (m, "mirror")):
                    res.setdefault(f"{mname}/{kind}/{var}", []).append(float(np.linalg.norm((l0 + v2.sample_u(i, l0) - l5) * v2.SPACING, axis=1).mean()))
        print(f"C{c:02d} " + " ".join(f"{k.replace('/plain','')} {v[-1]:.2f}" for k, v in res.items() if k.endswith("plain")), flush=True)
summ = {k: round(float(np.mean(v)), 3) for k, v in res.items()}
(A / "realistic_xray2_bspline.json").write_text(json.dumps({"summary": summ, "cases": res}, indent=1)); print(json.dumps(summ, indent=1))
