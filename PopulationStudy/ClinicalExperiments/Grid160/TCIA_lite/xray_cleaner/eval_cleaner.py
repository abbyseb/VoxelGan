"""After training: does the X-ray Cleaner rescue the motion models on realistic X-rays? DIR-Lab TRE300 plain, 10 cases,
the SAME realistic X-rays (clinic setting, fixed seed per volume and view, as score_xray3s.py) with and without cleaning:
  X2  (front; trained on perfect DRRs: 2.93 perfect / ~9.3 realistic)    X3S (front + side; trained on realistic mix)
Also the look-and-correct residual test (lncc2d, as lookcorrect_feasibility2) on cleaned X-rays, and a real SPARE clinical
projection before/after cleaning (figure).
  python3 eval_cleaner.py [run/ckpt/best_G.pt]"""
import json, sys, importlib.util
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; L = HERE.parent; A = L / "analysis_2026-10-03"
sys.path.insert(0, str(HERE))
import train_cleaner as TC
ck = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "run" / "ckpt" / "best_G.pt"
src = (A / "score_xray3s.py").read_text(); src = src[:src.index("out = json.loads")]
__file__ = str(A / "score_xray3s.py"); exec(src)                       # X1 = X3S module, packs, DIR_IN/TG, realistic_target, pred
X3 = X1
spec = importlib.util.spec_from_file_location("x2", L / "run_A2_full160_aug_xray2" / "scripts" / "train_xray2.py")
X2m = importlib.util.module_from_spec(spec); spec.loader.exec_module(X2m)
G = TC.UNet2D().to(dev); G.load_state_dict(torch.load(str(ck), map_location="cpu", weights_only=False)["G"]); G.eval()
g2 = X2m.XrayUNet2(im_size=160, n_phases=10)
g2.load_state_dict(torch.load(str(L / "run_A2_full160_aug_xray2/DecoderCRB/checkpoints/epoch_040.pt"), map_location="cpu", weights_only=False)["generator"]); g2 = g2.to(dev).eval()
c3 = sorted((L / "run_A2_full160_aug_xray3s/DecoderCRB/checkpoints").glob("epoch_0*.pt"))[-1]
g3 = load(c3)
dt = X3.dt; std = X3._std2d
clean = lambda p: torch.cat([G(std(p[:, i:i + 1])) for i in range(p.shape[1])], 1)


def tre(u, c):
    pk = packs[c]; l0 = v2.official_to_iso(landmarks_300(c, "T00"), pk); l5 = v2.official_to_iso(landmarks_300(c, "T50"), pk)
    return float(np.linalg.norm((l0 + v2.sample_u(u, l0) - l5) * Sp, axis=1).mean())


F_ = lambda d: np.moveaxis(d[0].cpu().numpy(), 0, -1).astype(np.float32)
res = {k: [] for k in ("X2_perfect", "X2_realistic", "X2_cleaned", "X3S_perfect", "X3S_realistic", "X3S_cleaned")}
with torch.no_grad():
    for c in range(1, 11):
        x = torch.from_numpy(mm(DIR_IN[c])[None, None]).to(dev); t = torch.from_numpy(mm(DIR_TG[c])[None, None]).to(dev)
        rp, tp = torch.tensor([5], device=dev), torch.tensor([0], device=dev)
        seed = int(float(np.abs(DIR_TG[c]).sum()) * 1000) % (2 ** 31)
        real = realistic_target(DIR_TG[c], seed)                       # (1,2,n,n) front, side (raw -ln)
        perf = dt.drr(t, ANG)
        for name, P in (("perfect", std(perf)), ("realistic", std(real)), ("cleaned", std(clean(real)))):
            pr = std(dt.drr(x, ANG)); d = P - pr
            v3 = torch.cat([dt.backproject(P, ANG, x.shape[2:]), dt.backproject(d, ANG, x.shape[2:])], 1)
            res[f"X3S_{name}"].append(tre(F_(g3(x, rp, tp, v3, d)), c))
            P1, d1 = P[:, :1], d[:, :1]                                 # X2 = front view only
            v2_ = torch.cat([dt.backproject(P1, [0.0], x.shape[2:]), dt.backproject(d1, [0.0], x.shape[2:])], 1)
            res[f"X2_{name}"].append(tre(F_(g2(x, rp, tp, v2_, d1)), c))
        print(f"C{c:02d} " + " ".join(f"{k} {v[-1]:.2f}" for k, v in res.items()), flush=True)
summ = {k: float(np.mean(v)) for k, v in res.items()}
(HERE / "run" / "eval_cleaner.json").write_text(json.dumps({"checkpoint": str(ck), "x3s_ckpt": c3.name, "summary": summ, "cases": res}, indent=1))
print(json.dumps(summ, indent=1))

# real SPARE clinical projection before / after cleaning
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
P1 = Path("/media/abhishek/3CCA3CADCA3C6574/SpareDVFs_archive_20260915/ClinicalVarianDatasets/P1/CV_P1_T_01/Proj")
r = np.fromfile(P1 / "Proj_00171.bin", dtype=np.float32).reshape(768, 1024).reshape(192, 4, 256, 4).mean((1, 3))
r = torch.from_numpy(r.copy())[None, None].to(dev)
r128 = torch.nn.functional.interpolate(r, size=(128, 128), mode="bilinear", align_corners=False)
with torch.no_grad(): o = G(std(r128))
fig, ax = plt.subplots(1, 3, figsize=(13, 4.5))
for a_, im, ti in zip(ax, (std(r128), o, o - std(r128)), ("REAL SPARE X-ray (std, 128²)", "after cleaner", "change made by cleaner")):
    im = im[0, 0].cpu().numpy(); lo, hi = np.percentile(im, (1, 99.5)); a_.imshow(im, cmap="gray" if "change" not in ti else "RdBu_r",
                                                                              vmin=lo if "change" not in ti else -max(abs(lo), abs(hi)),
                                                                              vmax=hi if "change" not in ti else max(abs(lo), abs(hi))); a_.set_title(ti); a_.axis("off")
fig.tight_layout(); fig.savefig(HERE / "run" / "real_spare_cleaned.png", dpi=100)
