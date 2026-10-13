"""How much of X1 / X2's DIR-Lab TRE survives realistic X-rays? No retraining: only the TARGET X-ray (T00) is replaced by
xray/realistic_xray.py (polychromatic + bone/tissue + scatter + blur + Poisson/electronic noise [+ 1 px shift]).
The planning-CT X-ray Dr used by X2's difference image stays a clean software DRR (that is what the clinic would have).
Levels: clean (as trained) | physics (spectrum + materials only) | mild | moderate | severe. DIR-Lab 300-point TRE, plain.
Noise is seeded per case (same noise for X1 and X2).
  python3 realism_test.py [X2 checkpoint]"""
import json, os, sys, importlib.util
from pathlib import Path
import numpy as np, torch, SimpleITK as sitk
sys.path.insert(0, "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/scripts")
os.environ.setdefault("DIRLAB_ROOT", "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/data/dirlab_packs")
import eval_dir_tcia3_iso2mm_v2 as v2
from dirlab_tre import landmarks_300
L = Path("/home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Grid160/TCIA_lite")
sys.path.insert(0, str(L / "xray"))
import realistic_xray as rx
def load_mod(name, p):
    s = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
X1 = load_mod("x1", L / "run_A2_full160_aug_xray1/scripts/train_xray1.py")
X2 = load_mod("x2", L / "run_A2_full160_aug_xray2/scripts/train_xray2.py")
dev = torch.device("cuda:0"); Sp = v2.SPACING; mm = v2.minmax; dt = X2.dt
ck2 = Path(sys.argv[1]) if len(sys.argv) > 1 else L / "run_A2_full160_aug_xray2/DecoderCRB/checkpoints/latest.pt"
s2 = torch.load(str(ck2), map_location="cpu", weights_only=False)
g2 = X2.XrayUNet2(im_size=160, n_phases=10); g2.load_state_dict(s2["generator"]); g2 = g2.to(dev).eval()
g1 = X1.XrayUNet(n_xray=1, im_size=160, n_phases=10)
g1.load_state_dict(torch.load(str(L / "run_A2_full160_aug_xray1/DecoderCRB/checkpoints/epoch_040.pt"), map_location="cpu",
                              weights_only=False)["generator"]); g1 = g1.to(dev).eval()
T = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)[None, None]).to(dev)
LEVELS = list(rx.LEVELS)


def t00_hu(pk):
    im = sitk.ReadImage(str(pk["train"] / "CT_01.mha"))
    return v2.resample_to_iso(sitk.GetArrayFromImage(im).astype(np.float32), np.array(im.GetOrigin()),
                              np.array(im.GetSpacing()), pk["origin_iso_xyz"], order=1, cval=-1000.0)


def target_proj(hu_t, level, gen):
    if level == "clean":
        return dt.drr(T(mm(v2.hu_to_mu(hu_t))), [0.0])
    return rx.realistic_projection(T(hu_t), 0.0, level, gen)


def tre(u, c, pk):
    l0 = v2.official_to_iso(landmarks_300(c, "T00"), pk); l5 = v2.official_to_iso(landmarks_300(c, "T50"), pk)
    return float(np.linalg.norm((l0 + v2.sample_u(u, l0) - l5) * Sp, axis=1).mean())


F = lambda d: np.moveaxis(d[0].cpu().numpy(), 0, -1).astype(np.float32)
res = {m: {lv: [] for lv in LEVELS} for m in ("X1", "X2")}
figs = {}
with torch.no_grad():
    for c in range(1, 11):
        pk = v2.pack_case(c); ref = T(mm(pk["mu"])); hu_t = t00_hu(pk)
        pr = X2._std2d(dt.drr(ref, [0.0]))
        rp, tp = torch.tensor([5], device=dev), torch.tensor([0], device=dev)
        for lv in LEVELS:
            gen = torch.Generator().manual_seed(1000 + c)
            pt = X2._std2d(target_proj(hu_t, lv, gen))
            if c == 8:
                figs[lv] = (pt[0, 0].cpu().numpy(), (pt - pr)[0, 0].cpu().numpy())
            x1 = dt.backproject(pt, [0.0], ref.shape[2:])
            res["X1"][lv].append(tre(F(g1(ref, rp, tp, x1)), c, pk))
            d = pt - pr
            vol = torch.cat([dt.backproject(pt, [0.0], ref.shape[2:]), dt.backproject(d, [0.0], ref.shape[2:])], 1)
            res["X2"][lv].append(tre(F(g2(ref, rp, tp, vol, d)), c, pk))
        print(f"C{c:02d} " + " | ".join(f"{lv} X1 {res['X1'][lv][-1]:.2f} X2 {res['X2'][lv][-1]:.2f}" for lv in LEVELS), flush=True)

out = {"x2_checkpoint": str(ck2), "x2_epoch": s2.get("epoch_done"), "x1_checkpoint": "epoch_040", "levels": rx.LEVELS,
       "dir_tre300_plain": res}
A = Path(__file__).resolve().parent
(A / "realism_test.json").write_text(json.dumps(out, indent=1, default=str))
print(f"\nDIR-Lab TRE300 (mm), mean of 10 cases · X2 epoch {s2.get('epoch_done')}")
print(f"{'level':10s}{'X1':>8s}{'X2':>8s}")
for lv in LEVELS:
    print(f"{lv:10s}{np.mean(res['X1'][lv]):8.2f}{np.mean(res['X2'][lv]):8.2f}")

import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, len(LEVELS), figsize=(4 * len(LEVELS), 8))
for i, lv in enumerate(LEVELS):
    x, d = figs[lv]; m = np.percentile(np.abs(d), 99.5)
    ax[0, i].imshow(x, cmap="gray"); ax[0, i].set_title(f"{lv}: target X-ray\nX2 TRE {np.mean(res['X2'][lv]):.2f} mm")
    ax[1, i].imshow(d, cmap="RdBu_r", vmin=-m, vmax=m); ax[1, i].set_title(f"{lv}: 'what moved'")
for a in ax.flat: a.axis("off")
fig.suptitle("DIR-Lab C08 X-ray realism levels (as fed to X2)"); fig.tight_layout()
fig.savefig(L / "xray/check/realism_levels_c08.png", dpi=95)
