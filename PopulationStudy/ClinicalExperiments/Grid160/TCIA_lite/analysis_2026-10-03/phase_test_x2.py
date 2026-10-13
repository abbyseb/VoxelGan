"""In-between phases from only TWO X-rays (exhale = planning CT T50, inhale = T00)? DIR-Lab 75-point sets on T00..T40.
Input CT = T50 (exhale) for every method; target = phase Tk (k = 0..4); TRE = |lk + u(lk) - l50| in mm (2 mm iso scorer).
  own X-ray   X2 with the X-ray of Tk itself            (needs an X-ray of every moment; the clinic has this)
  old model   LNCC-only CT model, phase numbers 5 -> k (no X-ray)
  scale       alpha_k * X2(T50 -> T00 with the T00 X-ray), alpha_k = (1 + cos(pi k / 5)) / 2 (sinusoidal breathing)
  combine     old model's 5 -> k field * s, s = mean|X2 full field| / mean|old 5 -> 0 field| in the lung
              (old model gives the shape, the two X-rays give the breath size)
  no motion   u = 0
  python3 phase_test_x2.py [X2 checkpoint]"""
import json, os, sys, importlib.util
from pathlib import Path
import numpy as np, torch, SimpleITK as sitk
sys.path.insert(0, "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/scripts")
os.environ.setdefault("DIRLAB_ROOT", "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/data/dirlab_packs")
import eval_dir_tcia3_iso2mm_v2 as v2
from dirlab_tre import landmarks_75
L = Path("/home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Grid160/TCIA_lite")
spec = importlib.util.spec_from_file_location("x2", L / "run_A2_full160_aug_xray2/scripts/train_xray2.py")
X2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(X2)
dev = torch.device("cuda:0"); Sp = v2.SPACING; mm = v2.minmax
ck2 = Path(sys.argv[1]) if len(sys.argv) > 1 else L / "run_A2_full160_aug_xray2/DecoderCRB/checkpoints/latest.pt"
s2 = torch.load(str(ck2), map_location="cpu", weights_only=False)
g2 = X2.XrayUNet2(im_size=160, n_phases=10); g2.load_state_dict(s2["generator"]); g2 = g2.to(dev).eval()
g0 = X2.UNetCRBDecoder(im_size=160, n_phases=10)
g0.load_state_dict(torch.load(str(L / "run_A2_full160_aug_unsup_lncc/DecoderCRB/checkpoints/epoch_040.pt"),
                              map_location="cpu", weights_only=False)["generator"]); g0 = g0.to(dev).eval()
T = lambda a: torch.from_numpy(mm(a)[None, None]).to(dev)
F = lambda d: np.moveaxis(d[0].cpu().numpy(), 0, -1).astype(np.float32)


def phase_ct(pk, idx1):
    im = sitk.ReadImage(str(pk["train"] / f"CT_{idx1:02d}.mha"))
    return v2.hu_to_mu(v2.resample_to_iso(sitk.GetArrayFromImage(im).astype(np.float32), np.array(im.GetOrigin()),
                                          np.array(im.GetSpacing()), pk["origin_iso_xyz"], order=1, cval=-1000.0))


def tre(u, c, pk, ph):
    lk = v2.official_to_iso(landmarks_75(c, ph), pk); l5 = v2.official_to_iso(landmarks_75(c, "T50"), pk)
    return float(np.linalg.norm((lk + (v2.sample_u(u, lk) if u is not None else 0) - l5) * Sp, axis=1).mean())


METHODS = ["own X-ray", "old model", "scale", "combine", "no motion"]
res = {m: {} for m in METHODS}
with torch.no_grad():
    for c in range(1, 11):
        pk = v2.pack_case(c); ref = T(pk["mu"]); lung = pk["lung_iso"]
        full = F(g2(ref, torch.tensor([5], device=dev), torch.tensor([0], device=dev), *X2.xray_inputs(ref, T(phase_ct(pk, 1)))))
        old0 = F(g0(ref, torch.tensor([5], device=dev), torch.tensor([0], device=dev)))
        s = np.linalg.norm(full, axis=-1)[lung].mean() / max(np.linalg.norm(old0, axis=-1)[lung].mean(), 1e-6)
        for k in range(5):
            ph = f"T{10 * k:02d}"
            own = F(g2(ref, torch.tensor([5], device=dev), torch.tensor([k], device=dev), *X2.xray_inputs(ref, T(phase_ct(pk, k + 1)))))
            oldk = F(g0(ref, torch.tensor([5], device=dev), torch.tensor([k], device=dev)))
            a = (1 + np.cos(np.pi * k / 5)) / 2
            for m, u in (("own X-ray", own), ("old model", oldk), ("scale", a * full), ("combine", s * oldk), ("no motion", None)):
                res[m].setdefault(ph, []).append(tre(u, c, pk, ph))
        print(f"C{c:02d} done (size factor s = {s:.2f})", flush=True)
out = {"x2_checkpoint": str(ck2), "x2_epoch": s2.get("epoch_done"), "tre_75_mm": res}
(Path(__file__).resolve().parent / "phase_test_x2.json").write_text(json.dumps(out, indent=1))
print(f"\nX2 epoch {s2.get('epoch_done')} · DIR-Lab 75-point TRE (mm), mean of 10 cases, input = T50 (exhale)")
print(f"{'phase':6s}" + "".join(f"{m:>12s}" for m in METHODS))
for k in range(5):
    ph = f"T{10 * k:02d}"
    print(f"{ph:6s}" + "".join(f"{np.mean(res[m][ph]):12.2f}" for m in METHODS))
print(f"{'mean':6s}" + "".join(f"{np.mean([np.mean(res[m][f'T{10 * k:02d}']) for k in range(5)]):12.2f}" for m in METHODS))
