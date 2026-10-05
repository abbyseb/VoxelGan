"""Probe 2: does the image term point toward the TRUE motion? DIR-Lab, TCIA3.5-hybrid ep100 (plain).
Correction c(x) = smooth interpolation (Gaussian, sigma 6 vox = 12 mm) of landmark residuals (true - predicted) at the
T00 landmarks. Field u_a = u + a*c, a in {-1,-0.5,0,0.5,1}. TRE must fall as a -> 1 (sanity). Then score each u_a with the
L1 image term and 1-LNCC (9^3): warped(x) = T50(x + u(x)) vs real T00, lung mask dilated 5 vox, images = min-max mu."""
import sys, os, json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from scipy.ndimage import binary_dilation
sys.path.insert(0, "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/scripts")
os.environ.setdefault("DIRLAB_ROOT", "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/data/dirlab_packs")
import SimpleITK as sitk, eval_dir_tcia3_iso2mm_v2 as v2
from rescore_oracle_iso2mm_v3 import load_ckpt
from dirlab_tre import landmarks_300
dev = torch.device("cuda:0")
g = load_ckpt(Path("/home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Grid160/TCIA3.5_hybrid/DecoderCRB/checkpoints/epoch_100.pt"), dev)
A = [-1.0, -0.5, 0.0, 0.5, 1.0]
def warp(img, u):  # img (1,1,Z,Y,X); u (Z,Y,X,3) dx,dy,dz voxels
    Z, Y, X = img.shape[-3:]; zz, yy, xx = torch.meshgrid(*(torch.arange(n, device=dev, dtype=torch.float32) for n in (Z, Y, X)), indexing="ij")
    grid = torch.stack([2 * (xx + u[..., 0]) / (X - 1) - 1, 2 * (yy + u[..., 1]) / (Y - 1) - 1, 2 * (zz + u[..., 2]) / (Z - 1) - 1], -1)[None]
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)
def l1(w, t, m): return float(((t - w).abs() * m).sum() / m.sum())
def lncc(w, t, m, k=9):
    o = torch.ones(1, 1, k, k, k, device=dev) / k**3; mu = lambda x: F.conv3d(x, o, padding=k // 2)
    a, b = mu(w), mu(t); cc = (mu(w * t) - a * b) / torch.sqrt((mu(w * w) - a * a).clamp_min(1e-8) * (mu(t * t) - b * b).clamp_min(1e-8))
    return float(1 - (cc * m).sum() / m.sum())
out = {}
for c in range(1, 11):
    pk = v2.pack_case(c); u = torch.from_numpy(v2.predict(g, dev, pk["mu"], flip=False)).to(dev)
    t00 = v2.resample_to_iso(sitk.GetArrayFromImage(sitk.ReadImage(str(pk["train"] / "CT_01.mha"))).astype(np.float32), pk["origin_xyz"], pk["spacing_xyz"], pk["origin_iso_xyz"], 1, -1000.0)
    t50 = torch.from_numpy(v2.minmax(v2.hu_to_mu(pk["hu_iso"]))[None, None]).to(dev); t00 = torch.from_numpy(v2.minmax(v2.hu_to_mu(t00))[None, None]).to(dev)
    m = torch.from_numpy(binary_dilation(pk["lung_iso"], iterations=5)[None, None].astype(np.float32)).to(dev)
    l0 = v2.official_to_iso(landmarks_300(c, "T00"), pk); l5 = v2.official_to_iso(landmarks_300(c, "T50"), pk)
    res = torch.from_numpy((l5 - (l0 + v2.sample_u(u.cpu().numpy(), l0))).astype(np.float32)).to(dev)  # (N,3) xyz voxels
    pts = torch.from_numpy(l0.astype(np.float32)).to(dev)
    Z, Y, X = u.shape[:3]; zz, yy, xx = torch.meshgrid(*(torch.arange(n, device=dev, dtype=torch.float32) for n in (Z, Y, X)), indexing="ij")
    P = torch.stack([xx, yy, zz], -1).reshape(-1, 3); num = torch.zeros(P.shape[0], 3, device=dev); den = torch.zeros(P.shape[0], device=dev)
    for s in range(0, P.shape[0], 400000):
        w = torch.exp(-((P[s:s + 400000, None, :] - pts[None]) ** 2).sum(-1) / (2 * 6.0**2)); num[s:s + 400000] = w @ res; den[s:s + 400000] = w.sum(1)
    corr = (num / (den[:, None] + 0.05)).reshape(Z, Y, X, 3)
    row = {}
    for a in A:
        ua = u + a * corr; w = warp(t50, ua)
        tre = float(np.linalg.norm((l0 + v2.sample_u(ua.cpu().numpy(), l0) - l5) * v2.SPACING, axis=1).mean())
        row[a] = {"tre": tre, "l1": l1(w, t00, m), "lncc": lncc(w, t00, m)}
    out[c] = row
    print(f"C{c:02d} " + "  ".join(f"a={a:+.1f}: TRE {row[a]['tre']:.2f} L1 {row[a]['l1']:.5f} LNCC {row[a]['lncc']:.4f}" for a in (-1.0, 0.0, 1.0)), flush=True)
def summ(key):
    better = sum(out[c][1.0][key] < out[c][0.0][key] for c in out); worse_away = sum(out[c][-1.0][key] > out[c][0.0][key] for c in out)
    rel = np.median([(out[c][0.0][key] - out[c][1.0][key]) / out[c][0.0][key] * 100 for c in out])
    return better, worse_away, rel
for k in ("tre", "l1", "lncc"):
    b, w, r = summ(k); print(f"{k:5s}: toward truth lowers it in {b}/10 cases, away from truth raises it in {w}/10, median change toward truth -{r:.2f}%")
json.dump({str(c): {str(a): v for a, v in r.items()} for c, r in out.items()}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lncc_truth_probe.json"), "w"), indent=1)
