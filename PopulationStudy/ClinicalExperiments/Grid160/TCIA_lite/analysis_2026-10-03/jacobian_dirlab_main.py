"""Jacobian determinant of the network map phi(x) = x + u(x) on DIR-Lab (2 mm iso, T00 grid).
det J <= 0 -> folding (physically impossible). Lung region = T50 lung mask dilated 5 vox, as in ncc_dirlab.py."""
import sys, os, json, numpy as np, torch
from pathlib import Path
from scipy.ndimage import binary_dilation
sys.path.insert(0, "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/scripts")
os.environ.setdefault("DIRLAB_ROOT", "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/data/dirlab_packs")
import eval_dir_tcia3_iso2mm_v2 as v2
import mirror_tta_iso2mm as mt
from rescore_oracle_iso2mm_v3 import load_ckpt

L = "/home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Grid160/TCIA_lite"
M = {**{f"TCIA3.5-hybrid ep{e}": f"/home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Grid160/TCIA3.5_hybrid/DecoderCRB/checkpoints/epoch_{e:03d}.pt" for e in range(96, 101)},
     "TCIA3.5 old ep100": "/home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Grid160/TCIA3.5/DecoderCRB/checkpoints/epoch_100.pt"}

def detj(u):  # u (Z,Y,X,3) comps dx,dy,dz, voxel units; axes z=0,y=1,x=2
    g = [[np.gradient(u[..., c], axis=a) for a in (2, 1, 0)] for c in range(3)]  # g[c][d] = d u_c / d x_d
    J = np.empty(u.shape[:3] + (3, 3), np.float32)
    for c in range(3):
        for d in range(3):
            J[..., c, d] = g[c][d] + (1.0 if c == d else 0.0)
    return np.linalg.det(J)

dev = torch.device("cuda:0"); cases = range(1, 11)
P = {c: v2.pack_case(c) for c in cases}
reg = {c: binary_dilation(P[c]["lung_iso"], iterations=5) for c in cases}
rows = {}
for name, path in M.items():
    g = load_ckpt(Path(path), dev)
    for c in cases:
        a, m = mt.predict_mirror(g, dev, P[c]["mu"])
        for var, u in (("plain", a), ("mirror", m)):
            d = detj(u); dl = d[reg[c]]
            rows.setdefault(f"{name} {var}", {})[c] = dict(
                fold_lung_pct=float((dl <= 0).mean() * 100), fold_box_pct=float((d <= 0).mean() * 100),
                min_lung=float(dl.min()), mean_lung=float(dl.mean()), sd_lung=float(dl.std()),
                p1_lung=float(np.percentile(dl, 1)), p99_lung=float(np.percentile(dl, 99)))
    del g; torch.cuda.empty_cache()
print("det J | lung: % folded, min, 1st pct, mean, sd, 99th pct | whole box % folded")
for k, r in rows.items():
    f = lambda key: np.mean([v[key] for v in r.values()])
    print(f"{k:34s} {f('fold_lung_pct'):.4f}%  min {min(v['min_lung'] for v in r.values()):+.3f}  p1 {f('p1_lung'):.3f}  "
          f"mean {f('mean_lung'):.3f}  sd {f('sd_lung'):.3f}  p99 {f('p99_lung'):.3f} | box {f('fold_box_pct'):.4f}%")
print("\nTCIA3.5-hybrid ep100 mirror per case: % folded lung / min / mean")
for c, v in rows["TCIA3.5-hybrid ep100 mirror"].items():
    print(f"  C{c:02d}  {v['fold_lung_pct']:.4f}%  {v['min_lung']:+.3f}  {v['mean_lung']:.3f}")
json.dump({k: {str(c): v for c, v in r.items()} for k, r in rows.items()},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "jacobian_dirlab_main.json"), "w"), indent=1)
