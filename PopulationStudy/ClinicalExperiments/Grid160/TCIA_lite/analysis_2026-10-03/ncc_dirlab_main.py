"""NCC on DIR-Lab (verified 2 mm iso path): real CT_01 (T00) vs CT_06 (T50) warped by the network field.
u lives on the T00 grid (score_set: lm50 = lm00 + u(lm00)), so warped(x) = T50(x + u(x)) should match T00(x).
Region: T50 lung mask dilated 5 vox (10 mm) so the moving lung base is inside. Also whole 160^3 box."""
import sys, os, json, numpy as np, torch
from pathlib import Path
from scipy.ndimage import map_coordinates, binary_dilation
sys.path.insert(0, "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/scripts")
os.environ.setdefault("DIRLAB_ROOT", "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/data/dirlab_packs")
import SimpleITK as sitk
import eval_dir_tcia3_iso2mm_v2 as v2
import mirror_tta_iso2mm as mt
from rescore_oracle_iso2mm_v3 import load_ckpt

L = "/home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Grid160/TCIA_lite"
M = {"TCIA3.5-hybrid ep100": "/home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Grid160/TCIA3.5_hybrid/DecoderCRB/checkpoints/epoch_100.pt",
     "TCIA3.5-hybrid ep98": "/home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Grid160/TCIA3.5_hybrid/DecoderCRB/checkpoints/epoch_098.pt",
     "TCIA3.5 old ep100": "/home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Grid160/TCIA3.5/DecoderCRB/checkpoints/epoch_100.pt",
     "TCIA3.5 old ep92": "/home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Grid160/TCIA3.5/DecoderCRB/checkpoints/epoch_092.pt"}

def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def warp(img, u):  # u (Z,Y,X,3) comps dx,dy,dz in iso voxels
    zz, yy, xx = np.meshgrid(*(np.arange(n, dtype=np.float32) for n in img.shape), indexing="ij")
    return map_coordinates(img, [zz + u[..., 2], yy + u[..., 1], xx + u[..., 0]], order=1, mode="nearest")

dev = torch.device("cuda:0"); cases = range(1, 11); P, T00 = {}, {}
for c in cases:
    P[c] = v2.pack_case(c)
    ct = sitk.ReadImage(str(P[c]["train"] / "CT_01.mha"))
    T00[c] = v2.resample_to_iso(sitk.GetArrayFromImage(ct).astype(np.float32), P[c]["origin_xyz"],
                                P[c]["spacing_xyz"], P[c]["origin_iso_xyz"], order=1, cval=-1000.0)
reg = {c: binary_dilation(P[c]["lung_iso"], iterations=5) for c in cases}
rows = {"no motion (identity)": {c: (ncc(P[c]["hu_iso"][reg[c]], T00[c][reg[c]]), ncc(P[c]["hu_iso"], T00[c])) for c in cases}}
for name, path in M.items():
    g = load_ckpt(Path(path), dev)
    for var in ("plain", "mirror"):
        rows[f"{name} {var}"] = {}
    for c in cases:
        a, m = mt.predict_mirror(g, dev, P[c]["mu"])
        for var, u in (("plain", a), ("mirror", m)):
            w = warp(P[c]["hu_iso"], u)
            rows[f"{name} {var}"][c] = (ncc(w[reg[c]], T00[c][reg[c]]), ncc(w, T00[c]))
    del g; torch.cuda.empty_cache()
print("NCC (higher = better) | mean lung-region | mean whole box | per-case lung-region C01..C10")
for k, r in rows.items():
    print(f"{k:34s} {np.mean([v[0] for v in r.values()]):.4f}  {np.mean([v[1] for v in r.values()]):.4f}  "
          + " ".join(f"{r[c][0]:.3f}" for c in cases))
json.dump({k: {str(c): v for c, v in r.items()} for k, r in rows.items()},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ncc_dirlab_main.json"), "w"), indent=1)
