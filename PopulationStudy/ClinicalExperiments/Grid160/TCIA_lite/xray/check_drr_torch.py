"""X0 gate for drr_torch.py (no training). DIR-Lab case 1, 160³ 2 mm R3 frame (same as the scorer).
  1. rtk_matrix == RTK ThreeDCircularProjectionGeometry matrix (with and without detector offsets).
  2. torch DRR vs RTK Joseph DRR of the same volume and geometry: Pearson r > 0.99 at 0/45/90/180 deg.
  3. point blob: torch peak pixel == project_points() == RTK peak (within 1 px).
  4. back-projection: the blob voxel is in the top 0.1 % of the back-projected volume.
  5. DRR(T00) - DRR(T50) shows motion (diaphragm band); figure saved. Model input on DIR-Lab is CT_06 (T50) and
     the target is T00 (CT_01), so the conditioning X-ray is DRR(T00).
  6. timing + memory of one forward+backward DRR at 160³.
  CUDA_VISIBLE_DEVICES=0 python3 check_drr_torch.py
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/scripts")
os.environ.setdefault("DIRLAB_ROOT", "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/data/dirlab_packs")
import SimpleITK as sitk
import itk
from itk import RTK as rtk
import eval_dir_tcia3_iso2mm_v2 as v2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drr_torch as dt

OUT = Path(__file__).resolve().parent / "check"
OUT.mkdir(exist_ok=True)
dev = torch.device("cuda:0")
ANG = [0.0, 45.0, 90.0, 180.0]
res = {}


def rtk_geom(angles, ox=0.0, oy=0.0):
    g = rtk.ThreeDCircularProjectionGeometry.New()
    for a in angles:
        g.AddProjection(dt.SID, dt.SDD, float(a), float(ox), float(oy))
    return g


def rtk_drr(vol_dhw: np.ndarray, angles) -> np.ndarray:
    img = itk.image_from_array(np.ascontiguousarray(vol_dhw.astype(np.float32)))
    n = np.array(vol_dhw.shape[::-1])
    img.SetSpacing([2.0] * 3); img.SetOrigin([float(-(m - 1) / 2 * 2.0) for m in n])
    ImageF3 = itk.Image[itk.F, 3]
    src = rtk.ConstantImageSource[ImageF3].New()
    o = -(dt.N_DET - 1) / 2 * dt.DET_SPACING
    src.SetOrigin([o, o, 0.0]); src.SetSpacing([dt.DET_SPACING, dt.DET_SPACING, 1.0])
    src.SetSize([dt.N_DET, dt.N_DET, len(angles)]); src.SetConstant(0.0)
    pr = rtk.JosephForwardProjectionImageFilter[ImageF3, ImageF3].New()
    pr.SetInput(0, src.GetOutput()); pr.SetInput(1, img); pr.SetGeometry(rtk_geom(angles)); pr.Update()
    return np.asarray(itk.array_from_image(pr.GetOutput()))           # (n_angles, rows, cols)


# 1. matrices
err = 0.0
for ox, oy in ((0.0, 0.0), (148.0, -2.0)):
    g = rtk_geom([0.0, 0.53, 37.0, 90.0, 211.0], ox, oy)
    for i, a in enumerate([0.0, 0.53, 37.0, 90.0, 211.0]):
        R = np.array(itk.array_from_matrix(g.GetMatrix(i)))
        err = max(err, float(np.abs(R - dt.rtk_matrix(a, ox=ox, oy=oy)).max()))
res["1_matrix_max_abs_diff"] = err

# volumes: T50 (model input, CT_06) and T00 (target, CT_01) on the scorer's iso grid
pk = v2.pack_case(1)
t50 = pk["mu"].astype(np.float32)
ct01 = sitk.ReadImage(str(pk["train"] / "CT_01.mha"))
hu01 = sitk.GetArrayFromImage(ct01).astype(np.float32)
t00 = v2.hu_to_mu(v2.resample_to_iso(hu01, np.array(ct01.GetOrigin()), np.array(ct01.GetSpacing()),
                                     pk["origin_iso_xyz"], order=1, cval=-1000.0)).astype(np.float32)
res["volume_shape"] = list(t50.shape)

# 2. torch vs RTK
V = torch.from_numpy(t50)[None, None].to(dev)
with torch.no_grad():
    tor = dt.drr(V, ANG)[0].cpu().numpy()
rk = rtk_drr(t50, ANG)
res["2_pearson_r"] = {str(a): float(np.corrcoef(tor[i].ravel(), rk[i].ravel())[0, 1]) for i, a in enumerate(ANG)}
res["2_scale_torch_over_rtk"] = {str(a): float(tor[i].sum() / rk[i].sum()) for i, a in enumerate(ANG)}

# 3. point blob
blob = np.zeros_like(t50)
kk, jj, ii = np.meshgrid(*(np.arange(n) for n in blob.shape), indexing="ij")
c = (50, 100, 120)
blob[:] = np.exp(-((kk - c[0]) ** 2 + (jj - c[1]) ** 2 + (ii - c[2]) ** 2) / (2 * 1.5 ** 2))
world = (np.array([c[2], c[1], c[0]]) - (np.array(blob.shape[::-1]) - 1) / 2) * 2.0
B = torch.from_numpy(blob)[None, None].to(dev)
with torch.no_grad():
    tb = dt.drr(B, ANG)[0].cpu().numpy()
rb = rtk_drr(blob, ANG)
pk3 = {}
for i, a in enumerate(ANG):
    pt = np.array(np.unravel_index(tb[i].argmax(), tb[i].shape))[::-1]       # (col,row)
    pr = np.array(np.unravel_index(rb[i].argmax(), rb[i].shape))[::-1]
    pp = dt.project_points(world[None], a)[0]
    pk3[str(a)] = {"torch": pt.tolist(), "rtk": pr.tolist(), "matrix": np.round(pp, 2).tolist(),
                   "max_px_err": float(max(np.abs(pt - pr).max(), np.abs(pt - pp).max()))}
res["3_blob_peak"] = pk3

# 4. back-projection of the blob DRRs (AP + lateral): blob voxel in top 0.1 %
with torch.no_grad():
    bp = dt.backproject(torch.from_numpy(tb[[0, 2]])[None].to(dev), [ANG[0], ANG[2]], blob.shape)[0]
    s = bp.prod(0)                                                            # both rays agree only at the blob
    s_np = s.cpu().numpy()
res["4_blob_backproj_percentile"] = float((s_np < s_np[c]).mean() * 100)

# 5. motion signal
with torch.no_grad():
    d00 = dt.drr(torch.from_numpy(t00)[None, None].to(dev), [0.0, 90.0])[0].cpu().numpy()
diff = d00 - tor[[0, 2]]
lung = pk["lung_iso"]
res["5_diff_rel_rms"] = float(np.sqrt((diff ** 2).mean()) / np.sqrt((tor[[0, 2]] ** 2).mean()))

# 6. timing / memory (forward + backward, 1 angle)
torch.cuda.reset_peak_memory_stats()
Vg = V.clone().requires_grad_(True)
torch.cuda.synchronize(); t0 = time.time()
dt.drr(Vg, [0.0]).sum().backward()
torch.cuda.synchronize()
res["6_fwd_bwd_1view_s"] = time.time() - t0
res["6_peak_mem_MiB"] = torch.cuda.max_memory_allocated() / 2 ** 20

ok = (res["1_matrix_max_abs_diff"] < 1e-6 and min(res["2_pearson_r"].values()) > 0.99
      and all(v["max_px_err"] <= 1.0 for v in pk3.values()) and res["4_blob_backproj_percentile"] > 99.9)
res["ok"] = bool(ok)
(OUT / "check_drr_torch.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))

fig, ax = plt.subplots(2, 4, figsize=(16, 8.4))
for r, (a_i, name) in enumerate(((0, "AP 0°"), (2, "lateral 90°"))):
    for col, (img, t) in enumerate(((rk[a_i], f"RTK T50 {name}"), (tor[a_i], f"torch T50 {name}"),
                                    (d00[r], f"torch T00 {name}"), (diff[r], "T00 − T50 (motion)"))):
        m = np.abs(img).max()
        ax[r, col].imshow(img, cmap="RdBu_r" if col == 3 else "gray", vmin=-m if col == 3 else None,
                          vmax=m if col == 3 else None)
        ax[r, col].set_title(t); ax[r, col].axis("off")
fig.suptitle(f"X0 torch DRR check, DIR-Lab C01 · r = {min(res['2_pearson_r'].values()):.4f} · ok = {ok}")
fig.tight_layout(); fig.savefig(OUT / "check_drr_torch.png", dpi=100)
