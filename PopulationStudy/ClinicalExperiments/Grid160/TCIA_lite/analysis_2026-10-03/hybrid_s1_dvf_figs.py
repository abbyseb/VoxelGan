"""TCIA3.5-hybrid seed 1 ep100 (+ mirror) on DIR-Lab: predicted DVF, warped CT, target CT; Elastix DVF for reference.
Saves per case .mha (input CT_06, target CT_01, warped, DVF in mm, Elastix DVF in mm) on the 2 mm 160^3 cube, and a
coronal figure for cases 1, 4, 8. Array axes (Z,Y,X): X left-right, Y head-foot (+ toward the feet in these arrays; checked visually), Z front-back."""
import sys, os, numpy as np, torch, SimpleITK as sitk, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import map_coordinates
sys.path.insert(0, "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/scripts")
os.environ.setdefault("DIRLAB_ROOT", "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/data/dirlab_packs")
import eval_dir_tcia3_iso2mm_v2 as v2, mirror_tta_iso2mm as mt
from rescore_oracle_iso2mm_v3 import load_ckpt
OUT = Path(os.path.dirname(os.path.abspath(__file__))); VOL = OUT / "hybrid_s1_volumes"; S = v2.SPACING
dev = torch.device("cuda:0")
g = load_ckpt(Path("/home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Grid160/TCIA3.5_hybrid/DecoderCRB/checkpoints/epoch_100.pt"), dev)
def warp(vol, u):
    zz, yy, xx = np.meshgrid(*(np.arange(n, dtype=np.float32) for n in vol.shape), indexing="ij")
    return map_coordinates(vol, [zz + u[..., 2], yy + u[..., 1], xx + u[..., 0]], order=1, mode="nearest")
def save(arr, name, pk, vec=False):
    im = sitk.GetImageFromArray(arr.astype(np.float32), isVector=vec); im.SetSpacing((S, S, S)); im.SetOrigin(tuple(float(v) for v in pk["origin_iso_xyz"]))
    sitk.WriteImage(im, str(VOL / name), True)
data = {}
for c in range(1, 11):
    pk = v2.pack_case(c); _, u = mt.predict_mirror(g, dev, pk["mu"])
    t00 = v2.resample_to_iso(sitk.GetArrayFromImage(sitk.ReadImage(str(pk["train"] / "CT_01.mha"))).astype(np.float32), pk["origin_xyz"], pk["spacing_xyz"], pk["origin_iso_xyz"], 1, -1000.0)
    w = warp(pk["hu_iso"], u); el = v2.elastix_on_iso(pk)
    for arr, nm, vec in ([] if (VOL / f"C{c:02d}_dvf_net_mm.mha").exists() else ((pk["hu_iso"], "input_CT06_T50", False), (t00, "target_CT01_T00", False), (w, "warped_CT06_by_net", False), (u * S, "dvf_net_mm", True), (el * S, "dvf_elastix_mm", True))):
        save(arr, f"C{c:02d}_{nm}.mha", pk, vec)
    data[c] = (pk, u, t00, w, el)
    print("saved", c, flush=True)
cases = [1, 4, 8]
fig, ax = plt.subplots(len(cases), 6, figsize=(24, 4.4 * len(cases)))
for i, c in enumerate(cases):
    pk, u, t00, w, el = data[c]; z = int(np.round(np.where(pk["lung_iso"])[0].mean()))
    sl = lambda a: a[z]  # coronal: (Y, X)
    kw = dict(cmap="gray", vmin=-1000, vmax=300, origin="upper")
    mag = np.linalg.norm(u[z] * S, axis=-1); emag = np.linalg.norm(el[z] * S, axis=-1); vmax = max(emag.max(), 1)
    ax[i, 0].imshow(sl(pk["hu_iso"]), **kw); ax[i, 0].set_title(f"C{c:02d} input CT_06 (T50, exhale)")
    ax[i, 1].imshow(sl(t00), **kw); ax[i, 1].set_title("target CT_01 (T00, inhale)")
    ax[i, 2].imshow(sl(w), **kw); ax[i, 2].set_title("warped CT_06 by network")
    ax[i, 3].imshow(sl(t00) - sl(pk["hu_iso"]), cmap="bwr", vmin=-600, vmax=600, origin="upper"); ax[i, 3].set_title("target - input (before)")
    ax[i, 4].imshow(sl(t00) - sl(w), cmap="bwr", vmin=-600, vmax=600, origin="upper"); ax[i, 4].set_title("target - warped (after)")
    ax[i, 5].imshow(mag, cmap="magma", vmin=0, vmax=vmax, origin="upper"); st = 6
    yy, xx = np.mgrid[0:160:st, 0:160:st]
    ax[i, 5].quiver(xx, yy, u[z, ::st, ::st, 0] * S, u[z, ::st, ::st, 1] * S, color="cyan", angles="xy", scale_units="xy", scale=0.5, width=0.003, label="network")
    ax[i, 5].quiver(xx, yy, el[z, ::st, ::st, 0] * S, el[z, ::st, ::st, 1] * S, color="lime", angles="xy", scale_units="xy", scale=0.5, width=0.002, alpha=.6, label="Elastix")
    ax[i, 5].set_title(f"network |DVF| (mm), max {mag.max():.1f}; Elastix max {emag.max():.1f}"); ax[i, 5].legend(loc="lower right", fontsize=8)
    for a in ax[i]: a.axis("off")
plt.suptitle("TCIA3.5-hybrid seed 1, epoch 100 + mirror — coronal slice through lung centre (head at top; array Y increases toward the feet)", fontsize=14)
plt.tight_layout(); plt.savefig(OUT / "hybrid_s1_dvf_warped_target.png", dpi=90)
print("figure saved")
