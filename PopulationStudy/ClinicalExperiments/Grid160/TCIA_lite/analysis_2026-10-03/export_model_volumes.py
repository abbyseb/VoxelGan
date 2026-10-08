"""Export DIR-Lab volumes for every main model (seed 1 where seeds exist; all-82 only has seed 2) into ONE folder:
  model_volumes/common/CXX_{input_CT06_T50,target_CT01_T00,dvf_elastix_mm}.mha
  model_volumes/<model>/CXX_{warped_CT06,dvf_mm}.mha      (prediction with left-right mirror averaging)
plus model_volumes/tre_summary.csv and a comparison figure (coronal 'after' differences, cases 1, 4, 8).
2 mm 160^3 cube; array axes (Z,Y,X) = front-back, head-foot (+ toward feet), left-right; DVF in mm (dx, dy, dz)."""
import sys, os, numpy as np, torch, SimpleITK as sitk, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import map_coordinates
sys.path.insert(0, "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/scripts")
os.environ.setdefault("DIRLAB_ROOT", "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/data/dirlab_packs")
import eval_dir_tcia3_iso2mm_v2 as v2, mirror_tta_iso2mm as mt
from rescore_oracle_iso2mm_v3 import load_ckpt
from dirlab_tre import landmarks_300
G = "/home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Grid160"; L = G + "/TCIA_lite"
MODELS = {  # folder name: checkpoint
    "01_TCIA3_old_ep100": f"{G}/TCIA3/DecoderCRB/checkpoints/epoch_100.pt",
    "02_MagFT_ep30": f"{G}/TCIA3_magFT/DecoderCRB/checkpoints/epoch_030.pt",
    "03_TCIA3.5_old_s1_ep100": f"{G}/TCIA3.5/DecoderCRB/checkpoints/epoch_100.pt",
    "04_TCIA3.5_hybrid_s1_ep100": f"{G}/TCIA3.5_hybrid/DecoderCRB/checkpoints/epoch_100.pt",
    "05_small_full160_ep40": f"{L}/run_A2_full160/DecoderCRB/checkpoints/epoch_040.pt",
    "06_small_full160aug_old_ep40": f"{L}/run_A2_full160_aug/DecoderCRB/checkpoints/epoch_040.pt",
    "07_small_push_ep40": f"{L}/run_A2_full160_aug_push/DecoderCRB/checkpoints/epoch_040.pt",
    "08_small_hybrid_s1_ep40": f"{L}/run_A2_full160_aug_hybrid/DecoderCRB/checkpoints/epoch_040.pt",
    "09_small_hybrid_img30_ep40": f"{L}/run_A2_full160_aug_hybrid_img30/DecoderCRB/checkpoints/epoch_040.pt",
    "10_small_hybrid_deepest_ep40": f"{L}/run_A2_full160_aug_hybrid_extreme/DecoderCRB/checkpoints/epoch_040.pt",
    "11_small_hybrid_all82_s2_ep40": f"{L}/run_A2_full160_aug_hybrid_s2_all82/DecoderCRB/checkpoints/epoch_040.pt",
}
OUT = Path(os.path.dirname(os.path.abspath(__file__))) / "model_volumes"; (OUT / "common").mkdir(parents=True, exist_ok=True); S = v2.SPACING
dev = torch.device("cuda:0")
def warp(vol, u):
    zz, yy, xx = np.meshgrid(*(np.arange(n, dtype=np.float32) for n in vol.shape), indexing="ij")
    return map_coordinates(vol, [zz + u[..., 2], yy + u[..., 1], xx + u[..., 0]], order=1, mode="nearest")
def save(arr, path, pk, vec=False):
    if path.exists(): return
    im = sitk.GetImageFromArray(arr.astype(np.float32), isVector=vec); im.SetSpacing((S, S, S)); im.SetOrigin(tuple(float(v) for v in pk["origin_iso_xyz"])); sitk.WriteImage(im, str(path), True)
P, T00 = {}, {}
for c in range(1, 11):
    pk = P[c] = v2.pack_case(c)
    T00[c] = v2.resample_to_iso(sitk.GetArrayFromImage(sitk.ReadImage(str(pk["train"] / "CT_01.mha"))).astype(np.float32), pk["origin_xyz"], pk["spacing_xyz"], pk["origin_iso_xyz"], 1, -1000.0)
    save(pk["hu_iso"], OUT / "common" / f"C{c:02d}_input_CT06_T50.mha", pk); save(T00[c], OUT / "common" / f"C{c:02d}_target_CT01_T00.mha", pk)
    save(v2.elastix_on_iso(pk) * S, OUT / "common" / f"C{c:02d}_dvf_elastix_mm.mha", pk, True)
def tre(u, c):
    pk = P[c]; l0 = v2.official_to_iso(landmarks_300(c, "T00"), pk); l5 = v2.official_to_iso(landmarks_300(c, "T50"), pk)
    return float(np.linalg.norm((l0 + v2.sample_u(u, l0) - l5) * S, axis=1).mean())
rows, FIG = [], {}
for name, ck in MODELS.items():
    d = OUT / name; d.mkdir(exist_ok=True); g = load_ckpt(Path(ck), dev); t = []
    for c in range(1, 11):
        _, u = mt.predict_mirror(g, dev, P[c]["mu"]); w = warp(P[c]["hu_iso"], u)
        save(w, d / f"C{c:02d}_warped_CT06.mha", P[c]); save(u * S, d / f"C{c:02d}_dvf_mm.mha", P[c], True); t.append(tre(u, c))
        if c in (1, 4, 8):
            z = int(np.round(np.where(P[c]["lung_iso"])[0].mean())); FIG[(name, c)] = (T00[c][z] - w[z], float(np.linalg.norm(u[z] * S, axis=-1).max()))
    (d / "checkpoint.txt").write_text(ck + "\nprediction: plain + left-right mirror average\n")
    rows.append([name] + t); print(name, f"TRE300 mirror mean {np.mean(t):.3f}", flush=True); del g; torch.cuda.empty_cache()
with open(OUT / "tre_summary.csv", "w") as f:
    f.write("model," + ",".join(f"C{c:02d}" for c in range(1, 11)) + ",mean\n")
    for r in rows: f.write(r[0] + "," + ",".join(f"{x:.3f}" for x in r[1:]) + f",{np.mean(r[1:]):.3f}\n")
cases = (1, 4, 8); fig, ax = plt.subplots(len(MODELS) + 1, 3, figsize=(10, 3.2 * (len(MODELS) + 1)))
for j, c in enumerate(cases):
    z = int(np.round(np.where(P[c]["lung_iso"])[0].mean()))
    ax[0, j].imshow(T00[c][z] - P[c]["hu_iso"][z], cmap="bwr", vmin=-600, vmax=600); ax[0, j].set_title(f"C{c:02d}: target - input (no motion)", fontsize=9)
    for i, name in enumerate(MODELS, 1):
        diff, mx = FIG[(name, c)]; tr = rows[i - 1][c]
        ax[i, j].imshow(diff, cmap="bwr", vmin=-600, vmax=600); ax[i, j].set_title(f"{name}\nTRE {tr:.2f} mm, max |u| {mx:.1f} mm", fontsize=8)
for a in ax.ravel(): a.axis("off")
plt.suptitle("target - warped (coronal, head at top); blue/red bands = remaining mismatch", fontsize=11); plt.tight_layout()
plt.savefig(OUT / "comparison_after_difference.png", dpi=80); print("done")
