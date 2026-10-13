"""Real clinical kV projection (SPARE ClinicalVarian P1, Varian half-fan CBCT) vs our simulated X-ray of the SAME patient,
SAME projection matrix (Geometry.xml) and the SAME breathing bin (prior 4D-CT CT_{bin}.mha, HU).
Simulation: soft/bone mass line integrals by ray-marching the prior CT in physical (RTK world) coordinates, then
realistic_xray.from_line_integrals at each level; 'clean' = plain line integral of mu (what the models were trained on).
Detector: 1024 x 768, 0.388 mm, origin (-200, -150) mm (VoxelMap/MC-Varian convention), simulated at 4x4 binning.
Note: the prior CT is from an earlier day than the scan, so anatomy/setup are not identical (look at contrast/texture).
  python3 compare_real_xray.py"""
import sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, SimpleITK as sitk
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import realistic_xray as rx
sys.path.insert(0, "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/tools")
from tre_viewer.drr import parse_geometry_xml

P = Path("/media/abhishek/3CCA3CADCA3C6574/SpareDVFs_archive_20260915/ClinicalVarianDatasets/P1")
SCAN = P / "CV_P1_T_01" / "Proj"
dev = torch.device("cuda:0"); BIN = 4; NC, NR, PX = 1024, 768, 0.388; U0, V0 = -200.0, -150.0
geo = parse_geometry_xml(SCAN / "Geometry.xml")
resp = np.loadtxt(SCAN / "RespBin.csv", delimiter=",").ravel().astype(int)
ang = geo.gantry_deg % 360
cand = [i for i in range(len(ang)) if resp[i] == 1]
i = min(cand, key=lambda k: min(abs(ang[k] - 180), abs(ang[k] - 0)))            # inhale projection closest to AP/PA
real = np.fromfile(SCAN / f"Proj_{i + 1:05d}.bin", dtype=np.float32).reshape(NR, NC)

im = sitk.ReadImage(str(P / "CV_P1_Prior" / "CT_01.mha"))                     # bin 1 = max inhale
hu = torch.from_numpy(sitk.GetArrayFromImage(im).astype(np.float32))[None, None].to(dev)
org, sp = np.array(im.GetOrigin()), np.array(im.GetSpacing()); shp = np.array(im.GetSize())  # (x,y,z)
soft, bone = rx.materials(hu)
M = geo.matrices[i]; C = -np.linalg.solve(M[:, :3], M[:, 3])
cols = U0 + (np.arange(NC // BIN) * BIN + (BIN - 1) / 2) * PX
rows = V0 + (np.arange(NR // BIN) * BIN + (BIN - 1) / 2) * PX
uu, vv = np.meshgrid(cols, rows, indexing="xy")
d = np.stack([uu, vv, np.ones_like(uu)], -1) @ np.linalg.inv(M[:, :3]).T
d /= np.linalg.norm(d, axis=-1, keepdims=True); d *= np.sign(d @ -C)[..., None]
t = torch.arange(1000 - 350, 1000 + 350, 1.0, device=dev)
Ct, dt_ = torch.tensor(C, dtype=torch.float32, device=dev), torch.tensor(d, dtype=torch.float32, device=dev)
lin = {}
with torch.no_grad():
    for name, vol in (("soft", soft), ("bone", bone), ("mu", (1 + hu / 1000).clamp_min(0))):
        acc = torch.zeros(len(rows), len(cols), device=dev)
        for r0 in range(0, len(rows), 32):
            pts = Ct + dt_[r0:r0 + 32, :, None, :] * t[None, None, :, None]
            idx = (pts - torch.tensor(org, dtype=torch.float32, device=dev)) / torch.tensor(sp, dtype=torch.float32, device=dev)
            g = idx / torch.tensor((shp - 1) / 2, dtype=torch.float32, device=dev) - 1
            acc[r0:r0 + 32] = F.grid_sample(vol, g[None], mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0].sum(-1)
        lin[name] = acc[None, None] / 10.0                                       # g/cm^2 (1 mm steps)
sims = {"clean (trained on)": lin["mu"]}
for lv in ("physics", "mild", "moderate", "severe"):
    sims[lv] = rx.from_line_integrals(lin["soft"], lin["bone"], lv, torch.Generator().manual_seed(0), DET_PX_MM=PX * BIN)
real_b = real.reshape(NR // BIN, BIN, NC // BIN, BIN).mean((1, 3))
std = lambda a: (a - a.mean()) / (a.std() + 1e-8)
body = real_b > np.percentile(real_b, 30)
score = {k: float(np.corrcoef(std(v[0, 0].cpu().numpy())[body], std(real_b)[body])[0, 1]) for k, v in sims.items()}

fig, ax = plt.subplots(2, 3, figsize=(16, 9))
panels = [("REAL clinical X-ray", real_b)] + [(f"sim: {k}\ncorr with real {score[k]:.3f}", v[0, 0].cpu().numpy()) for k, v in sims.items()]
for a, (title, img) in zip(ax.flat, panels):
    lo, hi = np.percentile(img, (1, 99.5)); a.imshow(img, cmap="gray", vmin=lo, vmax=hi); a.set_title(title, fontsize=11); a.axis("off")
fig.suptitle(f"SPARE ClinicalVarian P1, projection {i + 1} (gantry {ang[i]:.1f}°, breathing bin 1 = max inhale) vs simulation from the "
             f"patient's prior 4D-CT bin 1 (same projection matrix; half-fan)", fontsize=11)
fig.tight_layout(); fig.savefig(Path(__file__).resolve().parent / "check" / "real_vs_sim_spare_p1.png", dpi=95)
print("projection", i + 1, "angle", round(float(ang[i]), 2), {k: round(v, 3) for k, v in score.items()})
