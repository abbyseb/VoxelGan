"""Why did adding the 4 held-out patients (104, 107, 110, 115; 17 scans) make the all-82 run worse?
Per scan: breathing size and folding (metrics_per_scan.tsv); label quality = inverse consistency of the Elastix pair
06->01 and 01->06 (|u_fwd(x) + u_bwd(x + u_fwd(x))| in lung, mm); label-image fit = |CT_01 - warp(CT_06, label)| /
|CT_01 - CT_06| in lung (lower = label explains the images better); image stats (lung volume, lung and tissue mu).
Compared against the 65 training scans; flags |z| > 2."""
import json, os, numpy as np, torch, torch.nn.functional as F
TR = "/media/abhishek/3CCA3CADCA3C6574/TCIA_4D-Lung/synth_g160_r3/pooled/train"
CH = "/home/abhishek/Voxel_GAN/TCIA_4D-Lung_dvf_characteristics"
dev = torch.device("cuda:0")
pmap = {l.split("\t")[0]: l.split("\t")[1] for l in open(f"{CH}/scan_patient_map.tsv").read().splitlines()[1:]}
met = {}
lines = open(f"{CH}/metrics_per_scan.tsv").read().splitlines(); head = lines[0].split("\t")
for l in lines[1:]:
    r = dict(zip(head, l.split("\t"))); met[r["scan"]] = r
HOLD = {"104", "107", "110", "115"}
def warp(img, u):  # img (1,1,Z,Y,X), u (1,3,Z,Y,X) dx,dy,dz voxels
    _, _, Z, Y, X = img.shape; zz, yy, xx = torch.meshgrid(*(torch.arange(n, device=dev, dtype=torch.float32) for n in (Z, Y, X)), indexing="ij")
    g = torch.stack([2 * (xx + u[:, 0]) / (X - 1) - 1, 2 * (yy + u[:, 1]) / (Y - 1) - 1, 2 * (zz + u[:, 2]) / (Z - 1) - 1], -1)
    return F.grid_sample(img, g, mode="bilinear", padding_mode="border", align_corners=True)
rows = []
for s, pid in pmap.items():
    n = int(s[1:]); S = f"S{n:02d}"
    try:
        fwd = torch.from_numpy(np.load(f"{TR}/{S}_06_to_01_pair.npy")).permute(3, 0, 1, 2)[None].float().to(dev)
        bwd = torch.from_numpy(np.load(f"{TR}/{S}_01_to_06_pair.npy")).permute(3, 0, 1, 2)[None].float().to(dev)
        c1 = torch.from_numpy(np.load(f"{TR}/{S}_CT_01.npy").squeeze()).float()[None, None].to(dev)
        c6 = torch.from_numpy(np.load(f"{TR}/{S}_CT_06.npy").squeeze()).float()[None, None].to(dev)
        m = torch.from_numpy(np.load(f"{TR}/{S}_Mask_Lung.npy")).float()[None, None].to(dev) > 0.5
    except FileNotFoundError:
        continue
    with torch.no_grad():
        ice = (fwd + warp(bwd, fwd)).norm(dim=1, keepdim=True)[m].mean().item() * 2.0  # mm
        fit = ((c1 - warp(c6, fwd)).abs()[m].mean() / (c1 - c6).abs()[m].mean().clamp_min(1e-9)).item()
        tissue = (c6 > 0.25)  # rough soft tissue in min-max mu
        rows.append({"scan": S, "patient": pid, "hold": pid in HOLD, "motion_mm": float(met[s]["mag_01_to_06_mm"]),
                     "max_pair": met[s]["max_pair"], "fold": float(met[s]["fold_frac_mean"]), "ice_mm": ice, "fit": fit,
                     "lung_L": m.sum().item() * 8e-6, "lung_mu": c6[m].mean().item(), "tissue_mu": c6[tissue].mean().item()})
keys = ["motion_mm", "fold", "ice_mm", "fit", "lung_L", "lung_mu", "tissue_mu"]
tr = [r for r in rows if not r["hold"]]; ho = [r for r in rows if r["hold"]]
mu = {k: np.mean([r[k] for r in tr]) for k in keys}; sd = {k: np.std([r[k] for r in tr]) for k in keys}
print(f"{'':10s}" + "".join(f"{k:>11s}" for k in keys))
print(f"{'train 65':10s}" + "".join(f"{mu[k]:11.4f}" for k in keys))
print(f"{'hold 17':10s}" + "".join(f"{np.mean([r[k] for r in ho]):11.4f}" for k in keys))
for p in sorted(HOLD):
    rs = [r for r in ho if r["patient"] == p]
    print(f"{'P'+p+f' ({len(rs)})':10s}" + "".join(f"{np.mean([r[k] for r in rs]):11.4f}" for k in keys))
print("\nflags |z|>2 vs the 65 training scans:")
for r in ho:
    f = [f"{k} z={(r[k]-mu[k])/sd[k]:+.1f}" for k in keys if sd[k] > 0 and abs(r[k] - mu[k]) / sd[k] > 2]
    if f: print(f"  {r['scan']} (P{r['patient']}, max pair {r['max_pair']}): " + ", ".join(f))
ntr = sum(1 for r in tr if any(sd[k] > 0 and abs(r[k] - mu[k]) / sd[k] > 2 for k in keys))
print(f"(for scale: {ntr} of {len(tr)} training scans also have at least one |z|>2)")
json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "holdout_patients_check.json"), "w"), indent=1)
