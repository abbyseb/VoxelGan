"""Probe: does local NCC see a misalignment that the current L1 image term does not?
Held-out TCIA-lite val pairs (4 patients never trained on), hybrid seed 2 ep40 (small model, whole 160^3).
For each pair: fields zero / Elastix label / prediction; image terms L1 (as in training, lung dilated 2) and 1-LNCC (9^3 window).
'room' = (loss(pred) - loss(label)) / (loss(zero) - loss(label)): share of the zero->label gap the prediction still has to close.
Also: per-pair correlation of each term's (pred - label) gap with the DVF error |pred - label| in the lung."""
import sys, runpy, json, os, numpy as np, torch, torch.nn.functional as F
RUN = "/home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Grid160/TCIA_lite/run_A2_full160_aug_hybrid_s2"
sys.argv = ["x"]; m = runpy.run_path(f"{RUN}/scripts/train_aug_hybrid_s2.py", run_name="probe")
dev = torch.device("cuda:0")
man = m["load_manifest"](); pairs = [p for p in man["val_pairs"] if p.split("_")[1] != p.split("_")[3]]
pick = pairs[:: max(1, len(pairs) // 50)][:50]
ds = m["FastMmapPhasePairDataset"](pair_files=pick, im_dir=man["pooled_train_dir"], im_size=160, random_crop=False, patches_per_pair=1, fov_aug=False)
g = m["UNetCRBDecoder"](im_size=160, n_phases=10).to(dev)
ck = torch.load(f"{RUN}/DecoderCRB/checkpoints/epoch_040.pt", map_location=dev, weights_only=False)
g.load_state_dict(ck["generator"] if "generator" in ck else ck); g.eval()
warp, dilate = m["warp"], m["dilate"]
def l1(ref, tgt, u, msk): w = warp(ref, u); return float(((tgt - w).abs() * msk).sum() / msk.sum())
def lncc(ref, tgt, u, msk, k=9):
    w = warp(ref, u); p = k // 2; ones = torch.ones(1, 1, k, k, k, device=dev) / k**3
    mu = lambda x: F.conv3d(x, ones, padding=p)
    a, b = mu(w), mu(tgt); va = mu(w * w) - a * a; vb = mu(tgt * tgt) - b * b; cov = mu(w * tgt) - a * b
    cc = cov / torch.sqrt(va.clamp_min(1e-8) * vb.clamp_min(1e-8))
    return float(1 - (cc * msk).sum() / msk.sum())
rows = []
with torch.no_grad():
    for i in range(len(ds)):
        d = ds[i]; ref, tgt, mask, lab = (d[k][None].to(dev) for k in ("reference_ct", "target_ct", "lung_mask", "target_dvf"))
        pred = g(ref, d["ref_phase"][None].to(dev), d["target_phase"][None].to(dev)); msk = dilate(mask); zero = torch.zeros_like(lab)
        lm = mask > 0.5; dvf_err = float((pred - lab).abs().sum(1, keepdim=True)[lm].mean())
        r = {"pair": pick[i], "dvf_err": dvf_err}
        for name, fn in (("l1", l1), ("lncc", lncc)):
            for fld, u in (("zero", zero), ("label", lab), ("pred", pred)): r[f"{name}_{fld}"] = fn(ref, tgt, u, msk)
        rows.append(r)
def room(n): return np.array([(r[f"{n}_pred"] - r[f"{n}_label"]) / max(r[f"{n}_zero"] - r[f"{n}_label"], 1e-9) for r in rows])
err = np.array([r["dvf_err"] for r in rows])
for n in ("l1", "lncc"):
    gap = np.array([r[f"{n}_pred"] - r[f"{n}_label"] for r in rows]); rm = room(n)
    print(f"{n:5s}: pred worse than label on {np.mean(gap > 0)*100:.0f}% of pairs | room left median {np.median(rm)*100:+.1f}% "
          f"(IQR {np.percentile(rm,25)*100:+.1f} to {np.percentile(rm,75)*100:+.1f}) | corr(gap, motion error) {np.corrcoef(gap, err)[0,1]:+.2f}")
print(f"pairs {len(rows)}, motion error |pred-label| median {np.median(err):.3f} vox")
json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lncc_probe.json"), "w"), indent=1)
