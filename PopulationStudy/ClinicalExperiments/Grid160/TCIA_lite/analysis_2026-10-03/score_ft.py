"""Free stabilisation checks for the big model (no training). (1) weight averaging of epochs 91-100 (SWA-style) per run;
(2) prediction averaging of seed 1 + seed 2. DIR-Lab TRE300 (10) and POPI (6), plain and mirror."""
import sys, os, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0, "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/scripts")
os.environ.setdefault("DIRLAB_ROOT", "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/data/dirlab_packs")
import eval_dir_tcia3_iso2mm_v2 as v2, eval_popi_tcia3_lps as P
from rescore_oracle_iso2mm_v3 import load_ckpt
from dirlab_tre import landmarks_300
G = "/home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Grid160"; TMP = Path(sys.argv[1]); dev = torch.device("cuda:0"); Sp = v2.SPACING
RUNS = {"old_s1": "TCIA3.5", "old_s2": "TCIA3.5_s2", "new_s1": "TCIA3.5_hybrid", "new_s2": "TCIA3.5_hybrid_s2"}
def avg(folder, eps, out):
    raws = [torch.load(f"{G}/{folder}/DecoderCRB/checkpoints/epoch_{e:03d}.pt", map_location="cpu", weights_only=False) for e in eps]; r = raws[0]
    sd = {k: (sum(x["generator"][k].float() for x in raws) / len(raws)).to(r["generator"][k].dtype) for k in r["generator"]}
    torch.save({**r, "generator": sd}, out); return out
packs = {c: v2.pack_case(c) for c in range(1, 11)}
metas = [json.loads((P.PACKED / pid / "pack_meta.json").read_text()) for pid, _, _ in P.PATIENTS]
def mir(pred, mu):
    a = pred(mu); b = pred(np.ascontiguousarray(mu[:, :, ::-1])); b = np.ascontiguousarray(b[:, :, ::-1, :]); b[..., 0] *= -1; return a, 0.5 * (a + b)
def fields(ck):
    g = load_ckpt(Path(ck), dev); P.CKPT = Path(ck); gp = P.load_generator(dev); F = {}
    for c in range(1, 11): F[("dir", c)] = mir(lambda m: v2.predict(g, dev, m, flip=False), packs[c]["mu"])
    for m in metas:
        ref = int(m["reference_phase_0idx"]); mu = np.load(P.PACKED / m["patient"] / "CT_01_mu.npy")
        F[("popi", m["patient"])] = mir(lambda x: P.predict(gp, dev, x, 0, ref), mu)
    del g, gp; torch.cuda.empty_cache(); return F
def score(F, var):
    i = 0 if var == "plain" else 1; d, p = [], []
    for c in range(1, 11):
        pk = packs[c]; l0 = v2.official_to_iso(landmarks_300(c, "T00"), pk); l5 = v2.official_to_iso(landmarks_300(c, "T50"), pk)
        d.append(float(np.linalg.norm((l0 + v2.sample_u(F[("dir", c)][i], l0) - l5) * Sp, axis=1).mean()))
    for m in metas:
        f = P.PACKED / m["patient"]; ref = int(m["reference_phase_0idx"]); src = np.load(f / "lm_00_zyx.npy"); dst = np.load(f / f"lm_{ref:02d}_zyx.npy"); n = min(len(src), len(dst))
        mv, ins = P.warp_landmarks(F[("popi", m["patient"])][i], src[:n]); p.append(float(P.tre_mm(mv[ins], dst[:n][ins]).mean()))
    return np.array(d), np.array(p)
import sys as _s
L = G + "/TCIA_lite"
CK = {"start (hybrid s1 ep40)": f"{L}/run_A2_full160_aug_hybrid/DecoderCRB/checkpoints/epoch_040.pt"}
for tag in _s.argv[2].split(","):
    for e in range(1, 6):
        p = f"{L}/run_A2_full160_aug_hybrid_ft_{tag}/DecoderCRB/checkpoints/epoch_{e:03d}.pt"
        if os.path.exists(p): CK[f"{tag} +{e}"] = p
out = {}
print(f"{'model':24s} {'DIR plain':>9s} {'DIR flip':>9s} {'POPI plain':>10s} {'POPI flip':>9s}")
for k, p in CK.items():
    F = fields(p); dp, pp = score(F, "plain"); df, pf = score(F, "mirror")
    out[k] = {"dir_plain": dp.tolist(), "dir_flip": df.tolist(), "popi_plain": pp.tolist(), "popi_flip": pf.tolist()}
    print(f"{k:24s} {dp.mean():9.3f} {df.mean():9.3f} {pp.mean():10.3f} {pf.mean():9.3f}", flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"lncc_ft_{_s.argv[2].replace(',', '_')}.json"), "w"), indent=1)
