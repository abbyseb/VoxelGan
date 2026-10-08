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
L = G + "/TCIA_lite"
RUNS = {"hybrid 65 scans (s2)": "run_A2_full160_aug_hybrid_s2", "hybrid all 82 (s2)": "run_A2_full160_aug_hybrid_s2_all82",
        "hybrid 78 clean (s2)": "run_A2_full160_aug_hybrid_s2_all82_clean", "old 65 scans": "run_A2_full160_aug", "old all 82 (s2 seed)": "run_A2_full160_aug_all82_old"}
out = {}
print(f"{'run (ep 36-40 mean)':26s} {'DIR':>6s} {'DIR+flip':>8s} {'POPI':>6s} {'POPI+flip':>9s} {'C06':>5s} {'C08':>5s}")
for n, r in RUNS.items():
    acc = {k: [] for k in ("dp", "df", "pp", "pf")}
    for e in range(36, 41):
        F = fields(f"{L}/{r}/DecoderCRB/checkpoints/epoch_{e:03d}.pt"); dp, pp = score(F, "plain"); df, pf = score(F, "mirror")
        for k, v in zip(("dp", "df", "pp", "pf"), (dp, df, pp, pf)): acc[k].append(v)
    m = {k: np.mean(v, 0) for k, v in acc.items()}; out[n] = {k: v.tolist() for k, v in m.items()}
    print(f"{n:26s} {m['dp'].mean():6.3f} {m['df'].mean():8.3f} {m['pp'].mean():6.3f} {m['pf'].mean():9.3f} {m['dp'][5]:5.2f} {m['dp'][7]:5.2f}", flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "all82_clean_old.json"), "w"), indent=1)
