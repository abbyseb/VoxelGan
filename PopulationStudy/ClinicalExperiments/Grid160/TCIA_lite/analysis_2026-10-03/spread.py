"""Mean ± SD for the main models: SD across landmarks (pooled, all landmarks) and SD across patients.
DIR-Lab TRE300 (verified 2 mm scorer) and POPI (LPS scorer). Epochs 96-100, plain and mirror; per epoch then averaged."""
import sys, os, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0, "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/scripts")
os.environ.setdefault("DIRLAB_ROOT", "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/data/dirlab_packs")
import eval_dir_tcia3_iso2mm_v2 as v2, mirror_tta_iso2mm as mt, eval_popi_tcia3_lps as P
from rescore_oracle_iso2mm_v3 import load_ckpt
from dirlab_tre import landmarks_300
G = "/home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Grid160"
dev = torch.device("cuda:0"); packs = {c: v2.pack_case(c) for c in range(1, 11)}
metas = [json.loads((P.PACKED / pid / "pack_meta.json").read_text()) for pid, _, _ in P.PATIENTS]
def dir_errs(u, c):
    pk = packs[c]; l0 = v2.official_to_iso(landmarks_300(c, "T00"), pk); l5 = v2.official_to_iso(landmarks_300(c, "T50"), pk)
    return np.linalg.norm((l0 + v2.sample_u(u, l0) - l5) * v2.SPACING, axis=1)
def popi_errs(g, pid, ref, flip):
    f = P.PACKED / pid; mu = np.load(f / "CT_01_mu.npy"); src = np.load(f / "lm_00_zyx.npy"); dst = np.load(f / f"lm_{ref:02d}_zyx.npy")
    n = min(len(src), len(dst)); src, dst = src[:n], dst[:n]
    a = P.predict(g, dev, mu, 0, ref)
    if flip:
        b = P.predict(g, dev, np.ascontiguousarray(mu[:, :, ::-1]), 0, ref); b = np.ascontiguousarray(b[:, :, ::-1, :]); b[..., 0] *= -1; a = 0.5 * (a + b)
    moved, inside = P.warp_landmarks(a, src); return P.tre_mm(moved[inside], dst[inside])
res = {}
for model, folder in (("old", "TCIA3.5"), ("new", "TCIA3.5_hybrid")):
    for e in range(96, 101):
        path = Path(f"{G}/{folder}/DecoderCRB/checkpoints/epoch_{e:03d}.pt"); g = load_ckpt(path, dev)
        for flip in (False, True):
            key = f"{model}|{'flip' if flip else 'plain'}"; R = res.setdefault(key, {"dir": [], "popi": []})
            de = {}
            for c in range(1, 11):
                a, m = mt.predict_mirror(g, dev, packs[c]["mu"]); de[c] = dir_errs(m if flip else a, c)
            P.CKPT = path; gp = P.load_generator(dev)
            pe = {m_["patient"]: popi_errs(gp, m_["patient"], int(m_["reference_phase_0idx"]), flip) for m_ in metas}
            R["dir"].append(de); R["popi"].append(pe); del gp
        del g; torch.cuda.empty_cache()
def summ(eps):
    per_pat = np.array([[e.mean() for e in ep.values()] for ep in eps]).mean(0)
    lm_sd = np.mean([np.concatenate(list(ep.values())).std() for ep in eps])
    lm_mean = np.mean([np.concatenate(list(ep.values())).mean() for ep in eps])
    return per_pat.mean(), per_pat.std(ddof=1), lm_mean, lm_sd
out = {}
for key, R in res.items():
    d, p = summ(R["dir"]), summ(R["popi"])
    comb = [{**a, **{f"P{k}": v for k, v in b.items()}} for a, b in zip(R["dir"], R["popi"])]
    c = summ(comb); out[key] = {"dir": d, "popi": p, "both": c}
    print(f"{key:10s} DIR {d[0]:.2f}±{d[1]:.2f} (pat) | {d[2]:.2f}±{d[3]:.2f} (lm)   POPI {p[0]:.2f}±{p[1]:.2f} (pat) | {p[2]:.2f}±{p[3]:.2f} (lm)   BOTH {c[0]:.2f}±{c[1]:.2f} (pat) | {c[2]:.2f}±{c[3]:.2f} (lm)", flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "spread.json"), "w"), indent=1)
