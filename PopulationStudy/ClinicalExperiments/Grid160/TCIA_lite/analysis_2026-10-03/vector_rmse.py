"""Vector RMSE (mm) between network motion and Elastix motion on DIR-Lab, 2 mm iso cube (Elastix via v2.elastix_on_iso,
round-trip verified). Inside the lung mask: RMSE = sqrt(mean |u_net - u_el|^2), mean end-point error, and the size ratio
|u_net|/|u_el|. Also at the 300 landmarks vs the expert truth, for both network and Elastix. Epochs 96-100, plain and mirror."""
import sys, os, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0, "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/scripts")
os.environ.setdefault("DIRLAB_ROOT", "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/data/dirlab_packs")
import eval_dir_tcia3_iso2mm_v2 as v2, mirror_tta_iso2mm as mt
from rescore_oracle_iso2mm_v3 import load_ckpt
from dirlab_tre import landmarks_300
G = "/home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Grid160"; dev = torch.device("cuda:0"); S = v2.SPACING
P, EL = {}, {}
for c in range(1, 11):
    P[c] = v2.pack_case(c); EL[c] = v2.elastix_on_iso(P[c])
def stats(u, c):
    m = P[c]["lung_iso"]; d = (u - EL[c])[m] * S
    l0 = v2.official_to_iso(landmarks_300(c, "T00"), P[c]); l5 = v2.official_to_iso(landmarks_300(c, "T50"), P[c]); tru = (l5 - l0) * S
    lm = v2.sample_u(u, l0) * S - tru
    return {"rmse_vs_el": float(np.sqrt((d ** 2).sum(1).mean())), "epe_vs_el": float(np.linalg.norm(d, axis=1).mean()),
            "size_ratio": float(np.linalg.norm(u[m], axis=1).mean() / np.linalg.norm(EL[c][m], axis=1).mean()),
            "lm_rmse_vs_truth": float(np.sqrt((lm ** 2).sum(1).mean()))}
res = {}
el_lm = {}
for c in range(1, 11):
    l0 = v2.official_to_iso(landmarks_300(c, "T00"), P[c]); l5 = v2.official_to_iso(landmarks_300(c, "T50"), P[c])
    e = v2.sample_u(EL[c], l0) * S - (l5 - l0) * S; el_lm[c] = float(np.sqrt((e ** 2).sum(1).mean()))
for name, folder in (("old", "TCIA3.5"), ("new", "TCIA3.5_hybrid")):
    for e in range(96, 101):
        g = load_ckpt(Path(f"{G}/{folder}/DecoderCRB/checkpoints/epoch_{e:03d}.pt"), dev)
        for c in range(1, 11):
            a, m = mt.predict_mirror(g, dev, P[c]["mu"])
            for var, u in (("plain", a), ("mirror", m)):
                res.setdefault(f"{name}|{var}", {}).setdefault(c, []).append(stats(u, c))
        del g; torch.cuda.empty_cache()
keys = ("rmse_vs_el", "epe_vs_el", "size_ratio", "lm_rmse_vs_truth")
summ = {k: {c: {q: float(np.mean([r[q] for r in v])) for q in keys} for c, v in res[k].items()} for k in res}
print("mean over 10 cases (epochs 96-100): vector RMSE vs Elastix | EPE vs Elastix | size ratio | landmark vector RMSE vs truth")
for k, v in summ.items():
    print(f"{k:12s} " + " | ".join(f"{np.mean([x[q] for x in v.values()]):.3f}" for q in keys))
print(f"{'Elastix':12s} 0 | 0 | 1 | {np.mean(list(el_lm.values())):.3f}")
print("\nper case, new|mirror: RMSE vs Elastix / size ratio / landmark RMSE vs truth  (Elastix landmark RMSE)")
for c in range(1, 11):
    x = summ["new|mirror"][c]; print(f"  C{c:02d} {x['rmse_vs_el']:.2f} / {x['size_ratio']:.2f} / {x['lm_rmse_vs_truth']:.2f}   ({el_lm[c]:.2f})")
json.dump({"models": {k: {str(c): x for c, x in v.items()} for k, v in summ.items()}, "elastix_lm_rmse": {str(c): x for c, x in el_lm.items()}},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vector_rmse.json"), "w"), indent=1)
