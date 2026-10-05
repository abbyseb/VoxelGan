"""Step 0 of the 2-CT idea: no training. Elastix registration between the two extremes (T00 inhale <-> T50 exhale, A1
DVF_sub_01 on the verified 2 mm cube) scaled by phase: landmark at Tk = lm00 + a_k * u(lm00), a_k = k/5 for T10..T40.
Scored with the 75-point DIR-Lab landmarks of each intermediate phase (Fuerst et al. scored intermediate exhale phases on
cases 6-10). Also: identity, and an oracle best single a per case and phase (ceiling, uses the answer)."""
import sys, os, json, numpy as np
sys.path.insert(0, "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/scripts")
os.environ.setdefault("DIRLAB_ROOT", "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/data/dirlab_packs")
import eval_dir_tcia3_iso2mm_v2 as v2
from dirlab_tre import landmarks_75
S = v2.SPACING; out = {}
for c in range(1, 11):
    pk = v2.pack_case(c); u = v2.elastix_on_iso(pk)
    l0 = v2.official_to_iso(landmarks_75(c, "T00"), pk); d0 = v2.sample_u(u, l0)
    row = {}
    for k in (1, 2, 3, 4, 5):
        lk = v2.official_to_iso(landmarks_75(c, f"T{k}0"), pk)
        t = lambda a: float(np.linalg.norm((l0 + a * d0 - lk) * S, axis=1).mean())
        best = min(np.linspace(0, 1.5, 151), key=t)
        row[f"T{k}0"] = {"identity": t(0.0), "linear": t(k / 5), "oracle_a": float(best), "oracle": t(best)}
    out[c] = row
    print(f"C{c:02d} " + " ".join(f"T{k}0 lin {row[f'T{k}0']['linear']:.2f} (id {row[f'T{k}0']['identity']:.2f}, best a={row[f'T{k}0']['oracle_a']:.2f})" for k in (1, 2, 3, 4)) + f" | T50 {row['T50']['linear']:.2f}", flush=True)
def m(cases, key): return np.mean([out[c][f"T{k}0"][key] for c in cases for k in (1, 2, 3, 4)])
for name, cs in (("all 10", range(1, 11)), ("cases 6-10", range(6, 11))):
    print(f"{name}: intermediate T10-T40 mean TRE75  identity {m(cs,'identity'):.2f} | linear-scaled Elastix {m(cs,'linear'):.2f} | oracle a {m(cs,'oracle'):.2f}")
print("cases 6-10 per case linear:", {c: round(np.mean([out[c][f'T{k}0']['linear'] for k in (1,2,3,4)]), 2) for c in range(6, 11)})
json.dump({str(c): r for c, r in out.items()}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "twoct_baseline.json"), "w"), indent=1)
