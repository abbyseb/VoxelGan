"""Average fields of finished models on the same realistic X-rays (seeds as the scorers): X4 (ep40), X5a (ep40), X3S (ep40).
Equal weights fixed in advance (no tuning on DIR-Lab). DIR-Lab TRE300 plain + mirror.  python3 ensemble_final.py"""
import json
from pathlib import Path
import numpy as np, itertools
A = Path(__file__).resolve().parent
NS = {}
for name, f, ck in (("X4", "score_x4.py", "run_A2_full160_aug_x4"), ("X5a", "score_x5a.py", "run_A2_full160_aug_x5a"), ("X3S", "score_xray3s.py", "run_A2_full160_aug_xray3s")):
    src = (A / f).read_text(); src = src[:src.index("out = json.loads")]
    ns = {"__file__": str(A / f)}; exec(src, ns); ns["g"] = ns["load"](ns["L"] / ck / "DecoderCRB/checkpoints/epoch_040.pt"); NS[name] = ns
v2, lm, packs = NS["X4"]["v2"], NS["X4"]["landmarks_300"], NS["X4"]["packs"]
F = {n: {} for n in NS}
for c in range(1, 11):
    for n, ns in NS.items():
        F[n][c] = ns["mir"](ns["g"], ns["DIR_IN"][c], ns["DIR_TG"][c], 5, 0, "clinic")
def tre(u, c):
    pk = packs[c]; l0 = v2.official_to_iso(lm(c, "T00"), pk); l5 = v2.official_to_iso(lm(c, "T50"), pk)
    return float(np.linalg.norm((l0 + v2.sample_u(u, l0) - l5) * v2.SPACING, axis=1).mean())
res = {}
for k in range(1, 4):
    for combo in itertools.combinations(NS, k):
        for i, var in ((0, "plain"), (1, "mirror")):
            v = [tre(np.mean([F[n][c][i] for n in combo], 0), c) for c in range(1, 11)]
            res[f"{'+'.join(combo)}/{var}"] = {"mean": float(np.mean(v)), "cases": [round(x, 2) for x in v]}
for k, v in sorted(res.items(), key=lambda kv: kv[1]["mean"]): print(f"{k:22s} {v['mean']:.3f}  {v['cases']}")
(A / "ensemble_final.json").write_text(json.dumps(res, indent=1))
