"""Average the motion fields of X3S (final) and X4 (latest checkpoint) on the same realistic X-rays (same seeds); DIR-Lab TRE300
plain and with mirror averaging. Also weighted mixes.  python3 ensemble_x3s_x4.py"""
import json, sys
from pathlib import Path
import numpy as np
A = Path(__file__).resolve().parent
ns3, ns4 = {}, {}
for ns, f in ((ns3, "score_xray3s.py"), (ns4, "score_x4.py")):
    src = (A / f).read_text(); src = src[:src.index("out = json.loads")]
    ns["__file__"] = str(A / f); exec(src, ns)
L = ns4["L"]
g3 = ns3["load"](L / "run_A2_full160_aug_xray3s/DecoderCRB/checkpoints/epoch_040.pt")
c4 = sorted((L / "run_A2_full160_aug_x4/DecoderCRB/checkpoints").glob("epoch_0*.pt"))[-1]; g4 = ns4["load"](c4)
v2, Sp, lm = ns4["v2"], ns4["Sp"], ns4["landmarks_300"]
W = [0.0, 0.3, 0.5, 0.7, 1.0]                                   # weight on X4
res = {f"w{w}": {"plain": [], "mirror": []} for w in W}
for c in range(1, 11):
    a3 = ns3["mir"](g3, ns3["DIR_IN"][c], ns3["DIR_TG"][c], 5, 0, "clinic")
    a4 = ns4["mir"](g4, ns4["DIR_IN"][c], ns4["DIR_TG"][c], 5, 0, "clinic")
    pk = ns4["packs"][c]; l0 = v2.official_to_iso(lm(c, "T00"), pk); l5 = v2.official_to_iso(lm(c, "T50"), pk)
    for w in W:
        for i, k in ((0, "plain"), (1, "mirror")):
            u = w * a4[i] + (1 - w) * a3[i]
            res[f"w{w}"][k].append(float(np.linalg.norm((l0 + v2.sample_u(u, l0) - l5) * Sp, axis=1).mean()))
    print(f"C{c:02d} " + " ".join(f"w{w}:{res[f'w{w}']['plain'][-1]:.2f}" for w in W), flush=True)
summ = {k: {m: float(np.mean(v[m])) for m in v} for k, v in res.items()}
(A / "ensemble_x3s_x4.json").write_text(json.dumps({"x4_ckpt": c4.name, "summary": summ, "cases": res}, indent=1))
print(c4.name, json.dumps(summ, indent=1))
