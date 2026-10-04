"""Score checkpoints on POPI (6 patients, phase 00 -> published reference) with the verified LPS scorer
(DIR EXPERIMENTS/scripts/eval_popi_tcia3_lps.py, packed cubes reused via its score_pair). Adds mirror TTA."""
import sys, json, os, numpy as np, torch
from pathlib import Path
sys.path.insert(0, "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/scripts")
import eval_popi_tcia3_lps as P

G = "/home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Grid160"
L = G + "/TCIA_lite"
M = {}
for e in range(96, 101):
    M[f"T35hyb@{e}"] = f"{G}/TCIA3.5_hybrid/DecoderCRB/checkpoints/epoch_{e:03d}.pt"
    M[f"T35@{e}"] = f"{G}/TCIA3.5/DecoderCRB/checkpoints/epoch_{e:03d}.pt"
for e in range(36, 41):
    M[f"base@{e}"] = f"{L}/run_A2_full160_aug/DecoderCRB/checkpoints/epoch_{e:03d}.pt"
    M[f"s1@{e}"] = f"{L}/run_A2_full160_aug_hybrid/DecoderCRB/checkpoints/epoch_{e:03d}.pt"
    M[f"s2@{e}"] = f"{L}/run_A2_full160_aug_hybrid_s2/DecoderCRB/checkpoints/epoch_{e:03d}.pt"
plain = P.predict
def mirror(g, device, mu, ref, tgt):
    a = plain(g, device, mu, ref, tgt)
    b = plain(g, device, np.ascontiguousarray(mu[:, :, ::-1]), ref, tgt)
    b = np.ascontiguousarray(b[:, :, ::-1, :]); b[..., 0] *= -1.0
    return 0.5 * (a + b)
dev = torch.device("cuda:0")
metas = [json.loads((P.PACKED / pid / "pack_meta.json").read_text()) for pid, _, _ in P.PATIENTS]
out = {}
for name, path in M.items():
    P.CKPT = Path(path); g = P.load_generator(dev)
    for var, fn in (("plain", plain), ("mirror", mirror)):
        P.predict = fn
        out[f"{name}|{var}"] = {m["patient"]: P.score_pair(g, dev, m["patient"], 0, int(m["reference_phase_0idx"]), False)["tcia3"]["mean_mm"] for m in metas}
    P.predict = plain
    print(name, " ".join(f"{v}:{np.mean(list(out[f'{name}|{v}'].values())):.3f}" for v in ("plain", "mirror")), flush=True)
    del g; torch.cuda.empty_cache()
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "popi_scores.json"), "w"), indent=1)
