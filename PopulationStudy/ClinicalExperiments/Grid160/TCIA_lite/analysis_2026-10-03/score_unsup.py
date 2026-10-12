"""Per-epoch TRE watcher for run_A2_full160_aug_unsup_lncc (LNCC only, no labels). Same scorer as score_ft.py:
DIR-Lab TRE300 (10 cases, 2 mm iso v2 scorer) and POPI (6), plain and mirror. Also folding on the DIR-Lab fields.
Reference: the hybrid run (Elastix labels) scored at the same epochs. Waits for each new epoch_NNN.pt, scores it,
rewrites unsup_lncc_tre.json, stops after epoch 40.
  CUDA_VISIBLE_DEVICES=0 python3 score_unsup.py"""
import sys, os, json, time, numpy as np, torch
from pathlib import Path
sys.path.insert(0, "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/scripts")
os.environ.setdefault("DIRLAB_ROOT", "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/data/dirlab_packs")
import eval_dir_tcia3_iso2mm_v2 as v2, eval_popi_tcia3_lps as P
from rescore_oracle_iso2mm_v3 import load_ckpt
from dirlab_tre import landmarks_300
L = Path("/home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Grid160/TCIA_lite")
RUNS = {"unsup_lncc": L / "run_A2_full160_aug_unsup_lncc", "hybrid": L / "run_A2_full160_aug_hybrid"}
OUT = Path(__file__).resolve().parent / "unsup_lncc_tre.json"
dev = torch.device("cuda:0"); Sp = v2.SPACING; EPOCHS = 40
packs = {c: v2.pack_case(c) for c in range(1, 11)}
metas = [json.loads((P.PACKED / pid / "pack_meta.json").read_text()) for pid, _, _ in P.PATIENTS]


def mir(pred, mu):
    a = pred(mu); b = pred(np.ascontiguousarray(mu[:, :, ::-1])); b = np.ascontiguousarray(b[:, :, ::-1, :]); b[..., 0] *= -1; return a, 0.5 * (a + b)


def fold_pct(u):
    """% voxels with det(I + grad u) <= 0; u is (Z,Y,X,3) with channels (x,y,z) in voxels."""
    g = [np.gradient(u[..., c], axis=(2, 1, 0)) for c in range(3)]
    J = np.stack([np.stack(r, -1) for r in g], -2) + np.eye(3)
    return float((np.linalg.det(J) <= 0).mean() * 100)


def fields(ck):
    g = load_ckpt(Path(ck), dev); P.CKPT = Path(ck); gp = P.load_generator(dev); F = {}
    for c in range(1, 11): F[("dir", c)] = mir(lambda m: v2.predict(g, dev, m, flip=False), packs[c]["mu"])
    for m in metas:
        ref = int(m["reference_phase_0idx"]); mu = np.load(P.PACKED / m["patient"] / "CT_01_mu.npy")
        F[("popi", m["patient"])] = mir(lambda x: P.predict(gp, dev, x, 0, ref), mu)
    del g, gp; torch.cuda.empty_cache(); return F


def score(F, i):
    d, p = [], []
    for c in range(1, 11):
        pk = packs[c]; l0 = v2.official_to_iso(landmarks_300(c, "T00"), pk); l5 = v2.official_to_iso(landmarks_300(c, "T50"), pk)
        d.append(float(np.linalg.norm((l0 + v2.sample_u(F[("dir", c)][i], l0) - l5) * Sp, axis=1).mean()))
    for m in metas:
        f = P.PACKED / m["patient"]; ref = int(m["reference_phase_0idx"]); src = np.load(f / "lm_00_zyx.npy"); dst = np.load(f / f"lm_{ref:02d}_zyx.npy"); n = min(len(src), len(dst))
        mv, ins = P.warp_landmarks(F[("popi", m["patient"])][i], src[:n]); p.append(float(P.tre_mm(mv[ins], dst[:n][ins]).mean()))
    return d, p


def score_ckpt(ck):
    F = fields(ck); dp, pp = score(F, 0); df, pf = score(F, 1)
    return {"dir_plain": dp, "dir_flip": df, "popi_plain": pp, "popi_flip": pf,
            "dir_fold_pct": [fold_pct(F[("dir", c)][0]) for c in range(1, 11)]}


out = json.loads(OUT.read_text()) if OUT.exists() else {k: {} for k in RUNS}
print(f"{'run':11s} {'ep':>3s} {'DIR plain':>9s} {'DIR flip':>9s} {'POPI plain':>10s} {'POPI flip':>9s} {'DIR fold%':>9s}", flush=True)
for e in range(1, EPOCHS + 1):
    for name, run in RUNS.items():
        ck = run / "DecoderCRB" / "checkpoints" / f"epoch_{e:03d}.pt"
        if str(e) in out[name]:
            continue
        while not ck.exists():
            time.sleep(120)
        time.sleep(30)  # let torch.save finish
        r = score_ckpt(ck); out[name][str(e)] = r
        OUT.write_text(json.dumps(out, indent=1))
        print(f"{name:11s} {e:3d} {np.mean(r['dir_plain']):9.3f} {np.mean(r['dir_flip']):9.3f} {np.mean(r['popi_plain']):10.3f} "
              f"{np.mean(r['popi_flip']):9.3f} {np.mean(r['dir_fold_pct']):9.4f}", flush=True)
