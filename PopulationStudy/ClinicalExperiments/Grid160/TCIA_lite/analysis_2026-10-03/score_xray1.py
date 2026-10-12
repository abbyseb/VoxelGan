"""Per-epoch TRE watcher for X1 (CT + 1 AP DRR) and its matched control (X-ray zeroed). Same scorer as score_unsup.py:
DIR-Lab TRE300 (10, 2 mm iso v2) and POPI (6), plain + mirror, folding on DIR fields.
The conditioning X-ray is the DRR of the TARGET CT made with xray/drr_torch.py (same code as training):
  DIR-Lab: input CT_06 (T50), phases 5 -> 0, target = CT_01 (T00) resampled onto the same iso grid.
  POPI:    input CT_01, phases 0 -> ref, target = CT_{ref+1}.
X1 is also scored with a ZERO X-ray and a SHUFFLED X-ray (the next case's / patient's target DRR): if the shuffled
score is not clearly worse, the model is not using the X-ray.
  CUDA_VISIBLE_DEVICES=0 python3 score_xray1.py"""
import sys, os, json, time, importlib.util, numpy as np, torch
from pathlib import Path
import SimpleITK as sitk
sys.path.insert(0, "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/scripts")
os.environ.setdefault("DIRLAB_ROOT", "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/data/dirlab_packs")
import eval_dir_tcia3_iso2mm_v2 as v2, eval_popi_tcia3_lps as P
from dirlab_tre import landmarks_300
L = Path("/home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Grid160/TCIA_lite")
spec = importlib.util.spec_from_file_location("x1", L / "run_A2_full160_aug_xray1" / "scripts" / "train_xray1.py")
X1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(X1)
RUNS = {"xray1": L / "run_A2_full160_aug_xray1", "xray1_ctrl": L / "run_A2_full160_aug_xray1_ctrl"}
OUT = Path(__file__).resolve().parent / "xray1_tre.json"
dev = torch.device("cuda:0"); Sp = v2.SPACING; EPOCHS = 10
mm = v2.minmax

packs = {c: v2.pack_case(c) for c in range(1, 11)}
def t00(c):
    pk = packs[c]; im = sitk.ReadImage(str(pk["train"] / "CT_01.mha"))
    return v2.hu_to_mu(v2.resample_to_iso(sitk.GetArrayFromImage(im).astype(np.float32), np.array(im.GetOrigin()),
                                           np.array(im.GetSpacing()), pk["origin_iso_xyz"], order=1, cval=-1000.0))
DIR_IN = {c: packs[c]["mu"] for c in range(1, 11)}
DIR_TG = {c: t00(c) for c in range(1, 11)}
metas = [json.loads((P.PACKED / pid / "pack_meta.json").read_text()) for pid, _, _ in P.PATIENTS]
PO_IN = {m["patient"]: np.load(P.PACKED / m["patient"] / "CT_01_mu.npy") for m in metas}
PO_TG = {m["patient"]: np.load(P.PACKED / m["patient"] / f"CT_{int(m['reference_phase_0idx']) + 1:02d}_mu.npy") for m in metas}


def xr(tgt_np):
    return X1.xray_channel(torch.from_numpy(mm(tgt_np)[None, None]).to(dev))


def pred(g, mu, tgt, rp, tp, mode):
    x = torch.from_numpy(mm(mu)[None, None]).to(dev)
    xx = torch.zeros(1, 1, *mu.shape, device=dev) if mode == "zero" else xr(tgt)
    with torch.no_grad():
        d = g(x, torch.tensor([rp], device=dev), torch.tensor([tp], device=dev), xx)[0].cpu().numpy()
    return np.moveaxis(d, 0, -1).astype(np.float32)


def mir(g, mu, tgt, rp, tp, mode):
    a = pred(g, mu, tgt, rp, tp, mode)
    f = lambda v: np.ascontiguousarray(v[:, :, ::-1])
    b = pred(g, f(mu), f(tgt), rp, tp, mode); b = np.ascontiguousarray(b[:, :, ::-1, :]); b[..., 0] *= -1
    return a, 0.5 * (a + b)


def fold_pct(u):
    gr = [np.gradient(u[..., c], axis=(2, 1, 0)) for c in range(3)]
    J = np.stack([np.stack(r, -1) for r in gr], -2) + np.eye(3)
    return float((np.linalg.det(J) <= 0).mean() * 100)


def load(ck):
    g = X1.XrayUNet(n_xray=1, im_size=160, n_phases=10)
    g.load_state_dict(torch.load(str(ck), map_location="cpu", weights_only=False)["generator"]); return g.to(dev).eval()


def score_mode(g, mode):
    cs = list(range(1, 11)); ps = [m["patient"] for m in metas]
    shd = lambda c: cs[cs.index(c) % 10 - 9] if mode == "shuffled" else c          # next case (wraps)
    shp = lambda p: ps[(ps.index(p) + 1) % len(ps)] if mode == "shuffled" else p
    F = {("dir", c): mir(g, DIR_IN[c], DIR_TG[shd(c)], 5, 0, mode) for c in cs}
    for m in metas:
        F[("popi", m["patient"])] = mir(g, PO_IN[m["patient"]], PO_TG[shp(m["patient"])], 0, int(m["reference_phase_0idx"]), mode)
    r = {}
    for i, var in ((0, "plain"), (1, "flip")):
        d, p = [], []
        for c in cs:
            pk = packs[c]; l0 = v2.official_to_iso(landmarks_300(c, "T00"), pk); l5 = v2.official_to_iso(landmarks_300(c, "T50"), pk)
            d.append(float(np.linalg.norm((l0 + v2.sample_u(F[("dir", c)][i], l0) - l5) * Sp, axis=1).mean()))
        for m in metas:
            f = P.PACKED / m["patient"]; ref = int(m["reference_phase_0idx"]); src = np.load(f / "lm_00_zyx.npy"); dst = np.load(f / f"lm_{ref:02d}_zyx.npy"); n = min(len(src), len(dst))
            mv, ins = P.warp_landmarks(F[("popi", m["patient"])][i], src[:n]); p.append(float(P.tre_mm(mv[ins], dst[:n][ins]).mean()))
        r[f"dir_{var}"], r[f"popi_{var}"] = d, p
    r["dir_fold_pct"] = [fold_pct(F[("dir", c)][0]) for c in cs]
    return r


out = json.loads(OUT.read_text()) if OUT.exists() else {k: {} for k in RUNS}
print(f"{'run':11s} {'ep':>3s} {'mode':8s} {'DIR plain':>9s} {'DIR flip':>9s} {'POPI plain':>10s} {'POPI flip':>9s} {'fold%':>7s}", flush=True)
for name, run in RUNS.items():
    for e in range(1, EPOCHS + 1):
        if str(e) in out[name]:
            continue
        ck = run / "DecoderCRB" / "checkpoints" / f"epoch_{e:03d}.pt"
        while not ck.exists():
            time.sleep(120)
        time.sleep(30)
        g = load(ck); out[name][str(e)] = {}
        for mode in (("real", "zero", "shuffled") if name == "xray1" else ("zero",)):
            r = score_mode(g, mode); out[name][str(e)][mode] = r
            OUT.write_text(json.dumps(out, indent=1))
            print(f"{name:11s} {e:3d} {mode:8s} {np.mean(r['dir_plain']):9.3f} {np.mean(r['dir_flip']):9.3f} {np.mean(r['popi_plain']):10.3f} "
                  f"{np.mean(r['popi_flip']):9.3f} {np.mean(r['dir_fold_pct']):7.4f}", flush=True)
        del g; torch.cuda.empty_cache()
