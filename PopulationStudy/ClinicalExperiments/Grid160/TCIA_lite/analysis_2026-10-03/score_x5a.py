"""Per-epoch TRE watcher for X5 (= X4 + breath-hold data + two-way stretch + bending); as for X4 (X3S + X-ray cleaner front-end + deep-breath practice). As score_xray3s.py, but every
target X-ray (perfect or realistic) AND the planning-CT X-ray pass through the cleaner, exactly as in train_x4.py.
Original: Per-epoch TRE watcher for X3S (front + side X-rays in separate channels, realistic). Extra diagnostic modes front_only /
side_only blank the other view's channels (never seen in training, so read them only as "how much does each view carry").
Base modes as X2R: Modes: clean (perfect DRR), clinic (one clinic-like
realistic X-ray per case, fixed seed, from the target HU via realistic_xray.clinic_projection), zero, shuffled (next case's clinic X-ray).
Inputs at test = xray_inputs(input CT, target CT) from train_xray2.py (both DRRs with the same code as training). Same scorer as score_unsup.py:
DIR-Lab TRE300 (10, 2 mm iso v2) and POPI (6), plain + mirror, folding on DIR fields.
The conditioning X-ray is the DRR of the TARGET CT made with xray/drr_torch.py (same code as training):
  DIR-Lab: input CT_06 (T50), phases 5 -> 0, target = CT_01 (T00) resampled onto the same iso grid.
  POPI:    input CT_01, phases 0 -> ref, target = CT_{ref+1}.
X1 is also scored with a ZERO X-ray and a SHUFFLED X-ray (the next case's / patient's target DRR): if the shuffled
score is not clearly worse, the model is not using the X-ray.
  CUDA_VISIBLE_DEVICES=0 python3 score_x5.py"""
import sys, os, json, time, importlib.util, numpy as np, torch
from pathlib import Path
import SimpleITK as sitk
sys.path.insert(0, "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/scripts")
os.environ.setdefault("DIRLAB_ROOT", "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/data/dirlab_packs")
import eval_dir_tcia3_iso2mm_v2 as v2, eval_popi_tcia3_lps as P
from dirlab_tre import landmarks_300
L = Path("/home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Grid160/TCIA_lite")
spec = importlib.util.spec_from_file_location("x5a", L / "run_A2_full160_aug_x5a" / "scripts" / "train_x5a.py")
X1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(X1)
RUNS = {"x5a": L / "run_A2_full160_aug_x5a"}
OUT = Path(__file__).resolve().parent / "x5a_tre.json"
dev = torch.device("cuda:0"); Sp = v2.SPACING; EPOCHS = 40
mm = v2.minmax

packs = {c: v2.pack_case(c) for c in range(1, 11)}
def t00(c):
    pk = packs[c]; im = sitk.ReadImage(str(pk["train"] / "CT_01.mha"))
    return v2.hu_to_mu(v2.resample_to_iso(sitk.GetArrayFromImage(im).astype(np.float32), np.array(im.GetOrigin()),
                                           np.array(im.GetSpacing()), pk["origin_iso_xyz"], order=1, cval=-1000.0))
DIR_IN = {c: packs[c]["mu"] for c in range(1, 11)}
DIR_TG = {c: t00(c) for c in range(1, 11)}
rx = X1.rx
metas = [json.loads((P.PACKED / pid / "pack_meta.json").read_text()) for pid, _, _ in P.PATIENTS]
PO_IN = {m["patient"]: np.load(P.PACKED / m["patient"] / "CT_01_mu.npy") for m in metas}
PO_TG = {m["patient"]: np.load(P.PACKED / m["patient"] / f"CT_{int(m['reference_phase_0idx']) + 1:02d}_mu.npy") for m in metas}


ANG = X1.XRAY_ANGLES


def realistic_target(tgt_mu, seed):
    """Clinic-like X-ray of the target per view (front, side) from its HU (mu -> HU), fixed seed per view."""
    t = torch.from_numpy(((tgt_mu / v2.MU_WATER) * 1000.0 - 1000.0)[None, None].astype(np.float32)).to(dev)
    soft, bone = rx.materials(t)
    return torch.cat([rx.clinic_projection(X1.dt.drr(soft, [a]) / 10.0, X1.dt.drr(bone, [a]) / 10.0,
                                           torch.Generator().manual_seed(seed + int(a * 10))) for a in ANG], 1)


def pred(g, mu, tgt, rp, tp, mode, seed=None):
    x = torch.from_numpy(mm(mu)[None, None]).to(dev)
    t = torch.from_numpy(mm(tgt)[None, None]).to(dev)
    if mode == "zero":
        vol, d2 = X1.zero_inputs(x, t)
    elif mode == "real":
        vol, d2 = X1.xray_inputs(x, t)
    else:                                                    # clinic / shuffled / front_only / side_only: realistic
        seed = int(float(np.abs(tgt).sum()) * 1000) % (2 ** 31) if seed is None else seed   # fixed per volume
        with torch.no_grad():
            G_ = X1.cleaner(dev)
            pt = X1._std2d(torch.cat([G_(X1._std2d(o)) for o in realistic_target(tgt, seed).split(1, 1)], 1))
            pr = X1._std2d(torch.cat([G_(X1._std2d(o)) for o in X1.dt.drr(x, ANG).split(1, 1)], 1))
            d2 = pt - pr
            vol = torch.cat([X1.dt.backproject(pt, ANG, x.shape[2:]), X1.dt.backproject(d2, ANG, x.shape[2:])], 1)
            if mode in ("front_only", "side_only"):                     # blank the other view (channels v and V+v)
                k = 1 if mode == "front_only" else 0
                vol[:, [k, len(ANG) + k]] = 0; d2[:, k] = 0
    with torch.no_grad():
        d = g(x, torch.tensor([rp], device=dev), torch.tensor([tp], device=dev), vol, d2)[0].cpu().numpy()
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
    g = X1.XrayUNet2(im_size=160, n_phases=10)
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
        for mode in ("real", "clinic", "zero", "shuffled", "front_only", "side_only"):
            r = score_mode(g, mode); out[name][str(e)][mode] = r
            OUT.write_text(json.dumps(out, indent=1))
            print(f"{name:11s} {e:3d} {mode:8s} {np.mean(r['dir_plain']):9.3f} {np.mean(r['dir_flip']):9.3f} {np.mean(r['popi_plain']):10.3f} "
                  f"{np.mean(r['popi_flip']):9.3f} {np.mean(r['dir_fold_pct']):7.4f}", flush=True)
        del g; torch.cuda.empty_cache()
