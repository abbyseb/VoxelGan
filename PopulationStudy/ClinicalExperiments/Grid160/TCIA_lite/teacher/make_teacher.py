"""Stage 1 teacher: per TRAINING pair, refine the Elastix label by direct optimisation against the full target CT
(3-D lung LNCC_9 + bending on a coarse 40^3 residual, Adam) -- the same optimiser that reaches 1.34 mm on DIR-Lab from a network
start (Elastix oracle on our scorer: 1.96 mm). Then pick ~2000 KEYPOINTS per pair inside the lung where the reference CT has
strong structure (Foerstner-like: largest smallest-eigenvalue of the 3-D structure tensor, non-maximum suppressed), and store the
teacher displacement there. TRAINING pairs of the TCIA manifest (all non-self pairs) + usable breath-hold pairs; never DIR-Lab/POPI.
Output per pair: {pair}_kp.npz with idx (N,3) int16 (z,y,x voxels, R3 grid) and u (N,3) float32 (x,y,z voxels, pull field).
  python3 make_teacher.py time            # timing + quality on 3 pairs
  python3 make_teacher.py run SHARD NSHARD  # all selected pairs (extreme + breath-hold), sharded
"""
import json, re, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
LITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LITE / "run_A2_full160_aug_x4" / "scripts"))
import importlib.util
spec = importlib.util.spec_from_file_location("x4", LITE / "run_A2_full160_aug_x4/scripts/train_x4.py"); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
dev = torch.device("cuda:0"); OUT = Path("/media/abhishek/3CCA3CADCA3C6574/TCIA_4D-Lung/teacher_kp"); OUT.mkdir(exist_ok=True)
POOL = Path(json.loads((LITE / "data_allscans/manifest.json").read_text())["pooled_train_dir"]); BH = Path("/media/abhishek/3CCA3CADCA3C6574/LungPairs/packed_r3")
mm = lambda a: (a - a.min()) / (a.max() - a.min() + 1e-8)


def bending(r):
    d2 = lambda x, ax: x.narrow(ax, 2, x.shape[ax] - 2) - 2 * x.narrow(ax, 1, x.shape[ax] - 2) + x.narrow(ax, 0, x.shape[ax] - 2)
    return sum(d2(r, a).pow(2).mean() for a in (2, 3, 4))


def refine(ref, tgt, mask, u0, steps=150, lam=0.5):
    r = torch.zeros(1, 3, 40, 40, 40, device=dev, requires_grad=True); opt = torch.optim.Adam([r], lr=0.05)
    for _ in range(steps):
        u = u0 + F.interpolate(r, size=u0.shape[2:], mode="trilinear", align_corners=True)
        loss = M.lncc_term(ref, tgt, u, mask) + lam * bending(r)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return u0 + F.interpolate(r, size=u0.shape[2:], mode="trilinear", align_corners=True)


def keypoints(ref, mask, n=2000, sigma=1.0):
    """Foerstner operator det(T)/trace(T) of the smoothed structure tensor, inside the eroded lung, 3x3x3 NMS, top n."""
    g = torch.gradient(ref[0, 0], dim=(0, 1, 2))
    blur = lambda x: F.avg_pool3d(x[None, None], 5, 1, 2)[0, 0]
    T = torch.stack([torch.stack([blur(g[i] * g[j]) for j in range(3)], -1) for i in range(3)], -2)
    lam = torch.linalg.det(T) / (T.diagonal(dim1=-2, dim2=-1).sum(-1) + 1e-8)      # Foerstner det/trace
    m = F.max_pool3d(-F.max_pool3d(-(mask[0] > 0.5).float()[None], 5, 1, 2), 1)[0, 0] > 0.5   # erode 2 vox
    s = torch.where(m, lam, torch.zeros_like(lam))
    nms = s == F.max_pool3d(s[None, None], 3, 1, 1)[0, 0]
    s = torch.where(nms, s, torch.zeros_like(s))
    k = min(n, int((s > 0).sum()))
    return torch.stack(torch.unravel_index(torch.topk(s.flatten(), k).indices, s.shape), 1)


def lncc_eval(ref, tgt, u, mask):
    with torch.no_grad(): return float(M.lncc_term(ref, tgt, u, mask))


def load(pair, d):
    p, a, b = re.match(r"(\w+?)_(\d+)_to_(\d+)_pair", pair).groups()
    ref = torch.from_numpy(mm(np.load(d / f"{p}_CT_{a}.npy")))[None, None].float().to(dev)
    tgt = torch.from_numpy(mm(np.load(d / f"{p}_CT_{b}.npy")))[None, None].float().to(dev)
    mask = torch.from_numpy(np.load(d / f"{p}_Mask_Lung.npy").astype(np.float32))[None, None].to(dev)
    u0 = torch.from_numpy(np.moveaxis(np.load(d / pair).astype(np.float32), -1, 0).copy())[None].to(dev)
    return ref, tgt, mask, u0


def selected():
    man = json.loads((LITE / "data_allscans/manifest.json").read_text())
    names = [Path(f).name for f in man["train_pairs"]]                       # TRAINING pairs only (never val / DIR-Lab / POPI)
    ext = [n for n in names if M.pair_weight(n) != M.W_SELF]                  # all non-self pairs (self pairs: teacher = 0)
    bh = [f"{i['id']}_{a:02d}_to_{c:02d}_pair.npy" for i in json.loads((BH / "index.json").read_text()) if not i["exclude"] for a, c in ((6, 1), (1, 6))]
    return [(n, POOL) for n in ext] + [(n, BH) for n in bh]


def do(pair, d):
    ref, tgt, mask, u0 = load(pair, d)
    u = refine(ref, tgt, mask, u0)
    idx = keypoints(ref, mask)
    uu = u[0][:, idx[:, 0], idx[:, 1], idx[:, 2]].T.cpu().numpy().astype(np.float32)
    np.savez_compressed(OUT / pair.replace(".npy", "_kp.npz"), idx=idx.cpu().numpy().astype(np.int16), u=uu)
    return lncc_eval(ref, tgt, u0, mask), lncc_eval(ref, tgt, u, mask), len(idx)


if __name__ == "__main__":
    sel = selected()
    if sys.argv[1] == "time":
        print("selected pairs", len(sel))
        for pair, d in sel[:2] + sel[-1:]:
            t0 = time.time(); a, b, n = do(pair, d); print(f"{pair}: LNCC loss Elastix {a:.4f} -> teacher {b:.4f}, {n} keypoints, {time.time() - t0:.1f} s", flush=True)
    else:
        k, K = int(sys.argv[2]), int(sys.argv[3]); log = []
        for i, (pair, d) in enumerate(sel):
            if i % K != k or (OUT / pair.replace(".npy", "_kp.npz")).exists(): continue
            a, b, n = do(pair, d); log.append((pair, a, b, n))
            if len(log) % 50 == 0: print(f"{len(log)} done, last {pair} {a:.4f}->{b:.4f}", flush=True)
        (OUT / f"log_{k}.json").write_text(json.dumps(log))
