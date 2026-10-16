"""corrField-style keypoint correspondences (re-implementation from the paper: Hansen & Heinrich, "Estimation of large motion in
lung CT by integrating regularized keypoint correspondences into dense deformable registration", arXiv 1807.00467), because the
official code links are no longer available. INDEPENDENT of our LNCC loss: it matches MIND-SSC structure descriptors.
  1. Foerstner keypoints in the FIXED image (lung, NMS).
  2. MIND-SSC descriptors (12 channels) of fixed and moving images.
  3. Discrete search: for every keypoint, cost(d) = mean |MIND_f(p) - MIND_m(p + d)| over a 3^3 patch, d on a cubic grid of
     +-R voxels (step 1 voxel = 2 mm), coarse-to-fine (search at 2x subsampled grid first, then refine).
  4. Regularisation: kNN graph (k = 8) mean-field / loopy belief smoothing of the per-keypoint displacement distributions
     (each message = min-convolution approximated by Gaussian smoothing of neighbour beliefs), then soft-argmin displacement.
  5. Symmetric consistency: keypoints whose forward + backward displacement disagree by > 1.5 vox are dropped.
Displacement convention = our pull field u(p): fixed(p) = moving(p + u(p)), voxels, channels (x, y, z) i.e. (axis2, axis1, axis0).
  python3 corrfield_lite.py dirlab   # accuracy on DIR-Lab (both CTs; teacher-quality check only, no training on DIR-Lab)
"""
import sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

dev = torch.device("cuda:0")


def mind_ssc(img, r=1, d=2):
    """MIND-SSC (Heinrich 2013), 12 channels, img (1,1,D,H,W)."""
    six = torch.tensor([[0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 2], [2, 1, 1], [1, 2, 1]], device=img.device).long()
    dist = ((six[:, None, :] - six[None, :, :]) ** 2).sum(-1)
    x, y = torch.meshgrid(torch.arange(6, device=img.device), torch.arange(6, device=img.device), indexing="ij")
    m = (x > y) & (dist == 2)
    i1, i2 = six[x[m]], six[y[m]]
    k1 = torch.zeros(12, 1, 3, 3, 3, device=img.device); k2 = torch.zeros(12, 1, 3, 3, 3, device=img.device)
    for c in range(12):
        k1[c, 0, i1[c, 0], i1[c, 1], i1[c, 2]] = 1; k2[c, 0, i2[c, 0], i2[c, 1], i2[c, 2]] = 1
    pad = lambda t: F.pad(t, (d,) * 6, mode="replicate")
    a = F.conv3d(pad(img), k1, dilation=d); b = F.conv3d(pad(img), k2, dilation=d)
    ssd = F.avg_pool3d(F.pad((a - b) ** 2, (r,) * 6, mode="replicate"), 2 * r + 1, stride=1)
    ssd = ssd - ssd.min(1, keepdim=True)[0]
    v = ssd.mean(1, keepdim=True).clamp(ssd.mean() * 0.001, ssd.mean() * 1000)
    return torch.exp(-ssd / v)


def foerstner(img, mask, n=2000):
    g = torch.gradient(img[0, 0], dim=(0, 1, 2)); blur = lambda x: F.avg_pool3d(x[None, None], 5, 1, 2)[0, 0]
    T = torch.stack([torch.stack([blur(g[i] * g[j]) for j in range(3)], -1) for i in range(3)], -2)
    s = torch.linalg.det(T) / (T.diagonal(dim1=-2, dim2=-1).sum(-1) + 1e-8)
    m = F.max_pool3d(-F.max_pool3d(-(mask[0] > 0.5).float()[None], 5, 1, 2), 1)[0, 0] > 0.5
    s = torch.where(m, s, torch.zeros_like(s)); s = torch.where(s == F.max_pool3d(s[None, None], 3, 1, 1)[0, 0], s, torch.zeros_like(s))
    return torch.stack(torch.unravel_index(torch.topk(s.flatten(), min(n, int((s > 0).sum()))).indices, s.shape), 1)


def sample(feat, pts):
    """feat (1,C,D,H,W), pts (...,3) as (z,y,x) voxel coords -> (..., C) trilinear."""
    D, H, W = feat.shape[2:]
    g = torch.stack([pts[..., 2] / (W - 1), pts[..., 1] / (H - 1), pts[..., 0] / (D - 1)], -1) * 2 - 1
    sh = g.shape[:-1]
    out = F.grid_sample(feat, g.reshape(1, 1, 1, -1, 3), mode="bilinear", padding_mode="border", align_corners=True)
    return out.reshape(feat.shape[1], -1).T.reshape(*sh, feat.shape[1])


def match(mf, mm_, kp, R, step):
    """cost volume (N, L) for displacements on grid {-R..R}*step (voxels, z,y,x)."""
    lin = torch.arange(-R, R + 1, device=kp.device, dtype=torch.float32) * step
    disp = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), -1).reshape(-1, 3)        # (L,3) z,y,x
    off = torch.stack(torch.meshgrid(*(torch.arange(-1, 2, device=kp.device, dtype=torch.float32),) * 3, indexing="ij"), -1).reshape(-1, 3)
    ff = sample(mf, kp[:, None, :].float() + off[None])                                             # (N,27,C)
    cost = torch.zeros(len(kp), len(disp), device=kp.device)
    for i in range(0, len(disp), 512):
        dd = disp[i:i + 512]
        fm = sample(mm_, kp[:, None, None, :].float() + dd[None, :, None, :] + off[None, None])    # (N,l,27,C)
        cost[:, i:i + 512] = (fm - ff[:, None]).abs().mean((2, 3))
    return cost, disp


def regularise(cost, disp, kp, R, k=8, alpha=1.0, iters=5, sigma=1.0):
    """kNN-graph mean-field: belief_i <- exp(-alpha*cost_i) * prod_j smooth(belief_j). Gaussian smoothing over displacement grid."""
    n = len(kp); L = 2 * R + 1
    d = torch.cdist(kp.float(), kp.float()); nbr = d.topk(k + 1, largest=False).indices[:, 1:]
    unary = -alpha * (cost - cost.mean(1, keepdim=True)) / (cost.std(1, keepdim=True) + 1e-6)
    logb = unary.log_softmax(1)
    gk = torch.exp(-torch.arange(-3, 4, device=kp.device, dtype=torch.float32) ** 2 / (2 * sigma ** 2)); gk = gk / gk.sum()
    for _ in range(iters):
        b = logb.exp().reshape(n, 1, L, L, L)
        for ax in range(3):
            sh = [1, 1, 1, 1, 1]; sh[2 + ax] = 7; pd = [0] * 6; pd[2 * (2 - ax)] = 3; pd[2 * (2 - ax) + 1] = 3
            b = F.conv3d(F.pad(b, pd), gk.view(*sh))
        msg = torch.log(b.reshape(n, -1).clamp_min(1e-12))
        logb = (unary + msg[nbr].mean(1)).log_softmax(1)
    p = logb.exp()
    return p @ disp                                                                               # (N,3) z,y,x soft-argmin


def correspondences(fixed, moving, mask_f, mask_m, R=12, n=2000):
    """fixed, moving (1,1,D,H,W) in [0,1]. Returns kp (N,3) z,y,x and u (N,3) x,y,z voxels, after symmetric check."""
    mf, mm_ = mind_ssc(fixed), mind_ssc(moving)
    kp = foerstner(fixed, mask_f, n)
    c1, d1 = match(mf, mm_, kp, R // 2, 2.0)                          # coarse +-R vox, step 2
    u1 = regularise(c1, d1, kp, R // 2)
    c2, d2 = match(mf, mm_, (kp.float() + u1), 3, 1.0)                # refine +-3 vox around coarse
    u = u1 + regularise(c2, d2, kp, 3, sigma=0.7)
    # backward check: match from moving points back to fixed
    bp = (kp.float() + u)
    cb, db = match(mm_, mf, bp.round().long().clamp(0, fixed.shape[2] - 1), 3, 1.0)
    ub = regularise(cb, db, bp.round().long(), 3, sigma=0.7) + (bp.round() - bp)
    ok = (kp.float() + u + ub - kp.float()).norm(dim=1) < 1.5 + 0 * u[:, 0]
    ok = ((u + ub).norm(dim=1) < 1.5)
    return kp[ok], u[ok][:, [2, 1, 0]]                                 # -> (x,y,z) channels


if __name__ == "__main__" and sys.argv[1] == "dirlab":
    import os, json, SimpleITK as sitk
    sys.path.insert(0, "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/scripts"); os.environ.setdefault("DIRLAB_ROOT", "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/data/dirlab_packs")
    import eval_dir_tcia3_iso2mm_v2 as v2
    from dirlab_tre import landmarks_300
    from scipy.interpolate import RBFInterpolator
    res = []
    for c in range(1, 11):
        pk = v2.pack_case(c); im = sitk.ReadImage(str(pk["train"] / "CT_01.mha"))
        t00 = v2.hu_to_mu(v2.resample_to_iso(sitk.GetArrayFromImage(im).astype(np.float32), np.array(im.GetOrigin()), np.array(im.GetSpacing()), pk["origin_iso_xyz"], order=1, cval=-1000.0))
        T = lambda a: torch.from_numpy(v2.minmax(a)[None, None].astype(np.float32)).to(dev)
        lung = torch.from_numpy(pk["lung_iso"].astype(np.float32))[None, None].to(dev)
        with torch.no_grad():
            kp, u = correspondences(T(t00), T(pk["mu"]), lung, lung)     # fixed = T00 (target), moving = T50 (input): pull field
        kp_, u_ = kp.cpu().numpy().astype(float), u.cpu().numpy()
        l0 = v2.official_to_iso(landmarks_300(c, "T00"), pk); l5 = v2.official_to_iso(landmarks_300(c, "T50"), pk)   # (x,y,z)
        rbf = RBFInterpolator(kp_[:, ::-1], u_, kernel="thin_plate_spline", smoothing=5.0, neighbors=64)
        pu = rbf(l0)
        tre = float(np.linalg.norm((l0 + pu - l5) * v2.SPACING, axis=1).mean()); res.append(tre)
        print(f"C{c:02d} keypoints {len(kp)} | corrfield-lite TRE (thin-plate interpolated to landmarks) {tre:.2f} mm | no motion "
              f"{float(np.linalg.norm((l5 - l0) * v2.SPACING, axis=1).mean()):.2f}", flush=True)
    print("mean", round(float(np.mean(res)), 3))
    (Path(__file__).resolve().parent / "corrfield_lite_dirlab.json").write_text(json.dumps(res))
