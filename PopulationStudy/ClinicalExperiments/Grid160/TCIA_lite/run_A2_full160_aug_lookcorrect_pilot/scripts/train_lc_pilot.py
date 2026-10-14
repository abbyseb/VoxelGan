#!/usr/bin/env python3
"""PILOT: look-and-correct (render-and-compare) second stage on top of a FROZEN X3S (front + side, realistic X-rays).
Population model, fixed weights at test time (no per-patient tuning).

Stage 1 (frozen X3S):  u1 = X3S(CT, X-rays)
Look:   W = warp(CT, u1);  DW_v = std DRR_v(W)  (clean software render of the current guess, front + side)
Stage 2 (trained):     u2 = S2([W, bp(std X_v), bp(DW_v)],  comparator code from the 2-D stack [X_f, X_s, DW_f, DW_s])
        The comparator SEES BOTH the real X-ray and the render of the guess and learns how to compare them (the feasibility
        study showed a fixed pixel/edge difference is swamped by scatter / rails on realistic X-rays).
Correct: u = u2 + u1(x + u2)  (pull-field composition);  loss = 1 - lung LNCC_9(tgt, warp(ref_geo, u)). No labels.
S2 output conv starts at ZERO -> at step 0 the cascade equals X3S exactly.
Same data / aug / realistic X-ray mix (80 % clinic-like, 20 % perfect, both views) / seed as X3S. Pilot: 4 epochs,
cosine lr 1e-4 -> 1e-5. After every epoch: DIR-Lab TRE300 (realistic front + side, 10 cases) for X3S alone vs cascade, and
val LNCC on every 10th val pair for both.
  python -u scripts/train_lc_pilot.py --gpu 1 --stage1 <X3S checkpoint>
"""
from __future__ import annotations

import argparse, importlib.util, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

RUN = Path(__file__).resolve().parents[1]
LITE = RUN.parent
spec = importlib.util.spec_from_file_location("x3s", LITE / "run_A2_full160_aug_xray3s" / "scripts" / "train_xray3s.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
ANG = M.XRAY_ANGLES; dt = M.dt


class Comparator(nn.Module):
    """2-D stack [X_front, X_side, DW_front, DW_side] (B,4,n,n) -> n_code numbers."""

    def __init__(self, n_code: int = 8, cin: int = 2 * len(ANG)):
        super().__init__()
        ch = [cin, 32, 48, 64, 64]
        layers = []
        for i in range(4):
            layers += [nn.Conv2d(ch[i], ch[i + 1], 3, stride=2, padding=1), nn.ReLU(inplace=True)]
        self.conv = nn.Sequential(*layers); self.fc = nn.Linear(ch[-1], n_code)

    def forward(self, x):
        return self.fc(self.conv(x).mean((2, 3)))


class Stage2(M.UNetCRBDecoder):
    def __init__(self, n_code: int = 8, **kw):
        super().__init__(**kw)
        cd = 2 + n_code
        self.enc1 = M.ResidualBlock(1 + 2 * len(ANG), 16)
        self.up4 = M.UpCRB(64, 64, 64, cond_dim=cd); self.up3 = M.UpCRB(64, 32, 32, cond_dim=cd)
        self.up2 = M.UpCRB(32, 32, 32, cond_dim=cd); self.up1 = M.UpCRB(32, 16, 16, cond_dim=cd)
        self.comp = Comparator(n_code)
        nn.init.zeros_(self.out_conv.weight); nn.init.zeros_(self.out_conv.bias)

    def forward(self, w_ct, rp, tp, vol, stack2d):
        cond = torch.cat([self._phase_vec(rp, tp), self.comp(stack2d)], 1)
        s1 = self.enc1(torch.cat([w_ct, vol], 1)); s2 = self.down2(s1); s3 = self.down3(s2); s4 = self.down4(s3)
        x = self.down5(s4)
        x = self.up4(x, s4, cond); x = self.up3(x, s3, cond); x = self.up2(x, s2, cond); x = self.up1(x, s1, cond)
        return self.out_conv(x)


def xr(ref_clean, tgt, key, gen, p_clean):
    """-> pt (std target X-rays, B,V,n,n), vol1/d1 inputs of X3S. No grad."""
    with torch.no_grad():
        pt = M._std2d(M.target_projection(tgt, key, gen, p_clean, ANG)); pr = M._std2d(dt.drr(ref_clean, ANG))
        d = pt - pr
        vol = torch.cat([dt.backproject(pt, ANG, tgt.shape[2:]), dt.backproject(d, ANG, tgt.shape[2:])], 1)
    return pt, vol, d


def cascade(s1, s2, ct_in, ct_clean, rp, tp, pt, vol1, d1):
    with torch.no_grad():
        u1 = s1(ct_in, rp, tp, vol1, d1)
        w_in, w_clean = M.warp(ct_in, u1), M.warp(ct_clean, u1)
        dw = M._std2d(dt.drr(w_clean, ANG))
        vol2 = torch.cat([dt.backproject(pt, ANG, ct_in.shape[2:]), dt.backproject(dw, ANG, ct_in.shape[2:])], 1)
        stack = torch.cat([pt, dw], 1)
    u2 = s2(w_in, rp, tp, vol2, stack)
    return u1, u2 + M.warp(u1, u2)


def dir_eval(s1, s2, device):
    """DIR-Lab TRE300 plain, realistic front + side (clinic seed per volume), X3S alone vs cascade."""
    sys.path.insert(0, "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/scripts")
    os.environ.setdefault("DIRLAB_ROOT", "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/data/dirlab_packs")
    import eval_dir_tcia3_iso2mm_v2 as v2, SimpleITK as sitk
    from dirlab_tre import landmarks_300
    out = {"x3s": [], "cascade": []}
    for c in range(1, 11):
        pk = v2.pack_case(c); im = sitk.ReadImage(str(pk["train"] / "CT_01.mha"))
        t_mu = v2.hu_to_mu(v2.resample_to_iso(sitk.GetArrayFromImage(im).astype(np.float32), np.array(im.GetOrigin()),
                                              np.array(im.GetSpacing()), pk["origin_iso_xyz"], order=1, cval=-1000.0))
        x = torch.from_numpy(v2.minmax(pk["mu"])[None, None]).to(device)
        hu = torch.from_numpy(((t_mu / v2.MU_WATER) * 1000 - 1000)[None, None].astype(np.float32)).to(device)
        seed = int(float(np.abs(t_mu).sum()) * 1000) % (2 ** 31)
        with torch.no_grad():
            soft, bone = M.rx.materials(hu)
            proj = torch.cat([M.rx.clinic_projection(dt.drr(soft, [a]) / 10.0, dt.drr(bone, [a]) / 10.0,
                                                     torch.Generator().manual_seed(seed + int(a * 10))) for a in ANG], 1)
            pt = M._std2d(proj); pr = M._std2d(dt.drr(x, ANG)); d = pt - pr
            vol = torch.cat([dt.backproject(pt, ANG, x.shape[2:]), dt.backproject(d, ANG, x.shape[2:])], 1)
            rp, tp = torch.tensor([5], device=device), torch.tensor([0], device=device)
            u1, u = cascade(s1, s2, x, x, rp, tp, pt, vol, d)
        l0 = v2.official_to_iso(landmarks_300(c, "T00"), pk); l5 = v2.official_to_iso(landmarks_300(c, "T50"), pk)
        for k, f in (("x3s", u1), ("cascade", u)):
            fz = np.moveaxis(f[0].cpu().numpy(), 0, -1)
            out[k].append(float(np.linalg.norm((l0 + v2.sample_u(fz, l0) - l5) * v2.SPACING, axis=1).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1); ap.add_argument("--stage1", required=True)
    ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-min", type=float, default=1e-5); ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    M.set_seeds(a.seed); device = torch.device("cuda:0")
    (RUN / "logs").mkdir(parents=True, exist_ok=True); (RUN / "checkpoints").mkdir(parents=True, exist_ok=True)
    man = M.load_manifest()
    kw = dict(im_dir=man["pooled_train_dir"], im_size=160, random_crop=False, patches_per_pair=1, fov_aug=False)
    trainset = M.FastMmapPhasePairDataset(pair_files=man["train_pairs"], aug_seed=a.seed, **kw)
    valset = M.FastMmapPhasePairDataset(pair_files=man["val_pairs"], **kw)
    s1 = M.XrayUNet2(im_size=160, n_phases=10)
    st1 = torch.load(a.stage1, map_location="cpu", weights_only=False)
    s1.load_state_dict(st1["generator"]); s1 = s1.to(device).eval()
    for p in s1.parameters(): p.requires_grad_(False)
    s2 = Stage2(im_size=160, n_phases=10).to(device)
    # gate: at init the cascade == X3S exactly; S2 receives a gradient
    d0 = valset[len(valset) // 2]; k0 = M.ct_key({"patient": [d0["patient"]], "target_phase": d0["target_phase"][None]})
    r0, t0, m0 = (d0[k][None].to(device) for k in ("reference_ct", "target_ct", "lung_mask"))
    rp0, tp0 = d0["ref_phase"][None].to(device), d0["target_phase"][None].to(device)
    pt0, v0, dd0 = xr(r0, t0, k0, torch.Generator().manual_seed(0), 0.0)
    u1_, u_ = cascade(s1, s2, r0, r0, rp0, tp0, pt0, v0, dd0)
    gate = {"init_max_abs_diff": float((u_ - u1_).abs().max()), "stage1_epoch": st1.get("epoch_done")}
    s2.train(); u1_, u_ = cascade(s1, s2, r0, r0, rp0, tp0, pt0, v0, dd0); M.lncc_term(r0, t0, u_, m0).backward()
    gate["s2_out_grad"] = float(s2.out_conv.weight.grad.norm()); s2.zero_grad(set_to_none=True)
    # at u2 = 0 the cascade equals X3S up to float rounding of the grid_sample round-trip of u1
    gate["ok"] = gate["init_max_abs_diff"] < 1e-3 and gate["s2_out_grad"] > 0
    print("gate", json.dumps(gate), flush=True); (RUN / "logs" / "gate.json").write_text(json.dumps(gate, indent=1))
    if not gate["ok"]: raise SystemExit("GATE FAILED")
    if a.check_only: return
    g_dl = torch.Generator().manual_seed(a.seed); g_aug = torch.Generator().manual_seed(a.seed + 1); g_xr = torch.Generator().manual_seed(a.seed + 2)
    dl = dict(batch_size=1, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    tl = DataLoader(trainset, shuffle=True, generator=g_dl, **dl)
    vl = DataLoader(Subset(valset, list(range(0, len(valset), 10))), shuffle=False, **dl)
    opt = optim.Adam(s2.parameters(), lr=a.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * len(tl), eta_min=a.lr_min)
    hist, t0_ = [], time.time()
    with torch.no_grad():
        s2.eval(); base = dir_eval(s1, s2, device)
    print(f"epoch 0 DIR-Lab realistic: X3S {np.mean(base['x3s']):.3f} cascade {np.mean(base['cascade']):.3f}", flush=True)
    for ep in range(1, a.epochs + 1):
        s2.train(); tr, n = 0.0, 0
        for data in tl:
            ref, tgt, mask, dvf = (data[k].to(device, non_blocking=True) for k in ("reference_ct", "target_ct", "lung_mask", "target_dvf"))
            ref, tgt, mask, dvf, ref_geo = M.augment(ref, tgt, mask, dvf, g_aug)
            rp, tp = data["ref_phase"].to(device), data["target_phase"].to(device)
            pt, v1, d1 = xr(ref_geo, tgt, M.ct_key(data), g_xr, M.P_CLEAN)
            _, u = cascade(s1, s2, ref, ref_geo, rp, tp, pt, v1, d1)
            loss = M.lncc_term(ref_geo, tgt, u, mask)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tr += float(loss); n += 1
        s2.eval(); v_s1, v_c, nv = 0.0, 0.0, 0
        with torch.no_grad():
            for data in vl:
                ref, tgt, mask = (data[k].to(device) for k in ("reference_ct", "target_ct", "lung_mask"))
                rp, tp = data["ref_phase"].to(device), data["target_phase"].to(device)
                pt, v1, d1 = xr(ref, tgt, M.ct_key(data), torch.Generator().manual_seed(90_000 + nv), 0.0)
                u1, u = cascade(s1, s2, ref, ref, rp, tp, pt, v1, d1)
                v_s1 += float(M.lncc_term(ref, tgt, u1, mask)); v_c += float(M.lncc_term(ref, tgt, u, mask)); nv += 1
            de = dir_eval(s1, s2, device)
        h = {"epoch": ep, "train_lncc": tr / n, "val_lncc_x3s": v_s1 / nv, "val_lncc_cascade": v_c / nv,
             "dir_x3s": de["x3s"], "dir_cascade": de["cascade"], "elapsed_h": (time.time() - t0_) / 3600}
        hist.append(h); (RUN / "logs" / "history.json").write_text(json.dumps({"epoch0": base, "hist": hist}, indent=1))
        torch.save({"epoch_done": ep, "stage2": s2.state_dict(), "stage1": a.stage1}, str(RUN / "checkpoints" / f"epoch_{ep:03d}.pt"))
        print(f"epoch {ep} | train {h['train_lncc']:.4f} | val LNCC X3S {h['val_lncc_x3s']:.4f} cascade {h['val_lncc_cascade']:.4f} | "
              f"DIR-Lab realistic X3S {np.mean(de['x3s']):.3f} cascade {np.mean(de['cascade']):.3f} | {h['elapsed_h']:.2f} h", flush=True)


if __name__ == "__main__":
    main()
