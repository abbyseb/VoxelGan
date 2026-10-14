#!/usr/bin/env python3
"""X-ray Cleaner (XC): pix2pix-style conditional GAN that turns a realistic kV X-ray into the clean DRR the motion models
understand (removes scatter fog, noise, blur, detector gamma, couch rails, cables). Patient-independent, trained on
synthetic PAIRS made on the fly from TCIA training-patient CTs (val patients held out).

Why: X2 (trained on perfect DRRs) gives 2.93 mm on DIR-Lab but 9.3 mm on realistic X-rays; the look-and-correct residual
signal is strong on clean X-rays (scale-direction corr 0.83) but faint on realistic ones (0.55; scatter and rails dominate).
Literature: DSE (Maier et al., U-Net scatter estimation trained on simulations, <2 % error), pix2pix X-ray->DRR translation,
Wei et al. 2020 (patient-independent cGAN intensity matching for single-projection lung localisation).

Pair per sample: random TCIA train scan/phase CT (160^3, 2 mm), random gantry angle (40 % front 0, 30 % side 90, 30 % any),
input  = std( realistic projection )  with WIDER randomisation than xray/realistic_xray.CLINIC (SPR 0-1.0, photons
         10^4.3-10^5.7, electronic noise 3-25, blur 0-1.2 px, gamma 0.85-1.15, rails 60 %, cables 50 %, no shift), 15 % of inputs
         clean (identity must be preserved);
target = std( clean DRR of the min-max mu volume )   (exactly what X1/X2/X3S were trained with).
Generator: 2-D U-Net (5 levels, base 32, ~7.8M), output = input + residual. Discriminator: conditional PatchGAN (3 strided
layers, LSGAN). Loss G = 100 * L1 + 50 * L1(Sobel gradients) + 1 * adversarial.
  python -u train_cleaner.py --gpu 0 [--iters 24000] [--check-only]
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
LITE = HERE.parent
sys.path.insert(0, str(LITE / "xray"))
import drr_torch as dt  # noqa: E402
import realistic_xray as rx  # noqa: E402

MAN = json.loads((LITE / "data_allscans" / "manifest.json").read_text())
IM_DIR = Path(MAN["pooled_train_dir"])
MU_MAX = {k: v[1] for k, v in json.loads((LITE / "xray" / "tables" / "tcia_ct_mu_minmax.json").read_text()).items()}
pat = lambda f: Path(f).name.split("_")[0]
TRAIN_P = sorted({pat(f) for f in MAN["train_pairs"]}); VAL_P = sorted({pat(f) for f in MAN["val_pairs"]})
assert not set(TRAIN_P) & set(VAL_P)
WIDE = dict(spr=(0.0, 1.0), n0_log10=(4.3, 5.7), sigma_e=(3.0, 25.0), blur=(0.0, 1.2), shift=(0, 0),
            gamma=(0.85, 1.15), p_rails=0.6, p_cables=0.5)
P_CLEAN_IN = 0.15


def std2d(p):
    return (p - p.mean((-2, -1), keepdim=True)) / p.std((-2, -1), keepdim=True).clamp_min(1e-6)


_CACHE: dict = {}


def load_ct(key, dev):
    """mu volume (min 0) on the GPU; cached after first use (all 820 volumes = ~13 GB float32)."""
    if key not in _CACHE:
        _CACHE[key] = torch.from_numpy(np.load(IM_DIR / f"{key}.npy").astype(np.float32))[None, None].to(dev)
    return _CACHE[key], MU_MAX[key]


def make_pair(patients, gen, dev):
    p = patients[int(torch.randint(len(patients), (1,), generator=gen))]
    key = f"{p}_CT_{int(torch.randint(10, (1,), generator=gen)) + 1:02d}"
    mu, mx = load_ct(key, dev)
    u = float(torch.rand(1, generator=gen))
    ang = 0.0 if u < 0.4 else (90.0 if u < 0.7 else 360.0 * float(torch.rand(1, generator=gen)))
    with torch.no_grad():
        clean = std2d(dt.drr(mu / mx, [ang]))                                # min-max mu (min 0) -> what the models saw
        if float(torch.rand(1, generator=gen)) < P_CLEAN_IN:
            x = clean.clone()
        else:
            hu = mu / 2e-5 - 1000.0
            soft, bone = rx.materials(hu)
            c0 = dict(rx.CLINIC); rx.CLINIC.clear(); rx.CLINIC.update(WIDE)
            try:
                x = std2d(rx.clinic_projection(dt.drr(soft, [ang]) / 10.0, dt.drr(bone, [ang]) / 10.0, gen))
            finally:
                rx.CLINIC.clear(); rx.CLINIC.update(c0)
    return x, clean


def batch(patients, gen, dev, n):
    xs, ys = zip(*(make_pair(patients, gen, dev) for _ in range(n)))
    return torch.cat(xs), torch.cat(ys)


def conv_block(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.InstanceNorm2d(co, affine=True), nn.LeakyReLU(0.2, True),
                         nn.Conv2d(co, co, 3, padding=1), nn.InstanceNorm2d(co, affine=True), nn.LeakyReLU(0.2, True))


class UNet2D(nn.Module):
    def __init__(self, base=32, depth=5):
        super().__init__()
        ch = [base * 2 ** min(i, 3) for i in range(depth)]                  # 32, 64, 128, 256, 256
        self.downs = nn.ModuleList([conv_block(1 if i == 0 else ch[i - 1], ch[i]) for i in range(depth)])
        self.ups = nn.ModuleList([conv_block(ch[i + 1] + ch[i], ch[i]) for i in reversed(range(depth - 1))])
        self.out = nn.Conv2d(ch[0], 1, 1); nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x):
        skips, h = [], x
        for i, d in enumerate(self.downs):
            h = d(h if i == 0 else F.avg_pool2d(h, 2)); skips.append(h)
        for u, s in zip(self.ups, reversed(skips[:-1])):
            h = u(torch.cat([F.interpolate(h, size=s.shape[-2:], mode="bilinear", align_corners=False), s], 1))
        return x + self.out(h)                                               # residual: starts as identity


class PatchD(nn.Module):
    def __init__(self, base=64):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(2, base, 4, 2, 1), nn.LeakyReLU(0.2, True),
                                 nn.Conv2d(base, base * 2, 4, 2, 1), nn.InstanceNorm2d(base * 2), nn.LeakyReLU(0.2, True),
                                 nn.Conv2d(base * 2, base * 4, 4, 2, 1), nn.InstanceNorm2d(base * 4), nn.LeakyReLU(0.2, True),
                                 nn.Conv2d(base * 4, 1, 3, 1, 1))

    def forward(self, x, y):
        return self.net(torch.cat([x, y], 1))


SOB = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])


def grad_img(y):
    k = SOB.to(y.device)
    yp = F.pad(y, (1, 1, 1, 1), mode="replicate")
    return torch.cat([F.conv2d(yp, k.view(1, 1, 3, 3)), F.conv2d(yp, k.t().contiguous().view(1, 1, 3, 3))], 1)


def evaluate(G, val, dev):
    G.eval(); r = {"l1_in": [], "l1_out": [], "gl1_in": [], "gl1_out": []}
    with torch.no_grad():
        for x, y in val:
            o = G(x)
            r["l1_in"].append(float((x - y).abs().mean())); r["l1_out"].append(float((o - y).abs().mean()))
            r["gl1_in"].append(float((grad_img(x) - grad_img(y)).abs().mean())); r["gl1_out"].append(float((grad_img(o) - grad_img(y)).abs().mean()))
    G.train(); return {k: float(np.mean(v)) for k, v in r.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--iters", type=int, default=24000)
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=20261015); ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu); dev = torch.device("cuda:0"); torch.manual_seed(a.seed)
    out = HERE / "run"; (out / "ckpt").mkdir(parents=True, exist_ok=True)
    gv = torch.Generator().manual_seed(12345)
    val = [batch(VAL_P, gv, dev, 8) for _ in range(25)]                     # 200 fixed val pairs (held-out patients)
    G, D = UNet2D().to(dev), PatchD().to(dev)
    gate = {"n_train_patients": len(TRAIN_P), "n_val_patients": len(VAL_P), "G_params_M": sum(p.numel() for p in G.parameters()) / 1e6,
            "init": evaluate(G, val, dev)}
    gate["identity_at_init"] = abs(gate["init"]["l1_out"] - gate["init"]["l1_in"]) < 1e-6
    print("gate", json.dumps(gate), flush=True); (out / "gate.json").write_text(json.dumps(gate, indent=1))
    if not gate["identity_at_init"]: raise SystemExit("gate failed")
    if a.check_only: return
    oG = torch.optim.Adam(G.parameters(), lr=a.lr, betas=(0.5, 0.999)); oD = torch.optim.Adam(D.parameters(), lr=a.lr, betas=(0.5, 0.999))
    sG = torch.optim.lr_scheduler.CosineAnnealingLR(oG, a.iters, eta_min=a.lr * 0.02); sD = torch.optim.lr_scheduler.CosineAnnealingLR(oD, a.iters, eta_min=a.lr * 0.02)
    gt = torch.Generator().manual_seed(a.seed); hist, t0, best = [], time.time(), 1e9
    for it in range(1, a.iters + 1):
        x, y = batch(TRAIN_P, gt, dev, a.bs)
        o = G(x)
        lD = 0.5 * ((D(x, y) - 1) ** 2).mean() + 0.5 * (D(x, o.detach()) ** 2).mean()
        oD.zero_grad(set_to_none=True); lD.backward(); oD.step(); sD.step()
        l1 = (o - y).abs().mean(); lg = (grad_img(o) - grad_img(y)).abs().mean(); la = ((D(x, o) - 1) ** 2).mean()
        lG = 100 * l1 + 50 * lg + la
        oG.zero_grad(set_to_none=True); lG.backward(); oG.step(); sG.step()
        if it % 1000 == 0 or it == a.iters:
            ev = evaluate(G, val, dev); ev.update({"iter": it, "train_l1": float(l1), "train_adv": float(la), "D": float(lD), "h": (time.time() - t0) / 3600})
            hist.append(ev); (out / "history.json").write_text(json.dumps(hist, indent=1))
            torch.save({"G": G.state_dict(), "D": D.state_dict(), "iter": it}, out / "ckpt" / "latest.pt")
            if ev["l1_out"] < best:
                best = ev["l1_out"]; torch.save({"G": G.state_dict(), "iter": it}, out / "ckpt" / "best_G.pt")
            print(f"it {it} | val L1 in {ev['l1_in']:.4f} -> out {ev['l1_out']:.4f} | grad L1 {ev['gl1_in']:.4f} -> {ev['gl1_out']:.4f} "
                  f"| adv {ev['train_adv']:.3f} D {ev['D']:.3f} | {ev['h']:.2f} h", flush=True)


if __name__ == "__main__":
    main()
