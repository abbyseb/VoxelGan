"""Realistic kV X-ray projection from a CT in HU, same cone-beam geometry as drr_torch.py (so it can replace the clean DRR).

Physics, following the open-source simulators:
  * DeepDRR (arcadelab/deepdrr): material decomposition of the CT by HU (air/soft tissue < 350 HU <= bone), density from
    HU, polychromatic spectrum, energy-integrating detector, scatter added to the primary.
  * TIGRE (CERN/TIGRE, CTnoise.add): Poisson quantum noise on the photon counts + Gaussian electronic noise, then the
    -log flat-field transform.
Data: NIST X-ray mass attenuation coefficients (physics.nist.gov/PhysRefData/XrayMassCoef), Soft Tissue (ICRU-44),
      Bone Cortical (ICRU-44), Aluminium (filtration); log-log interpolated.
Spectrum: Kramers' law for a 120 kVp tungsten tube (fluence ~ (kVp - E) / E), 2.5 mm Al filtration (no characteristic lines).
Scatter: scatter-to-primary ratio (SPR) model, S = SPR * Gaussian-blurred primary (sigma 48 mm at the detector), the usual
         kernel approximation of a broad scatter field.
Detector: Gaussian PSF blur, optional sub-pixel/pixel shift (calibration error).

realistic_projection(hu, angle, level) -> (B,1,n,n) line-integral-like image p = -ln(N / N0), ready for per-image
standardisation exactly like the clean DRR.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

import drr_torch as dt

E_NIST = np.array([15, 20, 30, 40, 50, 60, 80, 100, 150], float)              # keV
MU_SOFT = np.array([1.699, 0.8230, 0.3790, 0.2688, 0.2264, 0.2048, 0.1823, 0.1693, 0.1492])   # cm^2/g
MU_BONE = np.array([9.032, 4.001, 1.331, 0.6655, 0.4242, 0.3148, 0.2229, 0.1855, 0.1480])
MU_AL = np.array([7.955, 3.441, 1.128, 0.5685, 0.3681, 0.2778, 0.2018, 0.1704, 0.1378])
RHO_AL = 2.699

LEVELS = {
    #            SPR   photons/pixel  electronic sd  PSF sd (px)  detector shift (px)
    "clean":    None,
    "physics":  dict(spr=0.0, n0=None, sigma_e=0.0, blur=0.0, shift=0.0),
    "mild":     dict(spr=0.3, n0=1e5, sigma_e=10.0, blur=0.5, shift=0.0),
    "moderate": dict(spr=1.0, n0=2e4, sigma_e=20.0, blur=0.8, shift=0.0),
    "severe":   dict(spr=2.0, n0=3e3, sigma_e=30.0, blur=1.0, shift=1.0),
}


def _interp(e, mu):
    return np.exp(np.interp(np.log(e), np.log(E_NIST), np.log(mu)))


def spectrum(kvp: float = 120.0, al_mm: float = 2.5, de: float = 5.0):
    e = np.arange(15.0, kvp, de) + de / 2
    phi = (kvp - e) / e * np.exp(-_interp(e, MU_AL) * RHO_AL * al_mm / 10)
    return e, phi / phi.sum()


def materials(hu: torch.Tensor):
    """HU -> (soft-tissue density map, bone density map) in g/cm^3 (DeepDRR-style threshold at 350 HU)."""
    rho = (1.0 + hu / 1000.0).clamp_min(0.0)
    bone = (hu >= 350).float()
    return rho * (1 - bone), rho * bone


def _gauss2d(img: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return img
    r = int(math.ceil(3 * sigma))
    x = torch.arange(-r, r + 1, device=img.device, dtype=img.dtype)
    k = torch.exp(-x ** 2 / (2 * sigma ** 2)); k = k / k.sum()
    img = F.conv2d(F.pad(img, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(img, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))


def realistic_projection(hu: torch.Tensor, angle: float, level: str, gen: torch.Generator | None = None,
                         kvp: float = 120.0) -> torch.Tensor:
    """hu (B,1,D,H,W) on the 2 mm grid -> (B,1,n,n) -ln(N/N0). level 'clean' is not handled here (use dt.drr)."""
    soft, bone = materials(hu)
    with torch.no_grad():
        ls = dt.drr(soft, [angle])[:, :1] / 10.0                                # g/cm^2 (drr integrates in mm)
        lb = dt.drr(bone, [angle])[:, :1] / 10.0
    return from_line_integrals(ls, lb, level, gen, kvp, DET_PX_MM=dt.DET_SPACING)


def from_line_integrals(ls: torch.Tensor, lb: torch.Tensor, level: str, gen: torch.Generator | None = None,
                        kvp: float = 120.0, DET_PX_MM: float = dt.DET_SPACING) -> torch.Tensor:
    """Soft-tissue / bone mass line integrals (B,1,rows,cols) in g/cm^2 -> -ln(N/N0) with the level's physics."""
    cfg = LEVELS[level]
    hu = ls
    e, phi = spectrum(kvp)
    w = torch.as_tensor(phi * e / (phi * e).sum(), dtype=hu.dtype, device=hu.device)   # energy-integrating detector
    ms = torch.as_tensor(_interp(e, MU_SOFT), dtype=hu.dtype, device=hu.device)
    mb = torch.as_tensor(_interp(e, MU_BONE), dtype=hu.dtype, device=hu.device)
    prim = (w.view(1, -1, 1, 1) * torch.exp(-(ms.view(1, -1, 1, 1) * ls + mb.view(1, -1, 1, 1) * lb))).sum(1, keepdim=True)
    total = prim + cfg["spr"] * _gauss2d(prim, 48.0 / DET_PX_MM) if cfg["spr"] > 0 else prim
    total = _gauss2d(total, cfg["blur"])
    if cfg["shift"]:
        total = torch.roll(total, shifts=int(round(cfg["shift"])), dims=-1)
    if cfg["n0"] is None:
        return -torch.log(total.clamp_min(1e-12))
    n0 = cfg["n0"]
    lam = (n0 * total).double()
    counts = torch.poisson(lam.cpu(), generator=gen).to(hu.device).float()
    counts = counts + cfg["sigma_e"] * torch.randn(counts.shape, generator=gen).to(hu.device)
    return -torch.log(counts.clamp_min(1.0) / n0)


# ------------------------------------------------------------------ "clinic-like" randomised setting
# Chosen after comparing with a real SPARE ClinicalVarian projection (check/real_vs_sim_spare_p1.png): the real image
# looks between our 'physics' and 'mild' levels, with couch rails and cables that the CT does not contain.
CLINIC = dict(spr=(0.0, 0.5), n0_log10=(4.7, 5.5), sigma_e=(5.0, 15.0), blur=(0.3, 0.8), shift=(-1, 1),
              gamma=(0.9, 1.1), p_rails=0.7, p_cables=0.5)


def _u(g, lo, hi):
    return lo + (hi - lo) * float(torch.rand(1, generator=g))


def clinic_projection(ls: torch.Tensor, lb: torch.Tensor, gen: torch.Generator, DET_PX_MM: float = dt.DET_SPACING):
    """Mass line integrals -> one random clinic-like X-ray (-ln N/N0): random mild scatter/noise/blur/shift, a small
    gamma (detector non-linearity), and sometimes couch rails (2 bright vertical bars) and cables (thin bright curves)."""
    c = CLINIC
    cfg = dict(spr=_u(gen, *c["spr"]), n0=10 ** _u(gen, *c["n0_log10"]), sigma_e=_u(gen, *c["sigma_e"]),
               blur=_u(gen, *c["blur"]), shift=0.0)
    LEVELS["_clinic"] = cfg
    p = from_line_integrals(ls, lb, "_clinic", gen, DET_PX_MM=DET_PX_MM)
    n_r, n_c = p.shape[-2:]
    if float(torch.rand(1, generator=gen)) < c["p_rails"]:                     # carbon-fibre couch rails
        w = max(1, int(round(6.0 / DET_PX_MM)))
        x0 = int(_u(gen, 0.1, 0.35) * n_c); gap = int(_u(gen, 0.3, 0.5) * n_c)
        for x in (x0, min(n_c - w, x0 + gap)):
            p[..., :, x:x + w] += _u(gen, 0.3, 0.8)
    if float(torch.rand(1, generator=gen)) < c["p_cables"]:                     # ECG / monitoring cables
        yy = torch.arange(n_r, device=p.device, dtype=p.dtype)[:, None]; xx = torch.arange(n_c, device=p.device, dtype=p.dtype)[None]
        for _ in range(int(_u(gen, 1, 3.99))):
            a, b, f, ph = _u(gen, 0.1, 0.9) * n_r, _u(gen, 0.05, 0.2) * n_r, _u(gen, 0.5, 2.0), _u(gen, 0, 6.28)
            line = a + b * torch.sin(2 * math.pi * f * xx / n_c + ph)
            p += _u(gen, 0.1, 0.4) * torch.exp(-((yy - line) ** 2) / (2 * 0.8 ** 2))
    s = int(round(_u(gen, *c["shift"])))
    if s:
        p = torch.roll(p, shifts=s, dims=-1)
    gmm = _u(gen, *c["gamma"])
    lo = p.amin((-2, -1), keepdim=True)
    return lo + (p - lo).clamp_min(0) ** gmm
