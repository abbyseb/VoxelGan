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
    cfg = LEVELS[level]
    soft, bone = materials(hu)
    with torch.no_grad():
        ls = dt.drr(soft, [angle])[:, :1] / 10.0                                # g/cm^2 (drr integrates in mm)
        lb = dt.drr(bone, [angle])[:, :1] / 10.0
    e, phi = spectrum(kvp)
    w = torch.as_tensor(phi * e / (phi * e).sum(), dtype=hu.dtype, device=hu.device)   # energy-integrating detector
    ms = torch.as_tensor(_interp(e, MU_SOFT), dtype=hu.dtype, device=hu.device)
    mb = torch.as_tensor(_interp(e, MU_BONE), dtype=hu.dtype, device=hu.device)
    prim = (w.view(1, -1, 1, 1) * torch.exp(-(ms.view(1, -1, 1, 1) * ls + mb.view(1, -1, 1, 1) * lb))).sum(1, keepdim=True)
    total = prim + cfg["spr"] * _gauss2d(prim, 48.0 / dt.DET_SPACING) if cfg["spr"] > 0 else prim
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
