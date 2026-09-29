#!/usr/bin/env python3
"""DIR-Lab oracle TRE for TCIA3 on the training grid (v2).

Fixes over eval_dir_tcia3_iso2mm.py:
  1. Landmarks use the verified chain official -> pack -> R3 (eval_a1_tre).
     v1 skipped official->pack and mirrored every landmark head-to-foot.
     The identity check cannot see that mirror; the round-trip check below can.
  2. Round-trip gate: the A1 Elastix field DVF_sub_01 is moved onto the 2 mm
     cube and scored with the same code. It must match the old verified
     evaluator (~2.08 mm TRE75). If not, the script stops.
  3. Orientation is decided from the lung masks, not from TRE: the row where
     the lung cross-section is largest (the base) must sit at the same end in
     TCIA and in DIR. A coronal PNG is written for a visual check.
  4. Both input orientations are scored and reported; the one matching the
     mask check is marked "chosen". Inside-cube cohort means are reported too.

  cd "DIR EXPERIMENTS"
  CUDA_VISIBLE_DEVICES=1 python scripts/eval_dir_tcia3_iso2mm_v2.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from scipy.ndimage import map_coordinates

DIR_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIR_EXP / "scripts"))
from dirlab_tre import CASE_INFO, landmarks_300, landmarks_75, stats, tre_mm  # noqa: E402
from eval_a1_tre import (  # noqa: E402
    load_dvf_zyx3,
    official_to_pack,
    pack_to_r3,
    tre_t00_t50,
)

VOXEL = DIR_EXP.parent
A1 = DIR_EXP / "arms" / "A1_oracle_dirlab" / "runs"
CKPT = (
    VOXEL
    / "PopulationStudy/ClinicalExperiments/Grid160/TCIA3/DecoderCRB/checkpoints/epoch_100.pt"
)
NET = VOXEL / "PopulationStudy/ClinicalExperiments/Grid160/Experiment1"
TCIA_MU = Path(
    "/media/abhishek/3CCA3CADCA3C6574/TCIA_4D-Lung/packed_r3_hu/S1/all/CT_01_mu.npy"
)
TCIA_LUNG = Path(
    "/media/abhishek/3CCA3CADCA3C6574/TCIA_4D-Lung/packed_r3_hu/S1/all/Mask_Lung.npy"
)
OUT_DIR = VOXEL / "PopulationStudy/ClinicalExperiments/Grid160/TCIA3/DecoderCRB/plots/qc_dir_oracle"
OUT = OUT_DIR / "tre75_ep100_iso2mm_minmax_v2.json"
PNG = OUT_DIR / "orientation_tcia_s1_vs_dir_c01.png"

SPACING = 2.0
SIZE = 160
HALF = 0.5 * SIZE * SPACING
MU_WATER = 0.02
SI_AXIS = 1  # numpy (Z, Y, X); R3 itk y = superior-inferior


def minmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def hu_to_mu(hu: np.ndarray) -> np.ndarray:
    return ((hu.astype(np.float32) + 1000.0) * (MU_WATER / 1000.0)).astype(np.float32)


def brightness(mu: np.ndarray, lung: np.ndarray) -> dict:
    norm = minmax(mu)
    lung = lung > 0
    return {
        "mu_min": float(mu.min()),
        "mu_max": float(mu.max()),
        "water_after_minmax": float((MU_WATER - mu.min()) / (mu.max() - mu.min() + 1e-8)),
        "lung_median_after_minmax": float(np.median(norm[lung])) if lung.any() else None,
    }


def base_row_fraction(lung_zyx: np.ndarray) -> float:
    """Row (along SI) with the largest lung cross-section, as a fraction 0..1.

    The lung is widest near its base, so >0.5 means the base is at high index.
    """
    area = (lung_zyx > 0).sum(axis=(0, 2)).astype(np.float64)
    rows = np.nonzero(area)[0]
    if rows.size == 0:
        raise RuntimeError("empty lung mask")
    lo, hi = rows.min(), rows.max()
    return float((np.argmax(area) - lo) / max(hi - lo, 1))


# ---------------------------------------------------------------- geometry
def load_case_image(case: int):
    sid = f"DIR_C{case:02d}"
    train = A1 / sid / sid / "train"
    ct = sitk.ReadImage(str(train / "CT_06.mha"))
    mask = sitk.ReadImage(str(train / "Mask_Lung.mha"))
    if ct.GetSize() != mask.GetSize() or ct.GetSpacing() != mask.GetSpacing():
        raise RuntimeError(f"{sid} CT and lung mask grids differ")
    d = np.array(ct.GetDirection(), dtype=np.float64)
    if not np.allclose(d, np.eye(3).ravel()):
        raise RuntimeError(f"{sid} CT direction is not identity: {d}")
    hu = sitk.GetArrayFromImage(ct).astype(np.float32)
    lung = sitk.GetArrayFromImage(mask) > 0
    origin = np.array(ct.GetOrigin(), dtype=np.float64)
    spacing = np.array(ct.GetSpacing(), dtype=np.float64)
    return hu, lung, origin, spacing, train


def iso_physical_grid(origin_iso_xyz: np.ndarray):
    ii = np.arange(SIZE, dtype=np.float64)
    zz, yy, xx = np.meshgrid(
        origin_iso_xyz[2] + ii * SPACING,
        origin_iso_xyz[1] + ii * SPACING,
        origin_iso_xyz[0] + ii * SPACING,
        indexing="ij",
    )
    return xx, yy, zz


def resample_to_iso(vol_zyx, origin_xyz, spacing_xyz, origin_iso_xyz, order, cval):
    xx, yy, zz = iso_physical_grid(origin_iso_xyz)
    ix = (xx - origin_xyz[0]) / spacing_xyz[0]
    iy = (yy - origin_xyz[1]) / spacing_xyz[1]
    iz = (zz - origin_xyz[2]) / spacing_xyz[2]
    samp = map_coordinates(
        vol_zyx.astype(np.float32),
        np.stack([iz.ravel(), iy.ravel(), ix.ravel()], axis=0),
        order=order,
        mode="constant",
        cval=cval,
    )
    return samp.reshape(SIZE, SIZE, SIZE).astype(np.float32)


def pack_case(case: int) -> dict:
    (nx, ny, nz), _ = CASE_INFO[case]
    hu, lung, origin, spacing, train = load_case_image(case)
    if (hu.shape[2], hu.shape[1], hu.shape[0]) != (nx, nz, ny):
        raise RuntimeError(f"case {case}: CT zyx {hu.shape} is not R3 ({nx},{nz},{ny})")
    iz, iy, ix = np.where(lung)
    centroid = origin + np.stack([ix, iy, iz], 1).mean(0) * spacing
    origin_iso = centroid - HALF
    hu_iso = resample_to_iso(hu, origin, spacing, origin_iso, order=1, cval=-1000.0)
    lung_iso = resample_to_iso(lung.astype(np.float32), origin, spacing, origin_iso, 0, 0.0) > 0.5
    return {
        "hu_iso": hu_iso,
        "lung_iso": lung_iso,
        "mu": hu_to_mu(hu_iso),
        "origin_iso_xyz": origin_iso,
        "origin_xyz": origin,
        "spacing_xyz": spacing,
        "train": train,
        "nx": nx,
        "ny": ny,
        "nz": nz,
    }


def official_to_iso(official: np.ndarray, pack: dict) -> np.ndarray:
    """Verified chain: official -> pack (SI flip) -> R3 index -> physical -> iso index."""
    r3 = pack_to_r3(official_to_pack(official, pack["nz"]), pack["ny"], pack["nz"])
    phys = pack["origin_xyz"] + r3 * pack["spacing_xyz"]
    return (phys - pack["origin_iso_xyz"]) / SPACING


def inside(idx: np.ndarray) -> np.ndarray:
    return np.all((idx >= 0.0) & (idx <= (SIZE - 1)), axis=1)


def sample_u(dvf_zyx3: np.ndarray, idx_xyz: np.ndarray) -> np.ndarray:
    pts = np.stack([idx_xyz[:, 2], idx_xyz[:, 1], idx_xyz[:, 0]], axis=0)
    return np.stack(
        [map_coordinates(dvf_zyx3[..., c], pts, order=1, mode="nearest") for c in range(3)],
        axis=1,
    )


def score_set(dvf, case: int, which: str, pack: dict) -> dict:
    loader = landmarks_75 if which == "75" else landmarks_300
    off00, off50 = loader(case, "T00"), loader(case, "T50")
    lm00 = official_to_iso(off00, pack)
    lm50 = official_to_iso(off50, pack)
    keep = inside(lm00) & inside(lm50)
    pred = lm00 + sample_u(dvf, lm00)
    err = np.linalg.norm((pred - lm50) * SPACING, axis=1)
    ident = np.linalg.norm((lm00 - lm50) * SPACING, axis=1)
    return {
        "registered": stats(err),
        "registered_inside": stats(err[keep]) if keep.any() else None,
        "identity_iso_mm": float(ident.mean()),
        "identity_official_mm": float(tre_mm(off00, off50, CASE_INFO[case][1]).mean()),
        "n_inside": int(keep.sum()),
        "n": int(len(err)),
    }


# ---------------------------------------------------------------- round trip
def elastix_on_iso(pack: dict) -> np.ndarray:
    """A1 Elastix DVF_sub_01 (128³ sub-voxels, R3, elastix sign) -> pull u on the iso cube.

    Same unit chain as eval_a1_tre: sub index s = n*128/N; displacement
    sub-voxels * N/128 = native voxels; old rule pred = lm00 - disp,
    so u = -disp.
    """
    disp = load_dvf_zyx3(pack["train"] / "DVF_sub_01.mha")  # (128,128,128,3), comps x,y,z
    n_xyz = np.array([pack["nx"], pack["nz"], pack["ny"]], dtype=np.float64)  # R3 shape
    xx, yy, zz = iso_physical_grid(pack["origin_iso_xyz"])
    o, sp = pack["origin_xyz"], pack["spacing_xyz"]
    sx = (xx - o[0]) / sp[0] * 128.0 / n_xyz[0]
    sy = (yy - o[1]) / sp[1] * 128.0 / n_xyz[1]
    sz = (zz - o[2]) / sp[2] * 128.0 / n_xyz[2]
    pts = np.stack([sz.ravel(), sy.ravel(), sx.ravel()], axis=0)
    u = np.empty((SIZE, SIZE, SIZE, 3), dtype=np.float32)
    for c in range(3):
        d_sub = map_coordinates(disp[..., c], pts, order=1, mode="nearest")
        d_mm = d_sub * n_xyz[c] / 128.0 * sp[c]
        u[..., c] = (-d_mm / SPACING).reshape(SIZE, SIZE, SIZE)
    return u


# ---------------------------------------------------------------- network
def load_generator(device):
    if str(NET) not in sys.path:
        sys.path.insert(0, str(NET))
    from networks.generator_crb_dec import UNetCRBDecoder

    g = UNetCRBDecoder(im_size=SIZE, n_phases=10)
    raw = torch.load(str(CKPT), map_location=device, weights_only=False)
    state = raw["generator"] if isinstance(raw, dict) and "generator" in raw else raw
    g.load_state_dict(state, strict=True)
    return g.to(device).eval()


def predict(g, device, mu: np.ndarray, flip: bool) -> np.ndarray:
    vol = np.ascontiguousarray(mu[:, ::-1, :]) if flip else mu
    x = torch.from_numpy(minmax(vol)[None, None]).to(device)
    ref = torch.tensor([5], dtype=torch.long, device=device)
    tgt = torch.tensor([0], dtype=torch.long, device=device)
    with torch.no_grad():
        dvf = g(x, ref, tgt)[0].cpu().numpy()
    field = np.moveaxis(dvf, 0, -1).astype(np.float32)  # (Z,Y,X,3) comps dx,dy,dz
    if flip:
        field = np.ascontiguousarray(field[:, ::-1, :, :])
        field[..., 1] *= -1.0
    return field


def save_orientation_png(tcia_mu, tcia_lung, dir_mu, dir_lung, dir_flipped_mu):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[png] skipped: {e}", flush=True)
        return
    def cor(v, m):
        z = int(np.round(np.nonzero(m.any(axis=(1, 2)))[0].mean()))
        return v[z]  # (Y, X): Y = SI vertical, row 0 at top
    panels = [
        ("TCIA S1 (train)", cor(tcia_mu, tcia_lung)),
        ("DIR C01 as-is", cor(dir_mu, dir_lung)),
        ("DIR C01 flipped", cor(dir_flipped_mu, dir_lung[:, ::-1, :])),
    ]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.5))
    for a, (t, im) in zip(ax, panels):
        a.imshow(im, cmap="gray", origin="upper")
        a.set_title(t)
        a.set_ylabel("Y index (row 0 at top)")
    fig.suptitle("Mid-coronal: apex must be at the same end as TCIA")
    fig.tight_layout()
    fig.savefig(PNG, dpi=110)
    plt.close(fig)
    print(f"[png] wrote {PNG}", flush=True)


# ---------------------------------------------------------------- main
def main() -> None:
    os.environ.setdefault("DIRLAB_ROOT", str(DIR_EXP / "data" / "dirlab_packs"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tcia_mu = np.load(TCIA_MU)
    tcia_lung = np.load(TCIA_LUNG) > 0
    probe = pack_case(1)

    # ---- 1. orientation from masks
    f_tcia = base_row_fraction(tcia_lung)
    f_dir = base_row_fraction(probe["lung_iso"])
    same_end = (f_tcia > 0.5) == (f_dir > 0.5)
    chosen_flip = not same_end
    print(
        f"[orient] lung base row fraction: TCIA S1 {f_tcia:.2f} | DIR C01 {f_dir:.2f} "
        f"-> {'same end, NO flip' if same_end else 'opposite ends, FLIP'}",
        flush=True,
    )
    save_orientation_png(
        tcia_mu, tcia_lung, probe["mu"], probe["lung_iso"],
        np.ascontiguousarray(probe["mu"][:, ::-1, :]),
    )

    # ---- 2. identity + round-trip gate on C01
    ident = score_set(np.zeros((SIZE, SIZE, SIZE, 3), np.float32), 1, "300", probe)
    print(
        f"[gate] C01 identity300 iso {ident['identity_iso_mm']:.3f} "
        f"official {ident['identity_official_mm']:.3f}",
        flush=True,
    )
    if abs(ident["identity_iso_mm"] - ident["identity_official_mm"]) > 0.05:
        raise SystemExit("identity mismatch — landmark scaling broken")

    roundtrip = {}
    for case in (1, 8):
        pack = probe if case == 1 else pack_case(case)
        u_el = elastix_on_iso(pack)
        new75 = score_set(u_el, case, "75", pack)["registered"]["mean"]
        old75 = tre_t00_t50(load_dvf_zyx3(pack["train"] / "DVF_sub_01.mha"), case, "75", r3=True)[
            "registered"
        ]["mean"]
        roundtrip[case] = {"elastix_new_chain_tre75": new75, "elastix_old_chain_tre75": old75}
        print(f"[gate] C{case:02d} Elastix TRE75 new chain {new75:.3f} | old verified {old75:.3f}", flush=True)
        if abs(new75 - old75) > 0.25:
            raise SystemExit(
                f"round-trip FAILED on C{case:02d} ({new75:.2f} vs {old75:.2f}) — "
                "geometry still wrong, network numbers would be meaningless"
            )

    # ---- 3. network, both orientations
    print("brightness TCIA S1", brightness(tcia_mu, tcia_lung), flush=True)
    print("brightness DIR C01", brightness(probe["mu"], probe["lung_iso"]), flush=True)
    g = load_generator(device)
    results = {False: [], True: []}
    packs = {1: probe}
    for case in range(1, 11):
        pack = packs.get(case) or pack_case(case)
        for flip in (False, True):
            dvf = predict(g, device, pack["mu"], flip)
            row = {"case": case, "75": score_set(dvf, case, "75", pack),
                   "300": score_set(dvf, case, "300", pack)}
            results[flip].append(row)
            print(
                f"C{case:02d} flip={int(flip)} TRE75 {row['75']['registered']['mean']:.2f} "
                f"TRE300 {row['300']['registered']['mean']:.2f} "
                f"id75 {row['75']['identity_official_mm']:.2f} "
                f"inside {row['75']['n_inside']}/{row['75']['n']}",
                flush=True,
            )

    def cohort(rows, which, key="registered"):
        vals = np.array([r[which][key]["mean"] for r in rows if r[which][key]], np.float64)
        return {"mean_mm": float(vals.mean()), "std_mm": float(vals.std(ddof=1)),
                "per_case_mm": vals.tolist()}

    report = {
        "model": "TCIA3 UNetCRBDecoder epoch_100",
        "ckpt": str(CKPT),
        "grid": "A1 R3 CT -> 2 mm, 160³ centred on lung, µ per-scan min-max",
        "landmarks": "official -> pack -> R3 (eval_a1_tre chain) -> physical -> iso",
        "orientation": {"tcia_base_row_frac": f_tcia, "dir_c01_base_row_frac": f_dir,
                        "chosen_flip": chosen_flip, "png": str(PNG)},
        "roundtrip_gate": roundtrip,
        "brightness_tcia_s1": brightness(tcia_mu, tcia_lung),
        "brightness_dir_c01": brightness(probe["mu"], probe["lung_iso"]),
        "when": datetime.now(timezone.utc).astimezone().isoformat(),
    }
    for flip in (False, True):
        tag = f"flip{int(flip)}" + ("_chosen" if flip == chosen_flip else "")
        report[tag] = {
            "tre75": cohort(results[flip], "75"),
            "tre300": cohort(results[flip], "300"),
            "tre75_inside": cohort(results[flip], "75", "registered_inside"),
            "tre300_inside": cohort(results[flip], "300", "registered_inside"),
            "cases": results[flip],
        }
    OUT.write_text(json.dumps(report, indent=2) + "\n")

    print("\n=== cohort (mm) ===")
    for flip in (False, True):
        r = report[[k for k in report if k.startswith(f"flip{int(flip)}")][0]]
        mark = "  <- chosen by mask check" if flip == chosen_flip else ""
        print(f"flip={int(flip)}  TRE75 {r['tre75']['mean_mm']:.3f}±{r['tre75']['std_mm']:.3f}  "
              f"TRE300 {r['tre300']['mean_mm']:.3f}±{r['tre300']['std_mm']:.3f}  "
              f"(inside-only 300: {r['tre300_inside']['mean_mm']:.3f}){mark}")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
