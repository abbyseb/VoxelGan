#!/usr/bin/env python3
"""Four independent checks of the corrected (v2) DIR TRE path.

  1. Round trip, all 10 cases, 75 + 300 landmarks: A1 Elastix field moved onto
     the 2 mm cube and scored with the new code must match the old verified
     evaluator (eval_a1_tre) within TOL_MM.
  2. Per-channel agreement: Pearson r between the network field and Elastix,
     separately for left-right (dx), head-foot (dy), front-back (dz), inside
     the lung. All three must be > 0.
  3. Image check (no landmarks): warp CT_06 with the network field and compare
     with the real CT_01 on the same cube. NCC in the lung must beat the
     unwarped CT_06 (identity) on every case. Elastix-warped NCC is reported
     as the reference ceiling.
  4. Reverse direction: T50 -> T00 with the usual small-motion approximation
     (pred = lm50 - u(lm50)). For Elastix it must match the old evaluator's
     tre_t50_t00; for the network it is reported next to forward TRE.

Writes nothing except the JSON below. Weights are only read.

  cd "DIR EXPERIMENTS"
  CUDA_VISIBLE_DEVICES=1 python scripts/verify_iso2mm.py
  CUDA_VISIBLE_DEVICES=1 python scripts/verify_iso2mm.py --ckpt <path/to/other.pt>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from scipy.ndimage import binary_dilation, map_coordinates

DIR_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIR_EXP / "scripts"))
import eval_dir_tcia3_iso2mm_v2 as v2  # noqa: E402
from eval_a1_tre import load_dvf_zyx3, tre_t00_t50, tre_t50_t00  # noqa: E402

TOL_MM = 0.02


def score_reverse(dvf, case: int, which: str, pack: dict) -> float:
    """T50 -> T00, pred = lm50 - u(lm50) (same approximation as eval_a1_tre)."""
    loader = v2.landmarks_75 if which == "75" else v2.landmarks_300
    off00, off50 = loader(case, "T00"), loader(case, "T50")
    lm00 = v2.official_to_iso(off00, pack)
    lm50 = v2.official_to_iso(off50, pack)
    pred = lm50 - v2.sample_u(dvf, lm50)
    return float(np.linalg.norm((pred - lm00) * v2.SPACING, axis=1).mean())


def invert_at_points(dvf, targets: np.ndarray, iters: int = 50, tol: float = 1e-4):
    """Exact inverse at points: find x with x + u(x) = target (fixed-point iteration).

    u is the pull field defined on the T00 grid (x in T00 -> x + u(x) in T50).
    For a T50 landmark y we solve x = y - u(x). Converges when the field is
    smooth (|grad u| < 1). Returns x and the final residual |x + u(x) - y| in voxels.
    """
    x = targets - v2.sample_u(dvf, targets)  # the old shortcut, used as the start
    for _ in range(iters):
        x_new = targets - v2.sample_u(dvf, x)
        if np.max(np.abs(x_new - x)) < tol:
            x = x_new
            break
        x = x_new
    resid = np.linalg.norm(x + v2.sample_u(dvf, x) - targets, axis=1)
    return x, resid


def score_reverse_exact(dvf, case: int, which: str, pack: dict) -> tuple[float, float]:
    """T50 -> T00 done properly: invert the field at each T50 landmark."""
    loader = v2.landmarks_75 if which == "75" else v2.landmarks_300
    off00, off50 = loader(case, "T00"), loader(case, "T50")
    lm00 = v2.official_to_iso(off00, pack)
    lm50 = v2.official_to_iso(off50, pack)
    pred, resid = invert_at_points(dvf, lm50)
    tre = float(np.linalg.norm((pred - lm00) * v2.SPACING, axis=1).mean())
    return tre, float(resid.max())


def ncc(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    a = a[m].astype(np.float64)
    b = b[m].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def warp_pull(vol_zyx: np.ndarray, u_zyx3: np.ndarray) -> np.ndarray:
    """out(x) = vol(x + u(x)); u channels (dx, dy, dz) in iso voxels."""
    zz, yy, xx = np.meshgrid(*(np.arange(s, dtype=np.float32) for s in vol_zyx.shape), indexing="ij")
    coords = np.stack([zz + u_zyx3[..., 2], yy + u_zyx3[..., 1], xx + u_zyx3[..., 0]], axis=0)
    return map_coordinates(vol_zyx, coords, order=1, mode="nearest").astype(np.float32)


def ct01_on_cube(pack: dict) -> np.ndarray:
    img = sitk.ReadImage(str(pack["train"] / "CT_01.mha"))
    if not np.allclose(img.GetSpacing(), pack["spacing_xyz"]) or not np.allclose(
        img.GetOrigin(), pack["origin_xyz"]
    ):
        raise RuntimeError("CT_01 and CT_06 grids differ")
    hu = sitk.GetArrayFromImage(img).astype(np.float32)
    return v2.resample_to_iso(hu, pack["origin_xyz"], pack["spacing_xyz"], pack["origin_iso_xyz"], 1, -1000.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=v2.CKPT)
    ap.add_argument("--cases", type=int, nargs="*", default=list(range(1, 11)))
    args = ap.parse_args()

    os.environ.setdefault("DIRLAB_ROOT", str(DIR_EXP / "data" / "dirlab_packs"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    v2.CKPT = args.ckpt.resolve()
    g = v2.load_generator(device)

    rows = []
    fails = []
    for case in args.cases:
        pack = v2.pack_case(case)
        el_old = load_dvf_zyx3(pack["train"] / "DVF_sub_01.mha")
        u_el = v2.elastix_on_iso(pack)
        u_net = v2.predict(g, device, pack["mu"], flip=False)
        r = {"case": case}

        # 1. round trip, both sets, both directions
        rt = {}
        for which in ("75", "300"):
            new_f = v2.score_set(u_el, case, which, pack)["registered"]["mean"]
            old_f = tre_t00_t50(el_old, case, which, r3=True)["registered"]["mean"]
            new_r = score_reverse(u_el, case, which, pack)
            old_r = tre_t50_t00(el_old, case, which, r3=True)["registered"]["mean"]
            rt[which] = {"fwd_new": new_f, "fwd_old": old_f, "rev_new": new_r, "rev_old": old_r}
            if abs(new_f - old_f) > TOL_MM:
                fails.append(f"C{case:02d} round-trip fwd {which}: {new_f:.3f} vs {old_f:.3f}")
            if abs(new_r - old_r) > TOL_MM:
                fails.append(f"C{case:02d} round-trip rev {which}: {new_r:.3f} vs {old_r:.3f}")
        r["roundtrip"] = rt

        # 2. per-channel Pearson r in the lung
        lung = pack["lung_iso"]
        chans = {}
        for c, name in enumerate(("dx_LR", "dy_SI", "dz_AP")):
            a, b = u_net[..., c][lung], u_el[..., c][lung]
            chans[name] = float(np.corrcoef(a, b)[0, 1])
            if not chans[name] > 0:
                fails.append(f"C{case:02d} channel {name} r={chans[name]:.3f} ≤ 0")
        r["channel_r"] = chans

        # 3. image check (lung + small margin)
        m = binary_dilation(lung, iterations=3)
        ct06 = pack["hu_iso"]
        ct01 = ct01_on_cube(pack)
        img = {
            "ncc_identity": ncc(ct06, ct01, m),
            "ncc_network": ncc(warp_pull(ct06, u_net), ct01, m),
            "ncc_elastix": ncc(warp_pull(ct06, u_el), ct01, m),
        }
        if not img["ncc_network"] > img["ncc_identity"]:
            fails.append(f"C{case:02d} image NCC network {img['ncc_network']:.4f} ≤ identity {img['ncc_identity']:.4f}")
        r["image"] = img

        # 4. reverse: shortcut (as the old evaluator) and exact inversion
        r["network"] = {}
        r["elastix_rev_exact"] = {}
        for w in ("75", "300"):
            ex_net, res_net = score_reverse_exact(u_net, case, w, pack)
            ex_el, res_el = score_reverse_exact(u_el, case, w, pack)
            r["network"][w] = {
                "fwd": v2.score_set(u_net, case, w, pack)["registered"]["mean"],
                "rev": score_reverse(u_net, case, w, pack),
                "rev_exact": ex_net,
                "rev_exact_max_resid_vox": res_net,
            }
            r["elastix_rev_exact"][w] = {"tre": ex_el, "max_resid_vox": res_el}
            if max(res_net, res_el) > 0.01:
                fails.append(f"C{case:02d} inversion residual {w}: net {res_net:.3g} El {res_el:.3g} vox")
        rows.append(r)
        print(
            f"C{case:02d} | RT300 fwd {rt['300']['fwd_new']:.3f}/{rt['300']['fwd_old']:.3f} "
            f"rev {rt['300']['rev_new']:.3f}/{rt['300']['rev_old']:.3f} | "
            f"r LR {chans['dx_LR']:+.2f} SI {chans['dy_SI']:+.2f} AP {chans['dz_AP']:+.2f} | "
            f"NCC id {img['ncc_identity']:.4f} net {img['ncc_network']:.4f} El {img['ncc_elastix']:.4f} | "
            f"net300 fwd {r['network']['300']['fwd']:.2f} rev {r['network']['300']['rev']:.2f} "
            f"rev_exact {r['network']['300']['rev_exact']:.2f} | "
            f"El300 fwd {rt['300']['fwd_new']:.2f} rev_exact {r['elastix_rev_exact']['300']['tre']:.2f}",
            flush=True,
        )

    ok = not fails
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = v2.OUT_DIR / f"verify_iso2mm_{stamp}.json"
    out.write_text(json.dumps({
        "ckpt": str(v2.CKPT),
        "tolerance_mm": TOL_MM,
        "passed": ok,
        "failures": fails,
        "cases": rows,
        "when": datetime.now(timezone.utc).astimezone().isoformat(),
    }, indent=2) + "\n")

    print("\n================ VERIFY ================")
    print("1 round trip (all cases, 75+300, fwd+rev):", "PASS" if not any("round-trip" in f for f in fails) else "FAIL")
    print("2 per-channel r > 0:                      ", "PASS" if not any("channel" in f for f in fails) else "FAIL")
    print("3 image NCC beats identity:               ", "PASS" if not any("image" in f for f in fails) else "FAIL")
    fwd = np.mean([r["network"]["300"]["fwd"] for r in rows])
    rev = np.mean([r["network"]["300"]["rev"] for r in rows])
    rex = np.mean([r["network"]["300"]["rev_exact"] for r in rows])
    elf = np.mean([r["roundtrip"]["300"]["fwd_new"] for r in rows])
    elx = np.mean([r["elastix_rev_exact"]["300"]["tre"] for r in rows])
    print(f"4 network TRE300 fwd {fwd:.3f} | rev shortcut {rev:.3f} | rev EXACT {rex:.3f}")
    print(f"  Elastix TRE300 fwd {elf:.3f} | rev EXACT {elx:.3f}")
    print("  inversion residual < 0.01 vox:          ", "PASS" if not any("inversion" in f for f in fails) else "FAIL")
    for f in fails:
        print("  FAIL:", f)
    print("OVERALL:", "PASS" if ok else "FAIL")
    print("wrote", out)


if __name__ == "__main__":
    main()
