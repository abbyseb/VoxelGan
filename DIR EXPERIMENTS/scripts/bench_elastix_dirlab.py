#!/usr/bin/env python3
"""Benchmark Elastix settings on DIR-Lab 4DCT (T00 -> T50), native resolution, real spacing.

Configs (same data, same scoring):
  old        My v1.0/configs/elastix_bspline_masked.txt as is (grid 16 mm, 3 levels,
             2048 samples, MI, no bending penalty) - but now with REAL spacing.
  new_mi     4 levels, final grid 6 mm, 10k samples, 2000 it, MI + bending energy 0.05.
  new_ncc    same as new_mi with AdvancedNormalizedCorrelation.

Registration: fixed = T00 (CT_01), moving = T50 (CT_06), fixed lung mask only
(body-based lung mask of CT_01, dilated 3 voxels). Elastix maps a fixed point x to
x + d(x) in the moving image, so pred_T50 = lm00 + d(lm00). Landmarks go through the
verified chain official -> pack -> R3 index -> physical mm on the A1 R3 volumes.
Gate: identity TRE must equal the official identity TRE.

Also reports the fraction of folded voxels (Jacobian <= 0) in the lung.
Reference: current label pipeline (A1 Elastix field) = 1.96 mm TRE300 (verify_iso2mm).

  cd "DIR EXPERIMENTS"
  /home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python scripts/bench_elastix_dirlab.py
  # quicker first look:  --cases 1 8  --configs old new_mi
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, map_coordinates

DIR_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIR_EXP / "scripts"))
from dirlab_tre import CASE_INFO, landmarks_300, landmarks_75, tre_mm  # noqa: E402
from eval_a1_tre import official_to_pack, pack_to_r3  # noqa: E402

A1 = DIR_EXP / "arms" / "A1_oracle_dirlab" / "runs"
OLD_PARAM = DIR_EXP.parent / "My v1.0" / "configs" / "elastix_bspline_masked.txt"
OUT = DIR_EXP.parent / "PopulationStudy/ClinicalExperiments/Grid160/TCIA3/DecoderCRB/plots/qc_dir_oracle"


# ---------------------------------------------------------------- lung mask (body-based)
def lung_mask(hu: np.ndarray, axial_axis: int) -> np.ndarray:
    from scipy import ndimage

    lab, n = ndimage.label(hu > -400.0)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    body = lab == int(np.argmax(counts))
    filled = np.zeros_like(body)
    for k in range(body.shape[axial_axis]):
        sl = [slice(None)] * 3
        sl[axial_axis] = k
        filled[tuple(sl)] = ndimage.binary_fill_holes(body[tuple(sl)])
    lab, n = ndimage.label((hu < -400.0) & filled & ~body)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    keep = np.zeros(n + 1, bool)
    for i in np.argsort(counts)[::-1][:2]:
        if i and counts[i] > 0.1 * counts.max():
            keep[i] = True
    return keep[lab]


# ---------------------------------------------------------------- parameter maps
def param_object(name: str):
    import itk

    po = itk.ParameterObject.New()
    if name == "old":
        po.AddParameterFile(str(OLD_PARAM))
        return po
    p = po.GetDefaultParameterMap("bspline", 4, 6.0)
    metric = "AdvancedMattesMutualInformation" if name == "new_mi" else "AdvancedNormalizedCorrelation"
    p["Registration"] = ["MultiMetricMultiResolutionRegistration"]
    p["Metric"] = [metric, "TransformBendingEnergyPenalty"]
    p["Metric0Weight"] = ["1.0"]
    p["Metric1Weight"] = ["0.05"]
    p["NumberOfResolutions"] = ["4"]
    p["FinalGridSpacingInPhysicalUnits"] = ["6.0"]
    p["GridSpacingSchedule"] = ["8", "4", "2", "1"]
    p["MaximumNumberOfIterations"] = ["2000"]
    p["NumberOfSpatialSamples"] = ["10000"]
    p["ImageSampler"] = ["RandomCoordinate"]
    p["NewSamplesEveryIteration"] = ["true"]
    p["NumberOfHistogramBins"] = ["32"]
    p["Optimizer"] = ["AdaptiveStochasticGradientDescent"]
    p["FixedImagePyramid"] = ["FixedSmoothingImagePyramid"]
    p["MovingImagePyramid"] = ["MovingSmoothingImagePyramid"]
    p["WriteResultImage"] = ["false"]
    p["ErodeMask"] = ["false"]
    po.AddParameterMap(p)
    return po


# ---------------------------------------------------------------- one case
def run_case(case: int, config: str) -> dict:
    import itk

    sid = f"DIR_C{case:02d}"
    train = A1 / sid / sid / "train"
    fixed = itk.imread(str(train / "CT_01.mha"), itk.F)   # T00
    moving = itk.imread(str(train / "CT_06.mha"), itk.F)  # T50
    origin = np.array(fixed.GetOrigin(), float)
    spacing = np.array(fixed.GetSpacing(), float)
    hu = itk.array_from_image(fixed)
    # A1 R3 arrays are numpy (A, S, L): axial slices are along axis 1
    m = binary_dilation(lung_mask(hu, axial_axis=1), iterations=3).astype(np.uint8)
    mask = itk.image_from_array(m)
    mask.CopyInformation(fixed)

    t0 = time.time()
    reg = itk.ElastixRegistrationMethod.New(fixed, moving)
    reg.SetParameterObject(param_object(config))
    reg.SetFixedMask(mask)
    reg.SetLogToConsole(False)
    reg.Update()
    field = itk.transformix_deformation_field(moving, reg.GetTransformParameterObject())
    secs = time.time() - t0
    d = itk.array_from_image(field).astype(np.float64)   # (Z,Y,X,3), comps x,y,z in mm

    (nx, ny, nz), sp_off = CASE_INFO[case]
    out = {"case": case, "config": config, "seconds": secs}
    for which, loader in (("75", landmarks_75), ("300", landmarks_300)):
        l00, l50 = loader(case, "T00"), loader(case, "T50")

        def to_phys(off):
            r3 = pack_to_r3(official_to_pack(off, nz), ny, nz)  # itk index (x,y,z)
            return origin + r3 * spacing

        p00, p50 = to_phys(l00), to_phys(l50)
        idx = (p00 - origin) / spacing
        pts = np.stack([idx[:, 2], idx[:, 1], idx[:, 0]])
        disp = np.stack([map_coordinates(d[..., c], pts, order=1, mode="nearest") for c in range(3)], 1)
        pred = p00 + disp
        ident_new = float(np.linalg.norm(p00 - p50, axis=1).mean())
        ident_off = float(tre_mm(l00, l50, sp_off).mean())
        if abs(ident_new - ident_off) > 0.05:
            raise SystemExit(f"C{case:02d} identity mismatch {ident_new:.3f} vs {ident_off:.3f}")
        out[f"tre{which}"] = float(np.linalg.norm(pred - p50, axis=1).mean())
        out[f"identity{which}"] = ident_off

    # folding: Jacobian determinant of x + d(x) inside the lung
    g = [np.gradient(d[..., c], spacing[2], spacing[1], spacing[0]) for c in range(3)]  # d/dz,d/dy,d/dx
    J = np.empty(d.shape[:3] + (3, 3))
    for c in range(3):          # component x,y,z
        J[..., c, 0] = g[c][2]  # d/dx
        J[..., c, 1] = g[c][1]  # d/dy
        J[..., c, 2] = g[c][0]  # d/dz
    J += np.eye(3)
    det = np.linalg.det(J[m > 0])
    out["fold_frac_lung"] = float((det <= 0).mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=int, nargs="*", default=list(range(1, 11)))
    ap.add_argument("--configs", nargs="*", default=["old", "new_mi", "new_ncc"])
    args = ap.parse_args()
    os.environ.setdefault("DIRLAB_ROOT", str(DIR_EXP / "data" / "dirlab_packs"))

    rows = []
    for cfg in args.configs:
        for c in args.cases:
            r = run_case(c, cfg)
            rows.append(r)
            print(f"{cfg:8s} C{c:02d} TRE75 {r['tre75']:.2f}  TRE300 {r['tre300']:.2f}  "
                  f"(id {r['identity300']:.2f})  folds {100*r['fold_frac_lung']:.3f}%  {r['seconds']/60:.1f} min",
                  flush=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT / f"bench_elastix_dirlab_{stamp}.json"
    out.write_text(json.dumps({"when": datetime.now(timezone.utc).astimezone().isoformat(),
                               "rows": rows}, indent=2) + "\n")
    print("\n========= Elastix on DIR-Lab, T00->T50 (mm) =========")
    print("reference: current label pipeline (A1 Elastix field) TRE300 1.96")
    for cfg in args.configs:
        rr = [r for r in rows if r["config"] == cfg]
        print(f"{cfg:8s} TRE75 {np.mean([r['tre75'] for r in rr]):.3f}  "
              f"TRE300 {np.mean([r['tre300'] for r in rr]):.3f}  "
              f"max folds {100*max(r['fold_frac_lung'] for r in rr):.3f}%  "
              f"mean time {np.mean([r['seconds'] for r in rr])/60:.1f} min  (n={len(rr)})")
    print("wrote", out)


if __name__ == "__main__":
    main()
