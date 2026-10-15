"""Convert breath-hold inspiration/expiration CT pairs to the TCIA training format (pack_common.py recipe, validated to
reproduce synth_g160_r3/pooled/S01_CT_06.npy exactly).
  L2R-LungCT (Zenodo 4279348, CC-BY-4.0, 30 pairs, lung masks provided)            -> B01..B30
  TCIA CT-vs-PET-Ventilation-Imaging (CC-BY-4.0, 20 pairs, 'Thorax Insp/Exp 2.0 B70f') -> B31..B50 (lung masks by threshold)
Per scan: {B}_CT_01.npy = inspiration (phase index 0, like DIR-Lab T00), {B}_CT_06.npy = expiration (index 5, like T50),
{B}_Mask_Lung.npy = union of the two lung masks; grid centred on the EXPIRATION lung centroid (the reference scan).
Also writes index.json (source, sizes, lung volumes ml, mean HU in lung) and a QC figure per scan.
  python3 convert_breathhold.py OUT_DIR"""
import json, sys, glob, re
from pathlib import Path
import numpy as np, SimpleITK as sitk
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pack_common as P

LP = Path("/media/abhishek/3CCA3CADCA3C6574/LungPairs")
OUT = Path(sys.argv[1]); (OUT / "qc").mkdir(parents=True, exist_ok=True)


def threshold_lung(img):
    """Lungs = air (< -320 HU) INSIDE THE BODY. Body = largest 3-D component of tissue (> -400 HU) with holes filled slice by
    slice (so couch cushions / blankets / outside air are excluded). Then the 2 largest air components (>= 50 ml) inside the
    body, small closing, hole filling."""
    from scipy import ndimage as ndi
    a = sitk.GetArrayFromImage(img)                                   # (z, y, x) HU
    tis = a > -400
    tis = ndi.binary_opening(tis, structure=np.ones((1, 7, 7), bool))   # cut thin couch / cushion shells touching the body
    lab, n = ndi.label(tis)
    body = lab == (np.argmax(ndi.sum(tis, lab, range(1, n + 1))) + 1) if n else tis
    body = np.stack([ndi.binary_fill_holes(sl) for sl in body])
    air = (a < -320) & body
    lab, n = ndi.label(air)
    vox_ml = np.prod(img.GetSpacing()) / 1000
    sizes = ndi.sum(air, lab, range(1, n + 1)) * vox_ml
    keep = [i + 1 for i in np.argsort(sizes)[::-1][:2] if sizes[i] >= 50]
    m = np.isin(lab, keep).astype(np.uint8)
    mi = sitk.GetImageFromArray(m); mi.CopyInformation(img)
    return sitk.BinaryFillhole(sitk.BinaryMorphologicalClosing(mi, [3, 3, 1]))


def read_dicom(d):
    r = sitk.ImageSeriesReader(); r.SetFileNames(r.GetGDCMSeriesFileNames(str(d))); return sitk.Cast(r.Execute(), sitk.sitkFloat32)


def pairs():
    sc = LP / "L2R_LungCT" / "selectedCases_nii"
    for f in sorted(glob.glob(str(sc / "scans" / "case_*_exp.nii.gz")), key=lambda s: int(re.search(r"case_(\d+)_", s).group(1))):
        n = Path(f).name.replace("_exp.nii.gz", "")
        yield ("L2R-LungCT", n, lambda n=n: (sitk.ReadImage(str(sc / "scans" / f"{n}_insp.nii.gz"), sitk.sitkFloat32),
                                             sitk.ReadImage(str(sc / "scans" / f"{n}_exp.nii.gz"), sitk.sitkFloat32),
                                             sitk.ReadImage(str(sc / "lungMasks" / f"{n}_insp.nii.gz")), sitk.ReadImage(str(sc / "lungMasks" / f"{n}_exp.nii.gz"))))
    for d in sorted((LP / "TCIA_Ventilation").glob("CT-PET-VI-*")):
        def load(d=d):
            i, e = read_dicom(d / "insp"), read_dicom(d / "exp"); return i, e, threshold_lung(i), threshold_lung(e)
        yield ("TCIA-Ventilation", d.name, load)


index = []
for k, (src, name, load) in enumerate(pairs(), start=1):
    B = f"B{k:02d}"
    if (OUT / f"{B}_Mask_Lung.npy").exists():
        index.append(json.loads((OUT / "qc" / f"{B}.json").read_text())); continue
    ins, exp, mi, me = load()
    ref = P.grid(P.lung_centroid_zyx(me) - 160.0)
    ci, ce = P.pack_ct(ins, ref), P.pack_ct(exp, ref)
    m = np.maximum(P.pack_mask(mi, ref), P.pack_mask(me, ref))
    np.save(OUT / f"{B}_CT_01.npy", ci); np.save(OUT / f"{B}_CT_06.npy", ce); np.save(OUT / f"{B}_Mask_Lung.npy", m)
    vol = lambda mk: float(sitk.GetArrayFromImage(mk).astype(bool).sum() * np.prod(mk.GetSpacing()) / 1000)
    hu_l = lambda c, mk: float((c[mk > 0] / 2e-5 - 1000).mean())
    rec = {"id": B, "source": src, "name": name, "native_spacing": list(exp.GetSpacing()), "native_size": list(exp.GetSize()),
           "lung_ml_insp": vol(mi), "lung_ml_exp": vol(me), "lung_ml_change": vol(mi) - vol(me),
           "lung_mean_HU_insp": hu_l(ci, P.pack_mask(mi, ref)), "lung_mean_HU_exp": hu_l(ce, P.pack_mask(me, ref)),
           "mask_vox_in_grid": int(m.sum()), "mask_touches_grid_edge": bool(m[[0, -1]].any() or m[:, [0, -1]].any() or m[:, :, [0, -1]].any())}
    (OUT / "qc" / f"{B}.json").write_text(json.dumps(rec, indent=1)); index.append(rec)
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    for r, (c, t) in enumerate(((ci, "insp (CT_01)"), (ce, "exp (CT_06)"))):
        for j, sl in enumerate((c[80], c[:, 80], c[:, :, 80])):
            ax[r, j].imshow(sl, cmap="gray"); ax[r, j].contour([m[80], m[:, 80], m[:, :, 80]][j], [0.5], colors="r", linewidths=0.5)
            ax[r, j].set_title(f"{B} {t} {['axis0 (AP)', 'axis1 (SI)', 'axis2 (LR)'][j]}"); ax[r, j].axis("off")
    fig.suptitle(f"{B} = {src} {name} | lung {rec['lung_ml_exp']:.0f} -> {rec['lung_ml_insp']:.0f} ml"); fig.tight_layout()
    fig.savefig(OUT / "qc" / f"{B}.png", dpi=70); plt.close(fig)
    print(f"{B} {src} {name} lung exp {rec['lung_ml_exp']:.0f} ml insp {rec['lung_ml_insp']:.0f} ml change {rec['lung_ml_change']:.0f} | edge {rec['mask_touches_grid_edge']}", flush=True)
(OUT / "index.json").write_text(json.dumps(index, indent=1))
print("done", len(index))
