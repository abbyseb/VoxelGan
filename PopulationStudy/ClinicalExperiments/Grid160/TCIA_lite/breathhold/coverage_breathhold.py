"""Coverage QC for the converted breath-hold pairs: where does real CT data exist in the 160^3 grid (vs -1000 HU fill
outside the native scan extent)? Writes {B}_Cover.npy = data present in BOTH phases (R3 layout, uint8) and
coverage.json with the fraction of the union lung mask that is covered; pairs below 0.90 are flagged for exclusion.
  python3 coverage_breathhold.py OUT_DIR"""
import json, sys
from pathlib import Path
import numpy as np, SimpleITK as sitk
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pack_common as P
src = open(Path(__file__).resolve().parent / "convert_breathhold.py").read(); src = src[:src.index("index = []")]
sys.argv = [sys.argv[0], sys.argv[1]]; exec(src)
out, rows = Path(sys.argv[1]), []


def cover(img, ref):
    one = sitk.Image(img.GetSize(), sitk.sitkUInt8) + 1; one.CopyInformation(img)
    return P.to_r3((P.resample(one, ref, sitk.sitkNearestNeighbor, 0.0) > 0.5).astype(np.uint8))


for k, (src_, name, load) in enumerate(pairs(), start=1):
    B = f"B{k:02d}"
    ins, exp, mi, me = load()
    ref = P.grid(P.lung_centroid_zyx(me) - 160.0)
    c = cover(ins, ref) & cover(exp, ref); np.save(out / f"{B}_Cover.npy", c)
    m = np.load(out / f"{B}_Mask_Lung.npy") > 0
    frac = float((m & (c > 0)).sum() / max(m.sum(), 1))
    rows.append({"id": B, "source": src_, "name": name, "lung_covered_frac": frac})
    print(f"{B} {name} lung covered {frac:.3f}", flush=True)
info = json.loads((out / "index.json").read_text())
for r, i in zip(rows, info):
    i["lung_covered_frac"] = r["lung_covered_frac"]
    i["exclude"] = bool(r["lung_covered_frac"] < 0.90 or i["lung_ml_change"] < 0)
(out / "index.json").write_text(json.dumps(info, indent=1))
print("excluded", [i["id"] for i in info if i["exclude"]])
