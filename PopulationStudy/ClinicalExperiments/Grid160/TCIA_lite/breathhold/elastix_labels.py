"""Elastix motion labels for packed 160^3 R3 pairs, in the training-label format ({S}_{a}_to_{b}_pair.npy, (160,160,160,3)
float32, 2 mm voxels, channels (x, y, z) = (axis2, axis1, axis0); pull field target(p) = ref(p + u(p))).
Fixed = target (phase b), moving = reference (phase a), B-spline parameters of the DIR-Lab oracle
(DIR EXPERIMENTS/arms/A1_oracle_dirlab/Elastix_BSpline_DIR_masked.txt), fixed lung mask.
  python3 elastix_labels.py validate        # S01 06->01 vs the stored TCIA label
  python3 elastix_labels.py run OUT_DIR     # all usable breath-hold pairs (06->01 and 01->06; self pairs = zero)"""
import json, sys, time
from pathlib import Path
import numpy as np, itk

PARAM = "/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/arms/A1_oracle_dirlab/Elastix_BSpline_DIR_masked.txt"
TCIA = Path("/media/abhishek/3CCA3CADCA3C6574/TCIA_4D-Lung/synth_g160_r3/pooled/train")


def img(a):
    im = itk.image_from_array(np.ascontiguousarray(a.astype(np.float32))); im.SetSpacing([2.0] * 3); im.SetOrigin([0.0] * 3); return im


def label(ref, tgt, mask):
    po = itk.ParameterObject.New(); po.AddParameterFile(PARAM)
    m = itk.image_from_array(np.ascontiguousarray(mask.astype(np.uint8))); m.SetSpacing([2.0] * 3); m.SetOrigin([0.0] * 3)
    _, tp = itk.elastix_registration_method(img(tgt), img(ref), parameter_object=po, fixed_mask=m, log_to_console=False)
    f = itk.transformix_deformation_field(img(tgt), tp)                 # (z,y,x,3) mm, components (x,y,z)
    return (np.asarray(f, dtype=np.float32) / 2.0).astype(np.float32)   # -> 2 mm voxels


def consistency(ref, tgt, u, mask):
    import torch, torch.nn.functional as F
    r = torch.from_numpy(ref)[None, None]; d = torch.from_numpy(np.moveaxis(u, -1, 0).copy())[None]
    D, H, W = ref.shape; zz, yy, xx = torch.meshgrid(*(torch.arange(n, dtype=torch.float32) for n in (D, H, W)), indexing="ij")
    g = torch.stack([2 * (xx + d[:, 0]) / (W - 1) - 1, 2 * (yy + d[:, 1]) / (H - 1) - 1, 2 * (zz + d[:, 2]) / (D - 1) - 1], -1)
    w = F.grid_sample(r, g, mode="bilinear", padding_mode="border", align_corners=True)[0, 0].numpy()
    m = mask > 0
    return float(np.abs(tgt - w)[m].mean()), float(np.abs(tgt - ref)[m].mean())


if __name__ == "__main__":
    if sys.argv[1] == "validate":
        mm = lambda a: (a - a.min()) / (a.max() - a.min())
        ref, tgt = mm(np.load(TCIA / "S01_CT_06.npy")), mm(np.load(TCIA / "S01_CT_01.npy")); mask = np.load(TCIA / "S01_Mask_Lung.npy")
        t0 = time.time(); u = label(ref, tgt, mask); el = time.time() - t0
        old = np.load(TCIA / "S01_06_to_01_pair.npy")
        m = mask > 0
        print(f"elastix {el:.0f} s | new vs stored label: mean |diff| in lung {np.linalg.norm(u - old, axis=-1)[m].mean() * 2:.2f} mm, "
              f"mean |stored| {np.linalg.norm(old, axis=-1)[m].mean() * 2:.2f} mm, mean |new| {np.linalg.norm(u, axis=-1)[m].mean() * 2:.2f} mm")
        print("consistency |tgt - warp(ref,u)| new %.4f (no motion %.4f) | stored %.4f" % (*consistency(ref, tgt, u, mask), consistency(ref, tgt, old, mask)[0]))
    else:
        out = Path(sys.argv[2]); I = json.loads((out / "index.json").read_text()); log = []
        for i in [i for i in I if not i["exclude"]]:
            B = i["id"]; mask = np.load(out / f"{B}_Mask_Lung.npy")
            c = {p: np.load(out / f"{B}_CT_{p:02d}.npy") for p in (1, 6)}
            mmn = {p: (a - a.min()) / (a.max() - a.min()) for p, a in c.items()}
            for a, b in ((6, 1), (1, 6)):
                f = out / f"{B}_{a:02d}_to_{b:02d}_pair.npy"
                if f.exists(): continue
                t0 = time.time(); u = label(mmn[a], mmn[b], mask); np.save(f, u)
                e, e0 = consistency(mmn[a], mmn[b], u, mask)
                log.append({"pair": f.name, "s": time.time() - t0, "consistency": e, "no_motion": e0, "mean_motion_mm": float(np.linalg.norm(u, axis=-1)[mask > 0].mean() * 2)})
                print(log[-1], flush=True)
            for p in (1, 6):
                np.save(out / f"{B}_{p:02d}_to_{p:02d}_pair.npy", np.zeros((160, 160, 160, 3), np.float32))
        (out / "elastix_labels.json").write_text(json.dumps(log, indent=1))
