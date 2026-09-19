#!/usr/bin/env python3
"""A3 DIR case: mid-phase CT → G160-A1 synth 4D → DRR → synth-DVF labels → prep_train.

- Synth input: HU→µ then G160 Decoder (same as VoxelMap_Experiments synth_g160_a1_dec)
- Warped CTs stay in HU for DIR-matched DRRs
- Geometry: Geometry_SPARE.xml (OffsetY −2), same as R3 A1
- Source CT/masks: A1 R3 train volumes (not native staged GTVol symlinks)
- Downsample: keep-HU (no LEARN clip)

  cd "DIR EXPERIMENTS"
  LEARN-GUI/.venv/bin/python scripts/prepare_a3_dir_case.py --case 1 --gpu 1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

DIR_EXP = Path(__file__).resolve().parents[1]
VOXEL_GAN = DIR_EXP.parent
VMC = Path(os.environ.get("VOXELMAP_CLINICAL_ROOT", "/home/abhishek/Documents/VoxelMap_Clinical"))
LEARN = Path(os.environ.get("LEARN_GUI_ROOT", "/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python"))
A1 = DIR_EXP / "arms" / "A1_oracle_dirlab"
A3 = DIR_EXP / "arms" / "A3_synth_conditioned"
GEOMETRY_SPARE = LEARN / "modules" / "drr_generation" / "Geometry_SPARE.xml"
G160_CKPT = (
    VOXEL_GAN
    / "PopulationStudy"
    / "ClinicalExperiments"
    / "Grid160"
    / "Experiment1"
    / "DecoderCRB"
    / "weights"
    / "crb_dec_mse_iso_g160_fov_full_generator.pth"
)
G160_NET = VOXEL_GAN / "PopulationStudy" / "InitialExperiments" / "Experiment6"

SYNTH_SPEC = {
    "synth_id": "synth_g160_a1_dec",
    "label": "G160-A1 Decoder · HU→µ · linear",
    "ckpt": G160_CKPT,
    "net_root": G160_NET,
    "arch": "decoder",
    "norm": "mu",
    "im_size": 64,
    "infer_size": 160,
}


def scan_id_for_case(case: int) -> str:
    return f"DIR_C{case:02d}"


def hu_to_mu(hu: np.ndarray, mu_water: float = 0.02) -> np.ndarray:
    """Standard linear HU→µ_water mapping (air HU=-1000 → 0)."""
    return (hu.astype(np.float32) + 1000.0) * (mu_water / 1000.0)


def spare_mu_p1_p99() -> tuple[float, float]:
    """SPARE GTVol µ p1/p99 from staged µ volumes (fallback to ablation constants)."""
    vals = []
    root = VOXEL_GAN / "VoxelMap_Experiments" / "data" / "staged"
    if root.is_dir():
        for p in root.glob("**/GTVol_06.mha"):
            a = sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.float32)
            if float(a.max()) < 1.0:
                vals.append(a.ravel())
    if not vals:
        return -0.0344, 0.0423
    pool = np.concatenate(vals)
    p1, p99 = np.percentile(pool, [1.0, 99.0])
    return float(p1), float(p99)


def dir_hu_to_mu_for_synth(hu: np.ndarray, mode: str) -> tuple[np.ndarray, dict]:
    """Convert DIR HU → µ for G160 input. Warped output CTs stay HU."""
    meta = {"mu_mode": mode}
    if mode == "default":
        mu = hu_to_mu(hu, 0.02)
        meta["steps"] = ["hu_to_mu(0.02)"]
        return mu, meta
    if mode == "hist_match":
        # B: clip HU metal, then D: match p1–p99 to SPARE µ pool
        hu_c = np.clip(hu.astype(np.float32), -1000.0, 500.0)
        mu0 = hu_to_mu(hu_c, 0.02)
        s1, s99 = np.percentile(mu0, [1.0, 99.0])
        sp1, sp99 = spare_mu_p1_p99()
        mu = (mu0 - s1) / max(s99 - s1, 1e-8) * (sp99 - sp1) + sp1
        mu = np.clip(mu, min(sp1, float(mu.min())), max(sp99, float(np.percentile(mu0, 99.9))))
        meta.update(
            {
                "steps": ["clip_HU[-1000,500]", "hu_to_mu(0.02)", "p1p99_match_to_SPARE"],
                "dir_mu_p1_p99": [float(s1), float(s99)],
                "spare_mu_p1_p99": [sp1, sp99],
            }
        )
        return mu.astype(np.float32), meta
    raise ValueError(f"Unknown mu mode: {mode}")


def norm_mu(x: np.ndarray, air: float = 0.0, water: float = 0.02) -> np.ndarray:
    y = (x.astype(np.float32) - air) / max(water - air, 1e-8)
    return np.clip(y, 0.0, 1.5) / 1.5


def resize_volume_zyx(vol_zyx: np.ndarray, size: int) -> np.ndarray:
    t = torch.from_numpy(vol_zyx[None, None].astype(np.float32))
    t = F.interpolate(t, size=(size, size, size), mode="trilinear", align_corners=True)
    return t[0, 0].numpy()


def upsample_dvf_voxel(dvf_cxyz: np.ndarray, target_zyx_shape: tuple[int, int, int], src_size: int) -> np.ndarray:
    _, d0, h0, w0 = dvf_cxyz.shape
    dn, hn, wn = target_zyx_shape
    t = torch.from_numpy(dvf_cxyz[None].astype(np.float32))
    t = F.interpolate(t, size=(dn, hn, wn), mode="trilinear", align_corners=True)
    scales = torch.tensor([dn / d0, hn / h0, wn / w0], dtype=t.dtype).view(1, 3, 1, 1, 1)
    t = t * scales
    return t[0].numpy()


def sitk_from_array_like(arr_zyx: np.ndarray, ref: sitk.Image) -> sitk.Image:
    out = sitk.GetImageFromArray(arr_zyx)
    out.SetOrigin(ref.GetOrigin())
    out.SetSpacing(ref.GetSpacing())
    out.SetDirection(ref.GetDirection())
    return out


def write_vector_dvf_mha(dvf_c_zyx: np.ndarray, ref: sitk.Image, path: Path) -> None:
    sp_zyx = np.array(
        [ref.GetSpacing()[2], ref.GetSpacing()[1], ref.GetSpacing()[0]], dtype=np.float32
    )
    dvf_mm = dvf_c_zyx * sp_zyx.reshape(3, 1, 1, 1)
    arr = np.moveaxis(dvf_mm, 0, -1)[..., [2, 1, 0]]
    img = sitk.GetImageFromArray(arr, isVector=True)
    img.SetOrigin(ref.GetOrigin())
    img.SetSpacing(ref.GetSpacing())
    img.SetDirection(ref.GetDirection())
    sitk.WriteImage(img, str(path))


def load_generator(spec: dict, device: torch.device):
    net_root = Path(spec["net_root"])
    if str(net_root) not in sys.path:
        sys.path.insert(0, str(net_root))
    from networks.generator_crb_dec import UNetCRBDecoder

    g = UNetCRBDecoder(im_size=spec["im_size"], n_phases=10)
    ckpt = torch.load(spec["ckpt"], map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "generator" in ckpt:
        state = ckpt["generator"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    g.load_state_dict(state, strict=True)
    g.to(device).eval()
    return g


def stage_from_a1(case: int, staged: Path, geom_xml: Path) -> Path:
    """Stage R3 A1 mid-phase CT + masks; attach Geometry_SPARE."""
    scan_id = scan_id_for_case(case)
    a1_train = A1 / "runs" / scan_id / scan_id / "train"
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)

    # Prefer R3 train CT_06 (apply_r3 already done in A1 prepare). Native
    # data/staged GTVol_* are still PopulationStudy symlinks — do not use.
    ct06 = a1_train / "CT_06.mha"
    if not ct06.is_file():
        raise SystemExit(f"Missing A1 R3 CT_06 for case {case}: {ct06}")
    shutil.copy2(ct06, staged / "GTVol_06.mha")

    for name in ("Mask_Lung.mha", "Mask_Body.mha"):
        src = a1_train / name
        if not src.is_file():
            raise SystemExit(f"Missing A1 R3 {name} for case {case}: {src}")
        shutil.copy2(src, staged / name)

    proj = staged / "Proj"
    proj.mkdir()
    shutil.copy2(geom_xml, proj / "Geometry.xml")
    n_proj = len(ET.parse(geom_xml).findall("Projection"))
    bins = [str((i % 10) + 1) for i in range(n_proj)]
    (proj / "RespBin.csv").write_text("\n".join(bins) + "\n")
    logging.info(
        "Staged A3 case=%d from A1 R3 train | CT_06 + masks | geom=%s | %d projs",
        case,
        geom_xml.name,
        n_proj,
    )
    return staged / "GTVol_06.mha"


def ensure_layout(run_root: Path, staged: Path, scan_id: str) -> Path:
    train = run_root / scan_id / "train"
    if train.exists():
        shutil.rmtree(train)
    train.mkdir(parents=True)
    for src in staged.glob("Mask_*.mha"):
        shutil.copy2(src, train / src.name)
    shutil.copytree(staged / "Proj", train / "Proj")
    return train


def synthesize(
    train: Path,
    gt06: Path,
    device: torch.device,
    spec: dict,
    *,
    mu_mode: str = "default",
) -> dict:
    net_root = Path(spec["net_root"])
    if str(net_root) not in sys.path:
        sys.path.insert(0, str(net_root))
    from utilities.warp import warp

    ref_img = sitk.ReadImage(str(gt06))
    ct06_hu = sitk.GetArrayFromImage(ref_img).astype(np.float32)
    native_shape = ct06_hu.shape
    infer = int(spec["infer_size"])

    ct06_mu, mu_meta = dir_hu_to_mu_for_synth(ct06_hu, mu_mode)
    ct06_inf = resize_volume_zyx(ct06_mu, infer)
    ct_n = norm_mu(ct06_inf)
    logging.info(
        "Synth input mu_mode=%s | HU[%.0f,%.0f] → µ[%.4f,%.4f] → norm[%.3f,%.3f]",
        mu_mode,
        float(ct06_hu.min()),
        float(ct06_hu.max()),
        float(ct06_mu.min()),
        float(ct06_mu.max()),
        float(ct_n.min()),
        float(ct_n.max()),
    )

    x = torch.from_numpy(ct_n[None, None]).to(device)
    ref_ph = torch.tensor([5], dtype=torch.long, device=device)
    g = load_generator(spec, device)

    meta = {
        "synth_id": spec["synth_id"],
        "label": spec["label"],
        "ckpt": str(spec["ckpt"]),
        "norm": f"hu_to_mu[{mu_mode}]+norm_mu",
        "mu_meta": mu_meta,
        "infer_size": infer,
        "im_size": spec["im_size"],
        "native_shape": list(native_shape),
        "hu_range": [float(ct06_hu.min()), float(ct06_hu.max())],
        "mu_range": [float(ct06_mu.min()), float(ct06_mu.max())],
        "phases": {},
    }

    logging.info("Synthesizing from CT_06 | %s | native=%s | infer=%d", spec["label"], native_shape, infer)
    for phase in range(1, 11):
        tgt_ph = torch.tensor([phase - 1], dtype=torch.long, device=device)
        with torch.no_grad():
            if phase == 6:
                dvf_inf = torch.zeros(1, 3, infer, infer, infer, device=device)
            else:
                dvf_inf = g(x, ref_ph, tgt_ph)

        dvf_np = dvf_inf[0].cpu().numpy()
        dvf_nat = upsample_dvf_voxel(dvf_np, native_shape, infer)
        # Warp HU volume (DIR intensity for DRR / VoxelMap)
        ct_t = torch.from_numpy(ct06_hu[None, None])
        flow_t = torch.from_numpy(dvf_nat[None])
        with torch.no_grad():
            warped_nat = warp(ct_t, flow_t)[0, 0].numpy().astype(np.float32)

        out_ct = train / f"CT_{phase:02d}.mha"
        sitk.WriteImage(sitk_from_array_like(warped_nat, ref_img), str(out_ct))
        meta["phases"][f"{phase:02d}"] = {
            "dvf_l2_mean_infer": float(np.sqrt((dvf_np**2).sum(0)).mean()),
            "ct": out_ct.name,
        }
        logging.info(
            "  wrote %s | mean|u|_infer=%.4f",
            out_ct.name,
            meta["phases"][f"{phase:02d}"]["dvf_l2_mean_infer"],
        )
        np.save(train / f"_synth_dvf_infer_{phase:02d}.npy", dvf_np.astype(np.float32))

    sitk.WriteImage(ref_img, str(train / "CT_06.mha"))
    return meta


def downsample_volume_keep_hu(input_path: Path, output_path: Path, target_size=(128, 128, 128)) -> None:
    import itk

    image = itk.imread(str(input_path), itk.F)
    in_size = itk.size(image)
    in_spacing = image.GetSpacing()
    out_spacing = [(in_size[i] * in_spacing[i]) / target_size[i] for i in range(3)]
    pad = float(np.min(itk.GetArrayFromImage(image)))
    interpolator = itk.LinearInterpolateImageFunction.New(image)
    resampler = itk.ResampleImageFilter.New(image)
    resampler.SetInterpolator(interpolator)
    resampler.SetSize(list(target_size))
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputSpacing(out_spacing)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetDefaultPixelValue(pad)
    resampler.Update()
    arr = itk.GetArrayFromImage(resampler.GetOutput()).astype(np.float32)
    final_image = itk.GetImageFromArray(arr)
    final_image.SetSpacing([1.0, 1.0, 1.0])
    final_image.SetOrigin([0.0, 0.0, 0.0])
    final_image.SetDirection(itk.matrix_from_array(np.eye(3)))
    itk.imwrite(final_image, str(output_path))
    logging.info("Saved %s (HU kept, min=%.1f max=%.1f)", output_path.name, float(arr.min()), float(arr.max()))


def run_downsample(train: Path) -> int:
    for f in sorted(train.glob("CT_*.mha")):
        out = train / f"sub_CT{f.name[-7:]}"
        downsample_volume_keep_hu(f, out)
    return len(list(train.glob("sub_CT_*.mha")))


def write_synth_dvfs_after_downsample(
    train: Path,
    infer_size: int,
    *,
    convention: str = "elastix",
) -> int:
    """Write DVF_sub_*.mha labels for VoxelMap / TRE.

    G160 ``warp`` uses a pull field: I_tgt(x) = I_06(x + u(x)).
    A1 Elastix / eval_a1_tre expect fixed→moving d with x_mov ≈ x_fix + d(x_fix).
    First-order: d ≈ -u. Default ``convention='elastix'`` writes -u.
    Use ``convention='pull'`` only to keep raw G160 sampling vectors.
    """
    if convention not in ("elastix", "pull"):
        raise ValueError(f"convention must be 'elastix' or 'pull', got {convention!r}")
    ref = sitk.ReadImage(str(train / "sub_CT_06.mha"))
    n = 0
    for phase in range(1, 11):
        if phase == 6:
            continue
        npy = train / f"_synth_dvf_infer_{phase:02d}.npy"
        if not npy.is_file():
            raise FileNotFoundError(npy)
        dvf = np.load(npy)
        dvf_zyx = dvf[[2, 1, 0], ...]
        sd, sh, sw = sitk.GetArrayFromImage(ref).shape
        src = dvf_zyx.shape[1]
        if (sd, sh, sw) != dvf_zyx.shape[1:]:
            t = torch.from_numpy(dvf_zyx[None].astype(np.float32))
            t = F.interpolate(t, size=(sd, sh, sw), mode="trilinear", align_corners=True)
            scales = torch.tensor([sd / src, sh / src, sw / src], dtype=t.dtype).view(1, 3, 1, 1, 1)
            t = t * scales
            dvf_zyx = t[0].numpy()
        if convention == "elastix":
            dvf_zyx = -dvf_zyx
        out = train / f"DVF_sub_{phase:02d}.mha"
        write_vector_dvf_mha(dvf_zyx, ref, out)
        n += 1
        logging.info("  DVF %s (convention=%s)", out.name, convention)
    return n


def run_drr(train: Path, geom_xml: Path) -> None:
    from modules.drr_generation.run import run as run_drr_fn
    from elekta_drr import mc_varian_drr_opts_for_scan

    opts = mc_varian_drr_opts_for_scan(train)
    opts["geometry_path"] = str(geom_xml)
    mhas = sorted(train.glob("CT_*.mha"))
    for i, mha in enumerate(mhas, start=1):
        m = re.search(r"(\d+)", mha.stem)
        ct_num = int(m.group(1)) if m else i
        logging.info("DRR %s (%d/%d)", mha.name, i, len(mhas))
        ok, err = run_drr_fn(
            str(mha),
            str(train),
            geometry_file=str(geom_xml),
            ct_num=ct_num,
            dataset_type="spare",
            **opts,
        )
        if not ok:
            raise RuntimeError(f"DRR failed {mha.name}: {err}")


def run_compress(train: Path) -> int:
    from modules.drr_compression.compress import process_directory

    process_directory(train)
    return len(list(train.glob("*_Proj_*.bin")))


def run_prep(run_root: Path, scan_id: str, geom_xml: Path, *, with_test: bool) -> None:
    from modules.prep_train.run import run_prep_train

    split = "Train+Test" if with_test else "train"
    run_prep_train(
        run_root,
        {scan_id: split},
        dataset_type="spare",
        on_log=logging.info,
        angles_xml_path=geom_xml,
        prefer_patient_xml=False,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", type=int, required=True, choices=range(1, 11))
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--with-test", action="store_true", default=True)
    ap.add_argument("--skip-drr", action="store_true")
    ap.add_argument("--clean", action="store_true", default=True)
    ap.add_argument(
        "--mu-mode",
        choices=("default", "hist_match"),
        default="default",
        help="G160 input µ mapping; hist_match = clip HU[-1000,500] + p1–p99 match to SPARE",
    )
    ap.add_argument(
        "--run-suffix",
        default=None,
        help="Optional run folder suffix (default: muhist when --mu-mode=hist_match)",
    )
    ap.add_argument(
        "--synth-ckpt",
        type=Path,
        default=None,
        help="Override G160 generator weights (e.g. DIR fine-tuned Decoder)",
    )
    ap.add_argument(
        "--dvf-convention",
        choices=("elastix", "pull"),
        default="elastix",
        help="DVF_sub labels: elastix = -u (fixed→moving, default); pull = raw G160 sampling field",
    )
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if str(LEARN) not in sys.path:
        sys.path.insert(0, str(LEARN))
    if str(VMC / "config") not in sys.path:
        sys.path.insert(0, str(VMC / "config"))

    case = args.case
    scan_id = scan_id_for_case(case)
    # Prefer A1 Geometry.xml if it is already SPARE (OffsetY -2); else Geometry_SPARE.
    geom_xml = A1 / "Geometry.xml"
    if not geom_xml.is_file() or geom_xml.stat().st_size < 1000:
        geom_xml = GEOMETRY_SPARE
    if not geom_xml.is_file():
        raise SystemExit(f"Missing geometry: tried {A1 / 'Geometry.xml'} and {GEOMETRY_SPARE}")
    # Sanity: R3 A1 uses OffsetY -2
    geom_txt = geom_xml.read_text(errors="ignore")
    if "<ProjectionOffsetY>-2</ProjectionOffsetY>" not in geom_txt:
        logging.warning("Geometry %s may not be SPARE OffsetY=-2", geom_xml)

    spec = dict(SYNTH_SPEC)
    if args.synth_ckpt is not None:
        spec["ckpt"] = args.synth_ckpt.resolve()
        spec["label"] = f"{spec['label']} · FT={args.synth_ckpt.name}"
    if not Path(spec["ckpt"]).is_file():
        raise SystemExit(f"Missing synth ckpt: {spec['ckpt']}")

    suffix = args.run_suffix
    if suffix is None:
        if args.synth_ckpt is not None:
            suffix = "g160ft"
        elif args.mu_mode == "hist_match":
            suffix = "muhist"
    run_name = f"{scan_id}_{suffix}" if suffix else scan_id
    run_root = A3 / "runs" / run_name
    staged = A3 / "data" / "staged" / scan_id
    logs = run_root / "logs"
    if args.clean and run_root.exists():
        shutil.rmtree(run_root)
    logs.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logs / "prepare_a3.log", mode="w"),
        ],
        force=True,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    logging.info(
        "A3 prepare case=%d scan_id=%s run=%s mu_mode=%s dvf=%s ckpt=%s device=%s",
        case,
        scan_id,
        run_name,
        args.mu_mode,
        args.dvf_convention,
        Path(spec["ckpt"]).name,
        device,
    )

    gt06 = stage_from_a1(case, staged, geom_xml)
    train = ensure_layout(run_root, staged, scan_id)
    train_geom = train / "Proj" / "Geometry.xml"

    logging.info("=== SYNTHESIZE ===")
    meta = synthesize(train, gt06, device, spec, mu_mode=args.mu_mode)
    meta["dvf_label_convention"] = args.dvf_convention
    meta["dvf_label_note"] = (
        "elastix: DVF_sub = -u (G160 pull→ITK fixed→moving); "
        "pull: DVF_sub = u as used by warp(I_06, u)"
    )
    (run_root / "synth_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    logging.info("=== DOWNSAMPLE (HU kept) ===")
    logging.info("sub_CT count: %d", run_downsample(train))

    logging.info("=== WRITE SYNTH DVFs (convention=%s) ===", args.dvf_convention)
    logging.info(
        "DVFs: %d",
        write_synth_dvfs_after_downsample(
            train, int(SYNTH_SPEC["infer_size"]), convention=args.dvf_convention
        ),
    )

    if not args.skip_drr:
        logging.info("=== DRR ===")
        run_drr(train, train_geom)
        logging.info("=== COMPRESS ===")
        logging.info("bins: %d", run_compress(train))

    logging.info("=== PREP_TRAIN ===")
    run_prep(run_root, scan_id, train_geom, with_test=args.with_test)

    mt = run_root / "ModelTraining" / "train" / scan_id
    logging.info("Done in %.1f min → %s", (time.perf_counter() - t0) / 60, mt)
    return 0 if mt.is_dir() else 1


if __name__ == "__main__":
    raise SystemExit(main())
