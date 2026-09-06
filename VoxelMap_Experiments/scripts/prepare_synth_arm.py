#!/usr/bin/env python3
"""Arm B: CT_06 → synthesize 10 phases → DRR → prep_train.

Default synthesizer: G160-A1 Decoder + µ air/water (linear phase).
Legacy E3 Both + p1_p99 available via --synth-id synth_e3_both.

Run layout:
  VoxelMap_Experiments/runs/<scan_id>/<synth_id>/
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
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parent
VMC = Path(os.environ.get("VOXELMAP_CLINICAL_ROOT", "/home/abhishek/Documents/VoxelMap_Clinical"))
LEARN = Path(os.environ.get("LEARN_GUI_ROOT", "/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python"))

G160_E1 = REPO / "PopulationStudy" / "ClinicalExperiments" / "Grid160" / "Experiment1"
IE6 = REPO / "PopulationStudy" / "InitialExperiments" / "Experiment6"
G128_E3 = REPO / "PopulationStudy" / "ClinicalExperiments" / "Experiment3"

SYNTH_SPECS = {
    "synth_g160_a1_dec": {
        "label": "G160-A1 Decoder · µ air/water · linear",
        "ckpt": G160_E1 / "DecoderCRB" / "weights" / "crb_dec_mse_iso_g160_fov_full_generator.pth",
        "net_root": IE6,
        "arch": "decoder",
        "norm": "mu",
        "im_size": 64,
        "infer_size": 160,
        "cyclic": False,
    },
    "synth_e3_both": {
        "label": "Grid128-E3 Both · p1_p99 · cyclic (legacy)",
        "ckpt": G128_E3 / "BothCRB" / "weights" / "crb_both_mse_cyclic_fov_aug_generator.pth",
        "net_root": G128_E3,
        "arch": "both",
        "norm": "p1_p99",
        "im_size": 128,
        "infer_size": 128,
        "cyclic": True,
    },
}


def norm_p1_p99(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, [1.0, 99.0]).astype(np.float32)
    if hi <= lo:
        return np.zeros_like(x)
    y = np.clip(x, lo, hi)
    return (y - lo) / (hi - lo)


def norm_mu(x: np.ndarray, air: float = 0.0, water: float = 0.02) -> np.ndarray:
    x = x.astype(np.float32)
    y = (x - air) / max(water - air, 1e-8)
    return np.clip(y, 0.0, 1.5) / 1.5


def apply_norm(x: np.ndarray, name: str) -> np.ndarray:
    if name == "mu":
        return norm_mu(x)
    if name == "p1_p99":
        return norm_p1_p99(x)
    raise ValueError(name)


def sitk_from_array_like(arr_zyx: np.ndarray, ref: sitk.Image) -> sitk.Image:
    out = sitk.GetImageFromArray(arr_zyx)
    out.SetOrigin(ref.GetOrigin())
    out.SetSpacing(ref.GetSpacing())
    out.SetDirection(ref.GetDirection())
    return out


def resize_volume_zyx(vol_zyx: np.ndarray, size: int) -> np.ndarray:
    t = torch.from_numpy(vol_zyx[None, None].astype(np.float32))
    t = F.interpolate(t, size=(size, size, size), mode="trilinear", align_corners=True)
    return t[0, 0].numpy()


def upsample_dvf_voxel(dvf_cxyz: np.ndarray, target_zyx_shape: tuple[int, int, int], src_size: int) -> np.ndarray:
    """dvf (3,D,H,W) voxel units @ infer grid → (3,Dn,Hn,Wn) voxel units @ native."""
    _, d0, h0, w0 = dvf_cxyz.shape
    dn, hn, wn = target_zyx_shape
    t = torch.from_numpy(dvf_cxyz[None].astype(np.float32))
    t = F.interpolate(t, size=(dn, hn, wn), mode="trilinear", align_corners=True)
    scales = torch.tensor([dn / d0, hn / h0, wn / w0], dtype=t.dtype).view(1, 3, 1, 1, 1)
    t = t * scales
    return t[0].numpy()


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


def ensure_layout(run_root: Path, staged: Path, scan_id: str) -> Path:
    train = run_root / scan_id / "train"
    train.mkdir(parents=True, exist_ok=True)
    for src in staged.glob("Mask_*.mha"):
        dst = train / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
    proj = train / "Proj"
    if not (proj / "Geometry.xml").is_file():
        if proj.exists():
            shutil.rmtree(proj)
        shutil.copytree(staged / "Proj", proj)
    return train


def load_generator(spec: dict, device: torch.device):
    net_root = Path(spec["net_root"])
    if str(net_root) not in sys.path:
        sys.path.insert(0, str(net_root))
    if spec["arch"] == "decoder":
        from networks.generator_crb_dec import UNetCRBDecoder

        g = UNetCRBDecoder(im_size=spec["im_size"], n_phases=10)
    elif spec["arch"] == "both":
        from networks.generator_crb_both import UNetCRBBoth

        g = UNetCRBBoth(im_size=spec["im_size"], n_phases=10)
    else:
        raise ValueError(spec["arch"])
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


def synthesize(train: Path, staged: Path, device: torch.device, spec: dict) -> dict:
    from utilities.warp import warp  # noqa: WPS — after net_root on path

    ct06_path = staged / "GTVol_06.mha"
    ref_img = sitk.ReadImage(str(ct06_path))
    ct06_zyx = sitk.GetArrayFromImage(ref_img).astype(np.float32)
    native_shape = ct06_zyx.shape
    infer = int(spec["infer_size"])

    ct06_inf = resize_volume_zyx(ct06_zyx, infer)
    ct_n = apply_norm(ct06_inf, spec["norm"])
    x = torch.from_numpy(ct_n[None, None]).to(device)
    ref_ph = torch.tensor([5], dtype=torch.long, device=device)

    g = load_generator(spec, device)

    meta = {
        "synth_id": spec.get("synth_id"),
        "label": spec["label"],
        "ckpt": str(spec["ckpt"]),
        "norm": spec["norm"],
        "infer_size": infer,
        "im_size": spec["im_size"],
        "arch": spec["arch"],
        "native_shape": list(native_shape),
        "phases": {},
    }

    logging.info(
        "Synthesizing from CT_06 | %s | native=%s | infer=%d | norm=%s",
        spec["label"],
        native_shape,
        infer,
        spec["norm"],
    )
    for phase in range(1, 11):
        tgt_ph = torch.tensor([phase - 1], dtype=torch.long, device=device)
        with torch.no_grad():
            if phase == 6:
                dvf_inf = torch.zeros(1, 3, infer, infer, infer, device=device)
            else:
                dvf_inf = g(x, ref_ph, tgt_ph)
            _ = warp(x, dvf_inf)  # sanity; native warp below

        dvf_np = dvf_inf[0].cpu().numpy()
        dvf_nat = upsample_dvf_voxel(dvf_np, native_shape, infer)
        ct_t = torch.from_numpy(ct06_zyx[None, None])
        flow_t = torch.from_numpy(dvf_nat[None])
        with torch.no_grad():
            warped_nat = warp(ct_t, flow_t)[0, 0].numpy().astype(np.float32)

        out_ct = train / f"CT_{phase:02d}.mha"
        sitk.WriteImage(sitk_from_array_like(warped_nat, ref_img), str(out_ct))
        meta["phases"][f"{phase:02d}"] = {
            "dvf_l2_mean_infer": float(np.sqrt((dvf_np**2).sum(0)).mean()),
            "ct": str(out_ct.name),
        }
        logging.info(
            "  wrote %s | mean|u|_infer=%.4f",
            out_ct.name,
            meta["phases"][f"{phase:02d}"]["dvf_l2_mean_infer"],
        )
        np.save(train / f"_synth_dvf_infer_{phase:02d}.npy", dvf_np.astype(np.float32))

    sitk.WriteImage(ref_img, str(train / "CT_06.mha"))
    return meta


def write_synth_dvfs_after_downsample(train: Path, infer_size: int) -> int:
    ref = sitk.ReadImage(str(train / "sub_CT_06.mha"))
    n = 0
    for phase in range(1, 11):
        if phase == 6:
            continue
        npy = train / f"_synth_dvf_infer_{phase:02d}.npy"
        # legacy filename from old E3 runs
        if not npy.is_file():
            npy = train / f"_synth_dvf_128_{phase:02d}.npy"
        if not npy.is_file():
            raise FileNotFoundError(npy)
        dvf = np.load(npy)
        dvf_zyx = dvf[[2, 1, 0], ...]
        sd, sh, sw = sitk.GetArrayFromImage(ref).shape
        src = dvf_zyx.shape[1]
        if (sd, sh, sw) != dvf_zyx.shape[1:]:
            t = torch.from_numpy(dvf_zyx[None].astype(np.float32))
            t = F.interpolate(t, size=(sd, sh, sw), mode="trilinear", align_corners=True)
            scales = torch.tensor(
                [sd / src, sh / src, sw / src], dtype=t.dtype
            ).view(1, 3, 1, 1, 1)
            t = t * scales
            dvf_zyx = t[0].numpy()
        out = train / f"DVF_sub_{phase:02d}.mha"
        write_vector_dvf_mha(dvf_zyx, ref, out)
        n += 1
        logging.info("  DVF %s", out.name)
    return n


def run_downsample(train: Path) -> int:
    from modules.downsampling.downsample import process_directory

    process_directory(str(train))
    return len(list(train.glob("sub_CT_*.mha")))


def _drr_opts_for_scan(train: Path, scan_id: str, geom_xml: Path) -> dict:
    """CV_* → clinical Varian opts; MC_* → SPARE MC/Varian half-fan gold-batch."""
    if scan_id.startswith("MC_"):
        from elekta_drr import mc_varian_drr_opts_for_scan

        opts = mc_varian_drr_opts_for_scan(train)
    else:
        from varian_drr import varian_drr_opts_for_scan

        opts = varian_drr_opts_for_scan(train, scan_id)
    opts["geometry_path"] = str(geom_xml)
    return opts


def run_drr(train: Path, geom_xml: Path, scan_id: str) -> None:
    from modules.drr_generation.run import run as run_drr_fn

    opts = _drr_opts_for_scan(train, scan_id, geom_xml)
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
            dataset_type="spare" if scan_id.startswith("MC_") else "clinical",
            **opts,
        )
        if not ok:
            raise RuntimeError(f"DRR failed {mha.name}: {err}")


def run_compress(train: Path) -> int:
    from modules.drr_compression.compress import process_directory

    process_directory(train)
    return len(list(train.glob("*_Proj_*.bin")))


def run_prep(run_root: Path, scan_id: str, geom_xml: Path, with_test: bool) -> None:
    from modules.prep_train.run import run_prep_train

    split = "Train+Test" if with_test else "train"
    run_prep_train(
        run_root,
        {scan_id: split},
        dataset_type="spare",
        on_log=lambda m: logging.info(m),
        angles_xml_path=geom_xml,
        prefer_patient_xml=False,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-id", default="CV_P3_V_01")
    ap.add_argument(
        "--synth-id",
        default="synth_g160_a1_dec",
        choices=sorted(SYNTH_SPECS.keys()),
        help="subfolder under runs/<scan_id>/",
    )
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--with-test", action="store_true", default=True)
    ap.add_argument("--clean", action="store_true", default=True)
    ap.add_argument("--skip-drr", action="store_true")
    ap.add_argument("--infer-size", type=int, default=None, help="override model grid size")
    args = ap.parse_args()

    scan_id = args.scan_id
    synth_id = args.synth_id
    spec = dict(SYNTH_SPECS[synth_id])
    spec["synth_id"] = synth_id
    if args.infer_size is not None:
        spec["infer_size"] = args.infer_size

    m = re.search(r"P(\d+)", scan_id)
    if not m:
        raise SystemExit(f"Cannot parse patient number from scan-id: {scan_id}")
    pnum = m.group(1)
    staged = EXP / "data" / "staged" / f"P{pnum}" / scan_id
    run_root = EXP / "runs" / scan_id / synth_id

    if not staged.is_dir():
        raise SystemExit(f"Staged data missing: {staged}")
    if not Path(spec["ckpt"]).is_file():
        raise SystemExit(f"Missing synthesizer ckpt: {spec['ckpt']}")

    if args.clean and run_root.exists():
        shutil.rmtree(run_root)
    (run_root / "logs").mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_root / "logs" / "prepare_synth_arm.log", mode="w"),
        ],
    )

    # LEARN / VMC on path for DRR + prep
    if str(LEARN) not in sys.path:
        sys.path.insert(0, str(LEARN))
    if str(VMC / "config") not in sys.path:
        sys.path.insert(0, str(VMC / "config"))
    # networks + warp
    if str(spec["net_root"]) not in sys.path:
        sys.path.insert(0, str(spec["net_root"]))

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    t0 = time.perf_counter()
    train = ensure_layout(run_root, staged, scan_id)
    geom = train / "Proj" / "Geometry.xml"

    logging.info("=== SYNTHESIZE from CT_06 (%s) ===", spec["label"])
    logging.info("run_root=%s", run_root)
    meta = synthesize(train, staged, device, spec)
    (run_root / "synth_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    logging.info("=== DOWNSAMPLE ===")
    n_sub = run_downsample(train)
    logging.info("sub_CT count: %d", n_sub)

    logging.info("=== WRITE SYNTH DVFs ===")
    n_dvf = write_synth_dvfs_after_downsample(train, int(spec["infer_size"]))
    logging.info("DVFs: %d", n_dvf)

    if not args.skip_drr:
        logging.info("=== DRR ===")
        run_drr(train, geom, scan_id)

    logging.info("=== COMPRESS ===")
    n_bin = run_compress(train)
    logging.info("bins: %d", n_bin)

    logging.info("=== PREP_TRAIN ===")
    run_prep(run_root, scan_id, geom, with_test=args.with_test)

    mt = run_root / "ModelTraining" / "train" / scan_id
    logging.info("Done in %.1f min | ModelTraining=%s", (time.perf_counter() - t0) / 60, mt)
    if mt.is_dir():
        for sub in ("SourceProjections", "TargetProjections", "DVFs"):
            p = mt / sub
            logging.info("  %s: %d", sub, len(list(p.glob("*"))) if p.is_dir() else 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
