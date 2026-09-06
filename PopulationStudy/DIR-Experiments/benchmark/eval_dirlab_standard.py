#!/usr/bin/env python3
"""DIR-Lab standard benchmark: TRE (mm) on 300 landmarks, T00→T50.

Methods: identity, elastix (packed), crb checkpoints (E6 / E3 / E5).
Grids: native_mm (paper-style) and packed_vox (internal).

  cd PopulationStudy/DIR-Experiments
  /path/to/LEARN-GUI/.venv/bin/python benchmark/eval_dirlab_standard.py \\
      --methods identity,elastix,e6_both_mu --gpu 1

  # full sweep (longer; elastix ~10 cases × 1 pair)
  ... --methods identity,elastix,e6_both_mu,e6_both_minmax,e3_both_mu,e5_both_mu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

DIR = Path(__file__).resolve().parents[1]
POP = DIR.parent
PACKED = DIR / "Experiment1" / "packed"
BENCH = DIR / "benchmark"
CE = POP / "ClinicalExperiments"
E6 = CE / "Experiment6" / "Normal"
E6CY = CE / "Experiment6" / "Cyclic"
E3 = CE / "Experiment3"
IE1 = POP / "InitialExperiments" / "Experiment1"
REPO = POP.parent
PARAM = REPO / "My v1.0" / "configs" / "elastix_bspline_masked.txt"

sys.path.insert(0, str(IE1))
sys.path.insert(0, str(CE / "scripts"))
sys.path.insert(0, str(BENCH))

from tre_utils import (  # noqa: E402
    PHASE_T00,
    PHASE_T50,
    eval_tre_from_disp,
    find_300_landmarks,
    load_pack_meta,
    sample_dvf_at_xyz,
)

# Method registry: name -> (arch, ckpt_path, norm, net_family)
# net_family: e6 = linear phase (IE1 nets), e3 = cyclic phase (E3 nets)
METHOD_SPECS: dict[str, tuple] = {
    "e6_encoder_mu": ("encoder", E6 / "EncoderCRB/weights/crb_enc_mse_linear_full128_generator.pth", "mu", "e6"),
    "e6_decoder_mu": ("decoder", E6 / "DecoderCRB/weights/crb_dec_mse_linear_full128_generator.pth", "mu", "e6"),
    "e6_both_mu": ("both", E6 / "BothCRB/weights/crb_both_mse_linear_full128_generator.pth", "mu", "e6"),
    "e6_both_minmax": ("both", E6 / "BothCRB/weights/crb_both_mse_linear_full128_generator.pth", "minmax", "e6"),
    "e6_cyclic_encoder_mu": ("encoder", E6CY / "EncoderCRB/weights/crb_enc_mse_cyclic_full128_generator.pth", "mu", "e3"),
    "e6_cyclic_decoder_mu": ("decoder", E6CY / "DecoderCRB/weights/crb_dec_mse_cyclic_full128_generator.pth", "mu", "e3"),
    "e6_cyclic_both_mu": ("both", E6CY / "BothCRB/weights/crb_both_mse_cyclic_full128_generator.pth", "mu", "e3"),
    "e6_cyclic_both_minmax": ("both", E6CY / "BothCRB/weights/crb_both_mse_cyclic_full128_generator.pth", "minmax", "e3"),
    "e3_both_mu": ("both", E3 / "BothCRB/weights/crb_both_mse_cyclic_fov_aug_generator.pth", "mu", "e3"),
    "e5_both_mu": ("both", E3 / "BothCRB/weights/crb_both_mse_cyclic_fov_aug_generator.pth", "mu", "e3"),
}


def norm_minmax(x):
    x = x.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi <= lo else (x - lo) / (hi - lo)


def norm_mu(x, air=0.0, water=0.02):
    x = x.astype(np.float32)
    y = (x - air) / max(water - air, 1e-8)
    return np.clip(y, 0.0, 1.5) / 1.5


NORMS = {"minmax": norm_minmax, "mu": norm_mu}


def run_identity(pid: str, packed_root: Path, grid: str) -> dict:
    meta = load_pack_meta(packed_root, pid)
    ref_n, tgt_n, ref_p = find_300_landmarks(packed_root, pid, meta)
    spacing = meta["native_spacing_xyz_mm"]
    tre_id = __import__("tre_utils").tre_mm(ref_n, tgt_n, spacing)
    return {
        "method": "identity",
        "patient": pid,
        "grid": grid,
        "tre_mm": tre_id,
        "tre_identity_mm": tre_id,
    }


def run_elastix(pid: str, packed_root: Path, grid: str, param_file: Path, skip_existing: bool) -> dict:
    meta = load_pack_meta(packed_root, pid)
    ref_n, tgt_n, ref_p = find_300_landmarks(packed_root, pid, meta)
    spacing = meta["native_spacing_xyz_mm"]
    data = packed_root / pid / "all"
    cache = packed_root / pid / "benchmark" / f"elastix_{PHASE_T00:02d}_to_{PHASE_T50:02d}.npy"
    cache.parent.mkdir(parents=True, exist_ok=True)

    if cache.is_file():
        dvf = np.load(cache)
        if not skip_existing:
            print(f"  [{pid}] Elastix cache hit {cache.name}", flush=True)
    else:
        from prepare_clinical_dvf_library import register_pair  # noqa: E402

        mask = (np.load(data / "Mask_Lung.npy") > 0).astype(np.uint8)
        fixed = np.load(data / f"CT_{PHASE_T50:02d}.npy").astype(np.float32)
        moving = np.load(data / f"CT_{PHASE_T00:02d}.npy").astype(np.float32)
        print(f"  [{pid}] Elastix T00→T50 …", flush=True)
        t0 = time.time()
        dvf = register_pair(fixed, moving, mask, param_file)  # z,y,x,3
        np.save(cache, dvf.astype(np.float32))
        print(f"  [{pid}] Elastix done {time.time()-t0:.0f}s", flush=True)

    dvf_cdhw = np.moveaxis(dvf, -1, 0)
    disp = sample_dvf_at_xyz(dvf_cdhw, ref_p)
    out = eval_tre_from_disp(ref_n, tgt_n, ref_p, disp, meta, spacing, grid=grid)
    out.update(method="elastix", patient=pid, elastix_cache=str(cache))
    return out


def _load_crb(arch: str, ckpt: Path, device, net_family: str = "e6"):
    import importlib.util

    if net_family == "e3":
        mod_files = {
            "encoder": E3 / "networks/generator_crb.py",
            "decoder": E3 / "networks/generator_crb_dec.py",
            "both": E3 / "networks/generator_crb_both.py",
        }
        if not mod_files["encoder"].is_file():
            mod_files = {
                "encoder": E6CY.parent / "Experiment2" / "networks/generator_crb.py",
                "decoder": E6CY.parent / "Experiment2" / "networks/generator_crb_dec.py",
                "both": E6CY.parent / "Experiment2" / "networks/generator_crb_both.py",
            }
        class_names = {
            "encoder": "UNetCRB",
            "decoder": "UNetCRBDecoder",
            "both": "UNetCRBBoth",
        }
        spec = importlib.util.spec_from_file_location(
            f"dirlab_e3_{arch}", mod_files[arch]
        )
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        model_cls = getattr(mod, class_names[arch])
    else:
        from networks.generator_crb import UNetCRB  # noqa: E402
        from networks.generator_crb_both import UNetCRBBoth  # noqa: E402
        from networks.generator_crb_dec import UNetCRBDecoder  # noqa: E402

        model_cls = {
            "encoder": UNetCRB,
            "decoder": UNetCRBDecoder,
            "both": UNetCRBBoth,
        }[arch]

    g = model_cls(im_size=128, n_phases=10).to(device)
    g.load_state_dict(torch.load(ckpt, map_location=device))
    g.eval()
    return g


def run_crb(
    method: str,
    pid: str,
    packed_root: Path,
    grid: str,
    device: torch.device,
    model=None,
) -> dict:
    arch, ckpt, norm_name, _net_family = METHOD_SPECS[method]
    meta = load_pack_meta(packed_root, pid)
    ref_n, tgt_n, ref_p = find_300_landmarks(packed_root, pid, meta)
    spacing = meta["native_spacing_xyz_mm"]
    data = packed_root / pid / "all"
    apply_norm = NORMS[norm_name]
    ct_r = apply_norm(np.load(data / f"CT_{PHASE_T00:02d}.npy"))

    g = model if model is not None else _load_crb(arch, ckpt, device, _net_family)
    with torch.no_grad():
        ref_t = torch.from_numpy(ct_r)[None, None].to(device)
        pred_t = g(
            ref_t,
            torch.tensor([PHASE_T00 - 1], device=device),
            torch.tensor([PHASE_T50 - 1], device=device),
        )
        pred = pred_t[0].cpu().numpy()

    disp = sample_dvf_at_xyz(pred, ref_p)
    out = eval_tre_from_disp(ref_n, tgt_n, ref_p, disp, meta, spacing, grid=grid)
    out.update(method=method, patient=pid, arch=arch, norm=norm_name, ckpt=str(ckpt))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--methods",
        default="identity,elastix,e6_both_mu",
        help="comma list: identity,elastix,e6_both_mu,e6_both_minmax,e3_both_mu,...",
    )
    ap.add_argument("--grids", default="native_mm,packed_vox")
    ap.add_argument("--patients", default=",".join(f"P{i}_DIR" for i in range(1, 11)))
    ap.add_argument("--packed-root", type=Path, default=PACKED)
    ap.add_argument("--param-file", type=Path, default=PARAM)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--skip-elastix", action="store_true", help="reuse cached elastix DVF")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    grids = [g.strip() for g in args.grids.split(",") if g.strip()]
    patients = [p.strip() for p in args.patients.split(",") if p.strip()]
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print(f"DIR benchmark  device={device}  packed={args.packed_root}", flush=True)
    print(f"methods={methods}  grids={grids}", flush=True)

    rows = []
    all_results = []
    crb_cache: dict[str, object] = {}

    for pid in patients:
        if not (args.packed_root / pid / "pack_meta.json").is_file():
            print(f"WARN skip {pid}: not packed", flush=True)
            continue
        for method in methods:
            for grid in grids:
                print(f"\n[{pid}] {method} grid={grid}", flush=True)
                if method == "identity":
                    r = run_identity(pid, args.packed_root, grid)
                elif method == "elastix":
                    r = run_elastix(
                        pid, args.packed_root, grid, args.param_file, args.skip_elastix
                    )
                elif method in METHOD_SPECS:
                    if method not in crb_cache:
                        arch, ckpt, _, net_family = METHOD_SPECS[method]
                        crb_cache[method] = _load_crb(arch, ckpt, device, net_family)
                    r = run_crb(
                        method, pid, args.packed_root, grid, device, crb_cache[method]
                    )
                else:
                    raise SystemExit(f"unknown method: {method}")

                t = r["tre_mm"]
                ti = r["tre_identity_mm"]
                print(
                    f"  TRE mean={t['mean']:.2f} med={t['median']:.2f} "
                    f"(id={ti['mean']:.2f})",
                    flush=True,
                )
                rows.append(
                    f"{method}\t{grid}\t{pid}\t{t['mean']:.4f}\t{t['std']:.4f}\t"
                    f"{t['median']:.4f}\t{ti['mean']:.4f}\t{t['n']}"
                )
                all_results.append(r)

    # cohort summary (standard Tier-0 report block)
    print("\n======== COHORT MEAN TRE (mm) — native_mm ========", flush=True)
    print(f"{'method':<22s} {'mean':>7s} {'vs_id':>7s} {'beat_id':>8s}  n", flush=True)
    print("-" * 50, flush=True)
    grid_primary = "native_mm"
    id_means = [
        r["tre_identity_mm"]["mean"]
        for r in all_results
        if r["method"] == "identity" and r["grid"] == grid_primary
    ]
    cohort_id = float(np.mean(id_means)) if id_means else float("nan")
    for method in methods:
        subset = [
            r for r in all_results if r["method"] == method and r["grid"] == grid_primary
        ]
        if not subset:
            continue
        means = [r["tre_mm"]["mean"] for r in subset]
        ids = [r["tre_identity_mm"]["mean"] for r in subset]
        m = float(np.mean(means))
        beat_n = sum(1 for a, b in zip(means, ids) if a < b)
        vs_id = m - cohort_id if method != "identity" else 0.0
        beat_tag = f"{beat_n}/{len(means)}"
        print(
            f"{method:<22s} {m:7.2f} {vs_id:+7.2f} {beat_tag:>8s}  {len(means)}",
            flush=True,
        )

    elastix_rows = [
        r for r in all_results if r["method"] == "elastix" and r["grid"] == grid_primary
    ]
    if elastix_rows:
        beat = sum(
            1 for r in elastix_rows if r["tre_mm"]["mean"] < r["tre_identity_mm"]["mean"]
        )
        status = "PASS" if beat >= len(elastix_rows) // 2 + 1 else "FAIL"
        print(
            f"\nHarness check: Elastix beats identity on {beat}/{len(elastix_rows)} "
            f"patients ({status})",
            flush=True,
        )

    print("\n======== packed_vox (diagnostic) ========", flush=True)
    for method in methods:
        subset = [r for r in all_results if r["method"] == method and r["grid"] == "packed_vox"]
        if not subset:
            continue
        means = [r["tre_mm"]["mean"] for r in subset]
        print(f"{method:20s}  mean={np.mean(means):.2f}  n={len(means)}", flush=True)

    out_dir = BENCH / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    pack_tag = args.packed_root.name
    tsv = out_dir / f"tre_benchmark_{pack_tag}.tsv"
    tsv.write_text(
        "method\tgrid\tpatient\tTRE_mean\tTRE_std\tTRE_median\tTRE_identity\tn\n"
        + "\n".join(rows)
        + "\n"
    )
    (out_dir / f"tre_benchmark_{pack_tag}.json").write_text(
        json.dumps(all_results, indent=2) + "\n"
    )
    print(f"\nwrote {tsv}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
