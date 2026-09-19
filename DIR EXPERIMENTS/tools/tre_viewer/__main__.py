"""CLI entry: ``python -m tre_viewer`` from ``DIR EXPERIMENTS/tools``.

Phase 0 commands:
  --list-runs
  --verify [--arm A1] [--case N] [--field elastix_mha]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# Allow ``python -m tre_viewer`` when cwd or tools/ is on sys.path
_TOOLS = Path(__file__).resolve().parent.parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

def _runs(args):
    from tre_viewer.data import discover_runs, discover_runs_in_folder

    runs = (discover_runs_in_folder(args.runs_dir) if args.runs_dir
            else discover_runs([args.arm] if args.arm else None))
    return [r for r in runs if args.case is None or r.case == args.case]


def _cmd_doctor(args: argparse.Namespace) -> int:
    from importlib import import_module

    root = _TOOLS.parent
    checks = [("Python 3.10+", sys.version_info >= (3, 10), sys.version.split()[0])]
    for module in ("numpy", "scipy", "SimpleITK", "napari", "qtpy", "PyQt6.QtWidgets", "matplotlib"):
        try:
            import_module(module)
            checks.append((module, True, "imports successfully"))
        except (ImportError, OSError, RuntimeError) as exc:
            checks.append((module, False, str(exc)))
    evaluator = root / "scripts" / "eval_a1_tre.py"
    checks.append(("TRE coordinate adapter", evaluator.is_file(), str(evaluator)))
    packs = Path(os.environ.get("DIRLAB_ROOT", root / "data" / "dirlab_packs")).expanduser()
    checks.append(("DIR-Lab packs", packs.is_dir(), f"{packs} (set DIRLAB_ROOT to change)"))
    for label, ok, detail in checks:
        print(f"[{'OK' if ok else 'MISSING'}] {label}: {detail}")
    return 0 if all(ok for _, ok, _ in checks) else 1


def _nonnegative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a finite, non-negative number")
    return number


def _cmd_list_runs(args: argparse.Namespace) -> int:
    from tre_viewer.data import DIR_EXP, discover_runs, discover_runs_in_folder

    if args.runs_dir:
        runs = discover_runs_in_folder(
            args.runs_dir, require_summary=args.require_summary
        )
    else:
        arms = [args.arm] if args.arm else None
        runs = discover_runs(arms, require_summary=args.require_summary)
    if not runs:
        print("No runs found. Use --runs-dir /path/to/runs or run 'tre_viewer doctor'.")
        return 1
    print(f"{'arm':<28} {'case':>4} {'frame':<8} {'fields':<40} summary")
    print("-" * 100)
    for r in runs:
        fields = ",".join(r.fields_available)
        summ = (
            str(r.tre_summary_path.relative_to(DIR_EXP))
            if r.tre_summary_path and r.tre_summary_path.is_relative_to(DIR_EXP)
            else (str(r.tre_summary_path) if r.tre_summary_path else "—")
        )
        print(
            f"{r.arm:<28} C{r.case:02d}  {r.frame:<8} {fields:<40} {summ}"
        )
    print(f"\n{len(runs)} run(s). DIR_EXP={DIR_EXP}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    from tre_viewer.data import verify_against_summary

    runs = _runs(args)
    if not runs:
        print("No matching runs.")
        return 1

    n_fail = 0
    n_checked = 0
    n_skipped = 0
    for run in runs:
        print(f"\n=== {run.arm} C{run.case:02d} frame={run.frame} ===")
        try:
            checks = verify_against_summary(
                run, field=args.field, which=args.which, pair=args.pair, atol=args.atol,
            )
        except (OSError, ValueError, ImportError) as exc:
            if args.debug:
                raise
            print(f"  [FAIL] {exc}")
            n_fail += 1
            continue
        for c in checks:
            status = "SKIP" if c["ok"] is None else ("OK" if c["ok"] else "FAIL")
            if c["ok"] is None:
                n_skipped += 1
            elif not c["ok"]:
                n_fail += 1
            else:
                n_checked += 1
            exp = c.get("expected")
            got = c.get("got")
            delta = c.get("delta")
            extra = c.get("note") or ""
            print(
                f"  [{status}] {c['name']}: "
                f"expected={exp} got={got} delta={delta} {extra}"
            )
    if n_fail:
        print(f"\n{n_fail} check(s) failed.")
        return 2
    if not n_checked or n_skipped:
        print(f"\nVerification incomplete: {n_checked} passed, {n_skipped} skipped.")
        return 1
    print(f"\nAll {n_checked} checks passed.")
    return 0


def _cmd_per_landmark(args: argparse.Namespace) -> int:
    from tre_viewer.data import default_field, per_landmark, per_landmark_cache_path, resolve_run

    run = resolve_run(args.arm, args.case, runs_dir=args.runs_dir)
    field = args.field or default_field(run)
    pl = per_landmark(run, field, which=args.which, pair=args.pair)
    print(
        json.dumps(
            {
                "case": pl.case,
                "field": pl.field,
                "which": pl.which,
                "pair": pl.pair,
                "frame": pl.frame,
                "registered": pl.registered_stats,
                "identity": pl.identity_stats,
                "improvement_mm": pl.improvement_mm,
                "oob": int(pl.oob_mask.sum()),
                "worst_idx": int(pl.tre_mm.argmax()),
                "worst_tre_mm": float(pl.tre_mm.max()),
                "cache": str(
                    per_landmark_cache_path(run, pl.field, pl.which, pl.pair)
                ),
            },
            indent=2,
        )
    )
    return 0


def _cmd_phase_performance(args):
    from tre_viewer.phases import evaluate_cohort, export_results
    runs = _runs(args)
    if not runs:
        raise ValueError("No matching runs")
    results = list(evaluate_cohort(runs, args.stage, infer=args.infer_voxelmap,
                                  stride=args.stride, device=args.device, real_runs_dir=args.real_runs_dir))
    export_results(Path(args.output), results, expected_cells=len(runs) * 10)
    for result in results:
        score = f"mean {result.metric('mean'):.3f} mm" if result.landmarks else result.reason
        print(f"{result.run.run_root.name} {result.phase}: {result.status} {score}")
    print(f"Saved {len(results)} cells to {args.output}")
    return 0 if any(r.status == "ok" for r in results) and not any(r.status == "error" for r in results) else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tre_viewer",
        description="Inspect DIR-Lab registration: browse cases, view overlays, and verify TRE.",
    )
    p.add_argument("--debug", action="store_true", help="Show a traceback for troubleshooting")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="Check dependencies, coordinate adapter and data location").set_defaults(func=_cmd_doctor)

    lp = sub.add_parser("list-runs", help="Discover arms/*/runs/*")
    lp.add_argument("--arm", default=None, help="A1 or A1_oracle_dirlab")
    lp.add_argument(
        "--runs-dir",
        default=None,
        help="List cases under an arbitrary runs folder instead of arms/*",
    )
    lp.add_argument(
        "--require-summary",
        action="store_true",
        help="Only runs that already have tre_summary.json",
    )
    lp.set_defaults(func=_cmd_list_runs)

    vp = sub.add_parser("verify", help="Match adapter means to tre_summary / A0")
    vp.add_argument("--arm", default="A1")
    vp.add_argument("--runs-dir", default=None, help="Check an external runs folder")
    vp.add_argument("--case", type=int, choices=range(1, 11), default=None)
    vp.add_argument(
        "--field",
        default=None,
        help="Single field (default: all disk fields except voxelmap_ckpt)",
    )
    vp.add_argument("--which", choices=("75", "300"), default="75")
    vp.add_argument(
        "--pair", choices=("T00_T50", "T50_T00"), default="T00_T50"
    )
    vp.add_argument("--atol", type=_nonnegative_float, default=1e-6)
    vp.set_defaults(func=_cmd_verify)

    pp = sub.add_parser("per-landmark", help="Compute/cache per-landmark TRE")
    pp.add_argument("--arm", default="A1")
    pp.add_argument("--runs-dir", default=None, help="Compute TRE for an external runs folder")
    pp.add_argument("--case", type=int, choices=range(1, 11), required=True)
    pp.add_argument("--field", default=None)
    pp.add_argument("--which", choices=("75", "300"), default="75")
    pp.add_argument(
        "--pair", choices=("T00_T50", "T50_T00"), default="T00_T50"
    )
    pp.set_defaults(func=_cmd_per_landmark)

    phases = sub.add_parser("phase-performance", help="Evaluate all respiratory phases using 75 landmarks")
    phases.add_argument("--arm", default="A3")
    phases.add_argument("--runs-dir", default=None)
    phases.add_argument("--case", type=int, choices=range(1, 11), default=None)
    phases.add_argument("--stage", choices=("synth", "voxelmap"), default="synth")
    phases.add_argument("--output", required=True, help="JSON output; unavailable values are null")
    phases.add_argument("--infer-voxelmap", action="store_true", help="Infer missing per-phase caches using best.pt")
    phases.add_argument("--real-runs-dir", default=None, help="Real A1 runs directory for downstream evaluation (never synthetic training DRRs)")
    phases.add_argument("--device", default="cuda", help="PyTorch device for optional inference")
    phases.add_argument("--stride", type=int, default=10, help="Projection sampling stride")
    phases.set_defaults(func=_cmd_phase_performance)

    vp = sub.add_parser("view", help="Launch the interactive TRE viewer")
    vp.add_argument("--arm", default="A1")
    vp.add_argument("--case", type=int, choices=range(1, 11), default=1)
    vp.add_argument(
        "--runs-dir",
        default=None,
        help="Folder of DIR_Cxx runs (or arm root / …/runs). Enables case dropdown.",
    )
    vp.add_argument("--field", default=None)
    vp.add_argument("--which", choices=("75", "300"), default="75")
    vp.add_argument(
        "--pair", choices=("T00_T50", "T50_T00"), default="T00_T50"
    )
    vp.set_defaults(func=_cmd_view)

    sp = sub.add_parser("smoke", help="Offscreen smoke test (no GUI loop)")
    sp.add_argument("--arm", default="A1")
    sp.add_argument("--case", type=int, choices=range(1, 11), default=1)
    sp.set_defaults(func=_cmd_smoke)

    ps = sub.add_parser(
        "panel-smoke", help="Offscreen dock close→reopen regression test"
    )
    ps.add_argument("--arm", default="A1")
    ps.add_argument("--case", type=int, choices=range(1, 11), default=1)
    ps.set_defaults(func=_cmd_panel_smoke)

    return p


def _cmd_view(args: argparse.Namespace) -> int:
    from tre_viewer.app import main_view

    return main_view(
        arm=args.arm,
        case=args.case,
        field=args.field,
        which=args.which,
        pair=args.pair,
        runs_dir=args.runs_dir,
    )


def _cmd_smoke(args: argparse.Namespace) -> int:
    from tre_viewer.app import smoke_test

    info = smoke_test(arm=args.arm, case=args.case)
    print(json.dumps(info, indent=2))
    # Acceptance: identity ~ A0, layers non-empty
    if info["n_truth"] < 1 or info["n_vectors"] < 1:
        print("FAIL: empty landmark layers")
        return 2
    if abs(info["identity"] - 3.9133616688550745) > 1e-3 and args.case == 1:
        print("FAIL: identity TRE mismatch for case 1")
        return 2
    if "warp_improves" in info and not info["warp_improves"]:
        print("FAIL: warped lung MAE did not beat identity")
        return 2
    print("SMOKE OK")
    return 0


def _cmd_panel_smoke(args: argparse.Namespace) -> int:
    from tre_viewer.app import panel_smoke_test

    info = panel_smoke_test(arm=args.arm, case=args.case)
    print(json.dumps(info, indent=2))
    print("PANEL SMOKE OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Also accept legacy flags: python -m tre_viewer --list-runs
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0].startswith("--"):
        # rewrite --list-runs → list-runs
        flag = argv[0]
        if flag == "--list-runs":
            argv = ["list-runs", *argv[1:]]
        elif flag == "--verify":
            argv = ["verify", *argv[1:]]
        elif flag == "--view":
            argv = ["view", *argv[1:]]

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        if args.debug:
            raise
        print(f"tre_viewer: {exc}", file=sys.stderr)
        print("Run 'python -m tre_viewer doctor' for setup checks, or use --debug for details.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
