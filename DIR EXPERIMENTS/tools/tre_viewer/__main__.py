"""CLI entry: ``python -m tre_viewer`` from ``DIR EXPERIMENTS/tools``.

Phase 0 commands:
  --list-runs
  --verify [--arm A1] [--case N] [--field elastix_mha]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow ``python -m tre_viewer`` when cwd or tools/ is on sys.path
_TOOLS = Path(__file__).resolve().parent.parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from tre_viewer.data import (  # noqa: E402
    DIR_EXP,
    discover_runs,
    per_landmark,
    verify_against_summary,
)


def _cmd_list_runs(args: argparse.Namespace) -> int:
    from tre_viewer.data import discover_runs_in_folder

    if args.runs_dir:
        runs = discover_runs_in_folder(
            args.runs_dir, require_summary=args.require_summary
        )
    else:
        arms = [args.arm] if args.arm else None
        runs = discover_runs(arms, require_summary=args.require_summary)
    if not runs:
        print("No runs found.")
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
    arms = [args.arm] if args.arm else None
    runs = discover_runs(arms)
    if args.case is not None:
        runs = [r for r in runs if r.case == args.case]
    if not runs:
        print("No matching runs.")
        return 1

    n_fail = 0
    for run in runs:
        print(f"\n=== {run.arm} C{run.case:02d} frame={run.frame} ===")
        checks = verify_against_summary(
            run,
            field=args.field,
            which=args.which,
            pair=args.pair,
            atol=args.atol,
        )
        for c in checks:
            status = "OK" if c["ok"] else "FAIL"
            if not c["ok"]:
                n_fail += 1
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
    print("\nAll checks passed.")
    return 0


def _cmd_per_landmark(args: argparse.Namespace) -> int:
    runs = discover_runs([args.arm] if args.arm else None)
    runs = [r for r in runs if r.case == args.case]
    if not runs:
        print("No matching run.")
        return 1
    run = runs[0]
    field = args.field or next(
        (f for f in run.fields_available if f.startswith("elastix")), "identity"
    )
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
                    run.run_root
                    / "tre"
                    / f"per_landmark_{pl.which}_{pl.pair}_{pl.field}.npz"
                ),
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tre_viewer",
        description="TRE Viewer (Phase 0: discovery + adapter + verify)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

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
    vp.add_argument("--case", type=int, default=None)
    vp.add_argument(
        "--field",
        default=None,
        help="Single field (default: all disk fields except voxelmap_ckpt)",
    )
    vp.add_argument("--which", choices=("75", "300"), default="75")
    vp.add_argument(
        "--pair", choices=("T00_T50", "T50_T00"), default="T00_T50"
    )
    vp.add_argument("--atol", type=float, default=1e-6)
    vp.set_defaults(func=_cmd_verify)

    pp = sub.add_parser("per-landmark", help="Compute/cache per-landmark TRE")
    pp.add_argument("--arm", default="A1")
    pp.add_argument("--case", type=int, required=True)
    pp.add_argument("--field", default=None)
    pp.add_argument("--which", choices=("75", "300"), default="75")
    pp.add_argument(
        "--pair", choices=("T00_T50", "T50_T00"), default="T00_T50"
    )
    pp.set_defaults(func=_cmd_per_landmark)

    vp = sub.add_parser("view", help="Launch napari TRE viewer (Phase 1)")
    vp.add_argument("--arm", default="A1")
    vp.add_argument("--case", type=int, default=1)
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
    sp.add_argument("--case", type=int, default=1)
    sp.set_defaults(func=_cmd_smoke)

    ps = sub.add_parser(
        "panel-smoke", help="Offscreen dock close→reopen regression test"
    )
    ps.add_argument("--arm", default="A1")
    ps.add_argument("--case", type=int, default=1)
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
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
