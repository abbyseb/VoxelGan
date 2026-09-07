#!/usr/bin/env python3
"""dirlab_tre.py — target registration error on the DIR-Lab 4DCT dataset."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

_DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data" / "dirlab_packs"
ROOT = Path(os.environ.get("DIRLAB_ROOT", str(_DEFAULT_ROOT)))

# (nx, ny, nz) voxels and (dx, dy, dz) mm, as distributed. z is the slice (SI)
# axis. Landmark columns are in this same (x, y, z) order, so spacing applies
# straight across with no transposition.
CASE_INFO = {
    1: ((256, 256, 94), (0.97, 0.97, 2.5)),
    2: ((256, 256, 112), (1.16, 1.16, 2.5)),
    3: ((256, 256, 104), (1.15, 1.15, 2.5)),
    4: ((256, 256, 99), (1.13, 1.13, 2.5)),
    5: ((256, 256, 106), (1.10, 1.10, 2.5)),
    6: ((512, 512, 128), (0.97, 0.97, 2.5)),
    7: ((512, 512, 136), (0.97, 0.97, 2.5)),
    8: ((512, 512, 128), (0.97, 0.97, 2.5)),
    9: ((512, 512, 128), (0.97, 0.97, 2.5)),
    10: ((512, 512, 120), (0.97, 0.97, 2.5)),
}

# DIR-Lab's published identity TRE, T00 -> T50, 300-point sets, mean mm.
# The coordinate chain is verified against these and nothing else.
PUBLISHED = {
    1: 3.89,
    2: 4.34,
    3: 6.94,
    4: 9.83,
    5: 7.48,
    6: 10.89,
    7: 11.03,
    8: 14.99,
    9: 7.92,
    10: 7.30,
}

PHASES = [f"T{10 * i:02d}" for i in range(10)]


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def read_points(path):
    """(N, 3) zero-based voxel indices (x, y, z) from a DIR-Lab landmark file."""
    rows = [r.split() for r in Path(path).read_text().strip().splitlines() if r.strip()]
    return np.array([[float(v) for v in r[:3]] for r in rows], dtype=np.float64) - 1.0


def _pack_dir(case: int) -> Path:
    """CaseNPack, or CaseNDeploy (case 8), under ROOT."""
    for name in (f"Case{case}Pack", f"Case{case}Deploy"):
        p = ROOT / name
        if p.is_dir():
            return p
    raise SystemExit(
        f"no pack at {ROOT / f'Case{case}Pack'}. Set DIRLAB_ROOT (currently {ROOT})."
    )


def _find(case, *patterns):
    """First file under CaseNPack matching any glob, case-insensitively.

    Pack layouts are not uniform: case 1 has ExtremePhases/Case1_300_T00_xyz.txt
    while case 8 has Case8Deploy/extremePhases/case8_dirLab300_T00_xyz.txt. A
    recursive case-insensitive search is the only thing that handles all ten
    without a per-case table.
    """
    pack = _pack_dir(case)
    for pat in patterns:
        rx = re.compile(pat.replace("*", ".*"), re.IGNORECASE)
        hits = sorted(p for p in pack.rglob("*.txt") if rx.fullmatch(p.name))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"case {case}: nothing matching {patterns} under {pack}")


def landmarks_300(case, phase):
    """(300, 3) extreme-phase set. phase is 'T00' or 'T50'."""
    if phase not in ("T00", "T50"):
        raise ValueError(f"300-point sets exist only for T00/T50, not {phase}")
    return read_points(_find(case, f"*300*{phase}*xyz*.txt", f"*{phase}*xyz*.txt"))


def landmarks_75(case, phase):
    """(75, 3) Sampled4D set for one phase."""
    return read_points(_find(case, f"*4D-75*{phase}*.txt"))


def phases_75(case):
    """Phases with a 75-point set, in acquisition order."""
    out = []
    for ph in PHASES:
        try:
            _find(case, f"*4D-75*{ph}*.txt")
        except FileNotFoundError:
            continue
        out.append(ph)
    return out


# ---------------------------------------------------------------------------
# TRE
# ---------------------------------------------------------------------------
def tre_mm(pred, true, spacing):
    """Per-landmark TRE in mm. pred/true are (N, 3) voxel indices (x, y, z)."""
    d = (np.asarray(pred) - np.asarray(true)) * np.asarray(spacing)
    return np.linalg.norm(d, axis=-1)


def stats(d):
    return dict(
        mean=float(d.mean()),
        std=float(d.std()),
        p50=float(np.percentile(d, 50)),
        p95=float(np.percentile(d, 95)),
        max=float(d.max()),
        n=int(d.size),
    )


def sample_dvf(dvf, pts_xyz):
    """Trilinear sample of a dense field at (N, 3) voxel positions (x, y, z).

    `dvf` is (nz, ny, nx, 3): the displacement at each voxel of the native
    grid, given as (dx, dy, dz) in VOXELS. Positions outside the grid are
    clamped to the edge, which is the same thing grid_sample's border padding
    does -- a landmark should never be out there, so it is not silently
    tolerated: `evaluate_dvf` checks and warns.
    """
    nz, ny, nx = dvf.shape[:3]
    # positions are (x, y, z); the array is indexed [z, y, x]
    p = np.stack([pts_xyz[:, 2], pts_xyz[:, 1], pts_xyz[:, 0]], axis=1)
    hi = np.array([nz - 1, ny - 1, nx - 1], dtype=np.float64)
    p = np.clip(p, 0.0, hi)
    lo = np.floor(p).astype(np.int64)
    lo = np.minimum(lo, (hi - 1).astype(np.int64).clip(min=0))
    f = p - lo
    out = np.zeros((len(p), 3), dtype=np.float64)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                w = (
                    np.where(dz == 1, f[:, 0], 1 - f[:, 0])
                    * np.where(dy == 1, f[:, 1], 1 - f[:, 1])
                    * np.where(dx == 1, f[:, 2], 1 - f[:, 2])
                )
                idx = (
                    np.minimum(lo[:, 0] + dz, nz - 1),
                    np.minimum(lo[:, 1] + dy, ny - 1),
                    np.minimum(lo[:, 2] + dx, nx - 1),
                )
                out += w[:, None] * dvf[idx]
    return out


def evaluate_dvf(case, dvf, src_phase="T00", dst_phase="T50", which="75"):
    """TRE after displacing the source landmarks by a dense field.

    The field must map SOURCE positions to TARGET positions directly:
    pred = src + dvf(src). If your field is defined the other way round (the
    warp that pulls the target back to the source), invert it before calling
    this -- getting that backwards typically inflates TRE by ~1-2 mm on the
    large-motion cases, which looks like a mediocre result rather than a bug.
    """
    load = landmarks_75 if which == "75" else landmarks_300
    src, dst = load(case, src_phase), load(case, dst_phase)
    (nx, ny, nz), spacing = CASE_INFO[case]
    if dvf.shape[:3] != (nz, ny, nx):
        raise SystemExit(
            f"case {case}: field is {dvf.shape[:3]}, expected (nz, ny, nx) = "
            f"{(nz, ny, nx)}. The field must be on the native DIR-Lab grid."
        )
    oob = ((src < 0) | (src > np.array([nx - 1, ny - 1, nz - 1]))).any(axis=1)
    if oob.any():
        print(
            f"  WARNING: {int(oob.sum())} source landmarks lie outside the "
            f"volume; their displacement is clamped and their TRE is not a "
            f"measurement."
        )
    pred = src + sample_dvf(dvf, src)
    return stats(tre_mm(pred, dst, spacing)), stats(tre_mm(src, dst, spacing))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _row(label, s, extra=""):
    return (
        f'  {label:<12s} {s["n"]:5d}  {s["mean"]:7.3f}  {s["std"]:7.3f}  '
        f'{s["p50"]:7.3f}  {s["p95"]:7.3f}  {s["max"]:7.3f}{extra}'
    )


HEADER = "  set             n     mean      std      p50      p95      max"


def report_identity(case):
    (nx, ny, nz), spacing = CASE_INFO[case]
    pack = _pack_dir(case)
    print(f"\nCase {case}: {nx}x{ny}x{nz} voxels at " f"{spacing[0]}x{spacing[1]}x{spacing[2]} mm")
    print(f"\nIdentity TRE (no registration), mm")
    print(HEADER)

    ok = True
    try:
        d = tre_mm(landmarks_300(case, "T00"), landmarks_300(case, "T50"), spacing)
        s = stats(d)
        pub = PUBLISHED[case]
        delta = s["mean"] - pub
        flag = "  OK" if abs(delta) < 0.05 else f"  MISMATCH (published {pub})"
        ok = abs(delta) < 0.05
        print(_row("300pt T00-T50", s, f"   published {pub:5.2f}{flag}"))
    except FileNotFoundError as e:
        print(f"  300-point set unavailable: {e}")

    phs = phases_75(case)
    if phs:
        ref = landmarks_75(case, phs[0])
        for ph in phs:
            s = stats(tre_mm(ref, landmarks_75(case, ph), spacing))
            note = "  (reference)" if ph == phs[0] else ""
            print(_row(f"75pt {phs[0]}-{ph}", s, note))
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", type=int, default=None, help="DIR-Lab case 1-10; omit with --check for all")
    p.add_argument("--check", action="store_true", help="verify identity TRE against the published values")
    p.add_argument(
        "--pred",
        default=None,
        help="predicted landmark positions, DIR-Lab format " "(one-based x y z), aligned with the source set",
    )
    p.add_argument(
        "--dvf",
        default=None,
        help="dense field .npy, (nz, ny, nx, 3) as (dx, dy, dz) in " "voxels; pred = src + dvf(src)",
    )
    p.add_argument("--phase", default="T50", help="target phase (default T50)")
    p.add_argument("--src-phase", default="T00", help="source phase")
    p.add_argument("--set", choices=("75", "300"), default="75", dest="which", help="landmark set for --pred/--dvf")
    p.add_argument("--root", default=None, help="override DIRLAB_ROOT")
    A = p.parse_args()

    global ROOT
    if A.root:
        ROOT = Path(A.root)
    if A.case is not None and A.case not in CASE_INFO:
        p.error(f"--case must be one of {sorted(CASE_INFO)}, got {A.case}")

    if A.check and A.case is None:
        print(f"Checking all cases against published identity TRE  [{ROOT}]")
        bad = []
        for c in sorted(CASE_INFO):
            try:
                if not report_identity(c):
                    bad.append(c)
            except SystemExit as e:
                print(f"\nCase {c}: {e}")
                bad.append(c)
        print()
        if bad:
            sys.exit(f"FAILED on cases {bad}: the coordinate chain is wrong.")
        print("all cases match the published identity TRE")
        return

    if A.case is None:
        p.error("--case is required (or use --check with no case for all ten)")

    load = landmarks_75 if A.which == "75" else landmarks_300
    (nx, ny, nz), spacing = CASE_INFO[A.case]

    if A.pred or A.dvf:
        src = load(A.case, A.src_phase)
        dst = load(A.case, A.phase)
        ident = stats(tre_mm(src, dst, spacing))
        if A.pred:
            pred = read_points(A.pred)
            if pred.shape != src.shape:
                raise SystemExit(
                    f"--pred has {pred.shape[0]} points but the {A.which}-point "
                    f"{A.src_phase} set has {src.shape[0]}. They must "
                    f"correspond row by row."
                )
            got = stats(tre_mm(pred, dst, spacing))
        else:
            dvf = np.load(A.dvf)
            if dvf.shape == (3, nz, ny, nx):
                dvf = np.moveaxis(dvf, 0, -1)  # accept channel-first too
            got, ident = evaluate_dvf(A.case, dvf, A.src_phase, A.phase, A.which)

        print(f"\nCase {A.case}: {A.src_phase} -> {A.phase}, " f"{A.which}-point set")
        print(HEADER)
        print(_row("identity", ident))
        print(_row("registered", got))
        chg = ident["mean"] - got["mean"]
        pct = 100.0 * chg / ident["mean"] if ident["mean"] > 1e-9 else float("nan")
        print(f"\n  improvement: {chg:+.3f} mm ({pct:+.1f}%)")
        return

    ok = report_identity(A.case)
    if A.check and not ok:
        sys.exit("identity TRE does not match the published value.")


if __name__ == "__main__":
    main()
