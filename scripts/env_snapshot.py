#!/usr/bin/env python3
"""Snapshot and lock the training environment.

Usage:
  python3 scripts/env_snapshot.py                 # rewrite environment-lock.txt at repo root
  python3 scripts/env_snapshot.py --run-dir DIR   # write DIR/env.json for one run
  python3 scripts/env_snapshot.py --hold          # apt-mark hold GPU stack (sudo)
  python3 scripts/env_snapshot.py --unhold        # release the hold (sudo)

From a training script:
  sys.path.insert(0, "<repo>/scripts"); from env_snapshot import record_env
  record_env(run_dir)
"""
import argparse
import datetime
import json
import platform
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HOLD_PACKAGES = [
    "lambda-stack-cuda",
    "nvidia-driver-570",
    "python3-torch-cuda",
    "python3-torchvision-cuda",
]


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect():
    info = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "argv": sys.argv,
        "git_commit": _run(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "git_dirty": bool(_run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"])),
        "driver": (_run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]) or "").split("\n")[0] or None,
        "gpus": _run(["nvidia-smi", "--query-gpu=index,name,power.limit", "--format=csv,noheader"]),
        "lambda_stack": _run(["dpkg-query", "-W", "-f=${Version}", "lambda-stack-cuda"]),
        "held_packages": _run(["apt-mark", "showhold"]),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
    except ImportError:
        info["torch"] = None
    freeze = _run([sys.executable, "-m", "pip", "freeze"]) or ""
    info["packages"] = freeze.splitlines()
    return info


def record_env(run_dir):
    """Write env.json into run_dir; call once at the start of training."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    info = collect()
    (run_dir / "env.json").write_text(json.dumps(info, indent=2))
    if info["git_dirty"]:
        diff = _run(["git", "-C", str(REPO), "diff", "HEAD"]) or ""
        (run_dir / "git_diff.patch").write_text(diff)
    return info


def write_lockfile():
    info = collect()
    header = [
        f"# Environment snapshot {info['timestamp']}",
        f"# python: {info['python']} ({info['python_executable']})",
        f"# torch: {info.get('torch')} cuda {info.get('torch_cuda')} cudnn {info.get('cudnn')}",
        f"# driver: {info['driver']}",
        f"# lambda-stack-cuda: {info['lambda_stack']}",
        f"# gpus: {(info['gpus'] or '').replace(chr(10), ' | ')}",
        f"# held: {(info['held_packages'] or 'none').replace(chr(10), ', ')}",
    ]
    path = REPO / "environment-lock.txt"
    path.write_text("\n".join(header + info["packages"]) + "\n")
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--run-dir", help="write env.json into this run directory")
    g.add_argument("--hold", action="store_true", help="apt-mark hold the GPU stack")
    g.add_argument("--unhold", action="store_true", help="release the apt hold")
    a = p.parse_args()
    if a.hold or a.unhold:
        sys.exit(subprocess.call(["sudo", "apt-mark", "hold" if a.hold else "unhold", *HOLD_PACKAGES]))
    if a.run_dir:
        record_env(a.run_dir)
        print(f"wrote {Path(a.run_dir) / 'env.json'}")
    else:
        write_lockfile()


if __name__ == "__main__":
    main()
