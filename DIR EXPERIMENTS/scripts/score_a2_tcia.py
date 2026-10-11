#!/usr/bin/env python3
"""Score an A2-TCIA VoxelMap checkpoint on DIR-Lab C01-C10 (zero-shot), writing ONLY under arms/A2_tcia/eval/.
Same inference as A2 (infer_voxelmap_dvf_phase01, stride 10, R3 inputs from A1 ModelTraining) and the exact-inverse
T00->T50 TRE (tre_t00_t50_exact, r3=True) used for the A1/A2 table. Reports cases 1-5 and 6-10 separately.

  python scripts/score_a2_tcia.py --ckpt arms/A2_tcia/checkpoints/epochs/epoch_014.pt --tag ep014 --gpu 0
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
DIR_EXP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(DIR_EXP / "scripts"))
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", type=Path, required=True); ap.add_argument("--tag", required=True); ap.add_argument("--gpu", type=int, default=0)
args = ap.parse_args(); os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu); os.environ.setdefault("DIRLAB_ROOT", str(DIR_EXP / "data" / "dirlab_packs"))
import torch
from eval_a1_tre import infer_voxelmap_dvf_phase01
from eval_a1_a2_tre_v2 import tre_t00_t50_exact
out = DIR_EXP / "arms" / "A2_tcia" / "eval" / args.tag; out.mkdir(parents=True, exist_ok=True)
A2SAVED = json.load(open(DIR_EXP / "arms" / "A2_generic_spare" / "results" / "cohort_tre_v2_exact_inverse.json"))
res = {}
for c in range(1, 11):
    mt = DIR_EXP / "arms" / "A1_oracle_dirlab" / "runs" / f"DIR_C{c:02d}" / "ModelTraining" / "train" / f"DIR_C{c:02d}"
    dvf, n = infer_voxelmap_dvf_phase01(args.ckpt.resolve(), mt, "cuda", stride=10)
    np.save(out / f"C{c:02d}_dvf_phase01_mean.npy", dvf.astype(np.float32))
    r = {w: tre_t00_t50_exact(dvf, c, w, True) for w in ("75", "300")}
    res[c] = r; print(f"C{c:02d} {json.dumps({w: (round(v, 3) if isinstance(v, float) else v) for w, v in r.items()})[:160]}", flush=True)
json.dump({"ckpt": str(args.ckpt), "cases": {str(k): v for k, v in res.items()}}, open(out / "tre_exact.json", "w"), indent=1, default=float)
print("written", out)
