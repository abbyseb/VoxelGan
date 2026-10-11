#!/usr/bin/env python3
"""Train vs validation curves for the two X-ray (VoxelMap) models, A2-SPARE and A2-TCIA -> figs/22_voxelmap_curves.png.

Both were trained with VoxelMap_Clinical/ml/trainer.py, --val_split 0.1: random_split over ALL projection pairs of all
scans, so validation pairs come from the same patients/scans as training (not held-out patients).
A2-SPARE history is on the external drive (folder 'Incorrect DRR' = the old DIR-Lab TEST DRR orbit; the SPARE\ntraining data were correct, A2 re-verified 8 Oct, Results.md).
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = {
    "A2-SPARE (9 SPARE patients, 55 080 X-ray pairs)":
        Path("/media/abhishek/C8421E6E421E620C/Incorrect DRR/A2_generic_spare/plots_nofilm/loss_history.json"),
    "A2-TCIA (82 TCIA scans, 56 088 X-ray pairs)":
        Path(__file__).resolve().parents[1] / "arms/A2_tcia/plots_nofilm/loss_history.json",
}
DIRLAB = {"A2-SPARE": 7.12, "A2-TCIA": 5.14}

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
for a, (name, path) in zip(ax, RUNS.items()):
    h = json.load(open(path))
    ep = range(1, len(h["train_loss"]) + 1)
    a.plot(ep, h["train_loss"], color="tab:blue", label="train")
    a.plot(ep, h["val_loss"], color="tab:blue", ls="--", label="validation (same patients, random 10% of pairs)")
    short = name.split(" ")[0]
    a.set_title(f"{name}\nfinal train {h['train_loss'][-1]:.3f} · val {h['val_loss'][-1]:.3f} · DIR-Lab TRE300 {DIRLAB[short]:.2f} mm",
                fontsize=10)
    a.set_yscale("log"); a.set_xlabel("epoch"); a.grid(alpha=.3, which="both"); a.legend(fontsize=9)
ax[0].set_ylabel("training loss (log scale)")
fig.suptitle("X-ray (VoxelMap) models: train and validation merge because validation uses the same patients", fontsize=11)
plt.tight_layout()
out = Path(__file__).resolve().parent / "figs/22_voxelmap_curves.png"
plt.savefig(out, dpi=110)
print("wrote", out)
