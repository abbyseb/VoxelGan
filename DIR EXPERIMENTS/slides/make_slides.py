"""Builds DIR EXPERIMENTS/slides/SLIDES_CONTENT.md and its figures (figs/) from the saved result JSONs."""
import json, shutil, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
V = Path("/home/abhishek/Voxel_GAN"); G = V / "PopulationStudy/ClinicalExperiments/Grid160"; Q = G / "TCIA3/DecoderCRB/plots/qc_dir_oracle"
A = G / "TCIA_lite/analysis_2026-10-03"; OUT = V / "DIR EXPERIMENTS/slides"; F = OUT / "figs"; F.mkdir(parents=True, exist_ok=True)
C = list(range(1, 11)); J = lambda p: json.load(open(p))
IDN = [3.89, 4.34, 6.94, 9.83, 7.48, 10.89, 11.03, 14.99, 7.92, 7.30]
# ---------- data ----------
rq = J(Q / "rescore_oracle_iso2mm_v3_20260930_140122.json")["rescore"]
tcia3 = [rq["TCIA3@100"]["per_case"][str(c)]["tre300"] for c in C]
fe = J(G / "TCIA_lite/final_eval_20260930_140158.json")["runs"]
lite = {k: [fe[k]["per_case_mean"][str(c)] for c in C] for k in ("A2", "A2_small", "A2_deep128", "A2_full160")}
sf = J(A / "seeds_final.json")["per_patient"]
seedmean = lambda k: [np.mean([sf[f"{k}_s1|{v}"]["dir"][str(c)] for v in ["x"]] if False else [(sf[f"{k}_s1|{VAR}"]["dir"][str(c)] + sf[f"{k}_s2|{VAR}"]["dir"][str(c)]) / 2]) for c in C]
VAR = "plain"; old_p, new_p = seedmean("old"), seedmean("new")
VAR = "flip"; old_f, new_f = seedmean("old"), seedmean("new")
cl = J(A / "all82_clean_old.json")
small = {k: cl[k]["dp"] for k in cl}
s1 = J(A / "lncc_ft_ctrl.json")["start (hybrid s1 ep40)"]["dir_plain"]
def tcia_eval(tag): d = J(V / f"DIR EXPERIMENTS/arms/A2_tcia/eval/{tag}/tre_exact.json")["cases"]; return [d[str(c)]["300"]["registered"]["mean"] for c in C]
a2 = tcia_eval("check_A2_spare"); a2t = list(np.mean([tcia_eval(f"ep0{e}") for e in range(46, 51)], 0))
combo = J(A / "combo_small_big.json")["PRIMARY: small s1+s2 + big OLD s1+s2"]["dir_flip"]
popi = J(A / "seeds_final.json")["per_patient"]
# ---------- figures ----------
def bars(names, vals, title, fname, color="tab:blue", ylab="DIR-Lab TRE300 (mm)", hl=None):
    fig, ax = plt.subplots(figsize=(8, 4)); cs = [("tab:orange" if (hl is not None and i == hl) else color) for i in range(len(vals))]
    b = ax.bar(names, vals, color=cs)
    for r, v in zip(b, vals): ax.text(r.get_x() + r.get_width() / 2, v + 0.08, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_ylabel(ylab); ax.set_title(title); ax.set_ylim(0, max(vals) * 1.18); ax.grid(axis="y", alpha=.3); plt.xticks(rotation=15, ha="right")
    plt.tight_layout(); plt.savefig(F / fname, dpi=110); plt.close(fig)
bars(["No motion", "TCIA3 (old scorer)", "TCIA3 (fixed scorer)", "Best single model", "Pre-set combination"], [8.46, 4.70, 4.25, 4.07, 3.95], "Landmark error from one planning CT", "01_overview.png", hl=3)
shutil.copy(Q / "orientation_tcia_s1_vs_dir_c01.png", F / "02_orientation_tcia_vs_dir.png")
bars(["Old scorer\n(squeezed DIR CT)", "Fixed scorer\n(2 mm, lung-centred)"], [4.70, 4.25], "Same model (TCIA3), two scorers", "03_scorer_fix.png", hl=1)
bars(["Model as is", "One shared size\nfor everyone", "Each patient's\ntrue breath depth"], [4.25, 4.27, 3.60], "Scale test: the error is breath size (TCIA3)", "04_scale_test.png", hl=2)
bars(["64³ patches\n(baseline)", "Half width\n0.27 M", "Deeper\n(128³)", "Whole lung\n160³"], [np.mean(lite["A2"]), np.mean(lite["A2_small"]), np.mean(lite["A2_deep128"]), np.mean(lite["A2_full160"])], "Small setup: network size and view", "05_lite_variants.png", hl=3)
# small-setup training curves (train / validation L1 from each run's log)
import re as _re
LR = {"A2": ("A′ (64³, 1.07 M)", "tab:blue"), "A2_small": ("A′-small (64³, half width, 0.27 M)", "tab:red"),
      "A2_deep128": ("A′-deep128 (128³, +1 level, 2.20 M)", "tab:purple"), "A2_full160": ("A′-full160 (whole 160³, 1.07 M)", "tab:green")}
curves = {}
for k in LR:
    rows = [_re.search(r"Epoch: (\d+) \| train MAE: ([\d.]+) \| val MAE: ([\d.]+)", l) for l in open(G / f"TCIA_lite/run_{k}/logs/train_decoder.log")]
    curves[k] = np.array([[float(x) for x in m.groups()] for m in rows if m])
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
for k, (lab, col) in LR.items():
    e, tr, va = curves[k].T
    ax[0].plot(e, tr, color=col, label=lab); ax[1].plot(e, va, color=col, label=lab)
for a, t in zip(ax, ["Training error (seen patients)", "Validation error (4 new patients)"]):
    a.set_title(t); a.set_xlabel("epoch"); a.grid(alpha=.3)
ax[0].set_ylabel("lung-masked motion L1 (2 mm voxels)"); ax[1].legend(fontsize=9)
fig.suptitle("Small setup: training curves (validation views differ: 64³ crops / 128³ crop / whole 160³)", fontsize=11)
plt.tight_layout(); plt.savefig(F / "05b_lite_training_curves.png", dpi=110); plt.close(fig)
for k in LR: shutil.copy(G / f"TCIA_lite/run_{k}/DecoderCRB/plots/crb_dec_mae_tcia_lite_run_{k.lower()}.png", F / f"05c_curve_{k}.png")
fin = {k: (curves[k][-1, 1], curves[k][-1, 2], int(curves[k][-1, 0])) for k in LR}
fig, ax = plt.subplots(figsize=(8, 4)); ideas = ["Push labels", "Combine old models", "EPE loss", "Brightness match", "Mirror (flip)", "Hybrid loss (small)"]; g = [-0.09, 0.02, -0.06, 0.39, -0.18, -0.30]
ax.barh(ideas, g, color=["tab:green" if x <= -0.1 else ("tab:red" if x > 0.05 else "lightgrey") for x in g]); ax.axvline(-0.1, ls="--", color="k", lw=1); ax.text(-0.105, 5.4, "pass bar −0.1 mm", ha="right", fontsize=9)
for i, x in enumerate(g): ax.text(x + (0.02 if x >= 0 else -0.02), i, f"{x:+.2f}", va="center", ha="left" if x >= 0 else "right", fontsize=9)
ax.set_xlabel("change in DIR-Lab TRE (mm, negative = better)"); ax.set_title("Ideas scoreboard (green = passed the bar)"); ax.set_xlim(-0.45, 0.5); plt.tight_layout(); plt.savefig(F / "06_scoreboard.png", dpi=110); plt.close(fig)
fig, ax = plt.subplots(figsize=(9, 2.6)); ax.axis("off")
for x, t in ((0.08, "CT phase 6\n(input)"), (0.36, "network\npredicts motion u"), (0.64, "warp CT 6 with u"), (0.9, "compare with\nreal CT phase 1")):
    ax.text(x, 0.55, t, ha="center", va="center", fontsize=11, bbox=dict(boxstyle="round", fc="#e8f0fe", ec="tab:blue"))
for x in (0.2, 0.48, 0.76): ax.annotate("", xy=(x + 0.05, 0.55), xytext=(x, 0.55), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.text(0.36, 0.08, "motion error vs Elastix (old loss)", ha="center", fontsize=10, color="grey"); ax.text(0.9, 0.08, "image match (new part)", ha="center", fontsize=10, color="tab:blue")
plt.tight_layout(); plt.savefig(F / "07_hybrid_loss.png", dpi=110); plt.close(fig)
bars(["motion L1", "image L1 × 10", "smoothness × 0.1"], [0.25, 0.20, 0.0003], "Loss parts at the start of training (similar size by design)", "08_loss_balance.png", ylab="loss value")
x = np.linspace(0, 1, 200); fig, ax = plt.subplots(1, 3, figsize=(11, 3))
for a, (y, t, p) in zip(ax, ((x + 0.06 * np.sign(np.sin(40 * x)), "jagged motion", "1st order: high · 2nd order: high"), (1.2 * x, "even stretch", "1st order: some · 2nd order: zero"), (np.sin(np.pi * x), "smooth bend", "1st order: some · 2nd order: some"))):
    a.plot(x, y, lw=2); a.set_title(t); a.set_xlabel("position"); a.set_ylabel("displacement"); a.text(0.5, -0.32, p, transform=a.transAxes, ha="center", fontsize=9)
plt.tight_layout(); plt.savefig(F / "09_first_vs_second_order.png", dpi=110); plt.close(fig)
for src, dst in (("explanatory_03_TCIA3.5_old_s1_ep100.png", "10_big_model_example.png"), ("explanatory_08_small_hybrid_s1_ep40.png", "11_small_model_example.png"), ("small_model_training_curves.png", "12_small_training_curves.png")):
    shutil.copy(A / "figures" / src, F / dst)

# ---------- later-story figures ----------
def gbars(groups, series, title, fname, ylab="DIR-Lab TRE300 (mm)", ylim=None):
    fig, ax = plt.subplots(figsize=(8.5, 4)); w = 0.8 / len(series); xs = np.arange(len(groups))
    for i, (lab, vals) in enumerate(series.items()):
        b = ax.bar(xs + i * w - 0.4 + w / 2, vals, w, label=lab)
        for r, v in zip(b, vals): ax.text(r.get_x() + r.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(groups); ax.set_ylabel(ylab); ax.set_title(title); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
    if ylim: ax.set_ylim(*ylim)
    plt.tight_layout(); plt.savefig(F / fname, dpi=110); plt.close(fig)
gbars(["Seed 1", "Seed 2"], {"small old loss": [4.47, 4.47], "small hybrid": [4.17, 4.16]}, "Small model: hybrid loss repeats on both seeds (DIR-Lab)", "13_small_hybrid_seeds.png", ylim=(3.5, 4.8))
gbars(["Seed 1", "Seed 2", "Mean"], {"big old + flip": [4.02, 4.12, 4.07], "big hybrid + flip": [3.87, 4.37, 4.12]}, "Big model: hybrid result does NOT repeat on seed 2", "14_big_seeds.png", ylim=(3.5, 4.6))
bars(["Big old pair", "Small hybrid pair", "Pre-set combination\n(2 small + 2 big)"], [4.08, 4.06, 3.95], "Combining small and big models (DIR-Lab, flip)", "15_combination.png", hl=2)
gbars(["DIR-Lab", "POPI"], {"control (5 extra epochs)": [4.157, 4.829], "+ local NCC": [4.134, 4.828]}, "Local NCC top-up vs matched control (small model)", "16_local_ncc.png", ylim=(3.8, 5.0))
gbars(["DIR-Lab", "POPI (flip)"], {"65 scans": [4.16, 4.96], "all 82": [4.50, 4.92], "78 (4 bad labels removed)": [4.27, 4.73]}, "More scans help only if labels are clean (small hybrid)", "17_label_cleaning.png", ylim=(3.5, 5.3))
bars(["No motion", "A2: X-ray model\nSPARE 9 patients", "A2-TCIA: X-ray model\nTCIA 82 scans", "Single-CT model"], [8.46, 7.12, 5.14, 4.07], "X-ray (VoxelMap) models vs single CT (DIR-Lab)", "18_xray_models.png", hl=3)
bars(["No motion", "Fuerst et al.\n(2 CTs, physics)", "Scaled Elastix\n(2 CTs, no training)"], [5.89, 3.88, 2.35], "With two CTs a simple baseline wins (DIR-Lab C6-10, in-between phases)", "19_two_ct.png", hl=2)
# ---------- tables ----------
def row(name, v): return f"| {name} | " + " | ".join(f"{x:.2f}" for x in v) + f" | **{np.mean(v):.2f}** | {np.mean(v[:5]):.2f} | {np.mean(v[5:]):.2f} |"
H = "| Model | " + " | ".join(f"C{c}" for c in C) + " | **Mean** | C1–5 | C6–10 |\n|---" * 1 + "|" + "---|" * 13
def table(rows): return H + "\n" + "\n".join(row(n, v) for n, v in rows)
T_early = table([("No motion", IDN), ("TCIA3 (fixed scorer)", tcia3), ("Lite 64³ patches", lite["A2"]), ("Lite half width", lite["A2_small"]), ("Lite deeper 128³", lite["A2_deep128"]), ("Lite whole lung 160³", lite["A2_full160"])])
T_small = table([("Small, old loss (65 scans)", small["old 65 scans"]), ("Small, hybrid seed 1", s1), ("Small, hybrid seed 2", small["hybrid 65 scans (s2)"]), ("Small, hybrid all 82 scans", small["hybrid all 82 (s2)"]), ("Small, hybrid 78 clean scans", small["hybrid 78 clean (s2)"]), ("Small, old loss all 82", small["old all 82 (s2 seed)"])])
T_big = table([("Big, old loss", old_p), ("Big, hybrid loss", new_p), ("**Big, old loss + flip (best single)**", old_f), ("Big, hybrid + flip", new_f), ("Pre-set combination (2 small + 2 big, flip)", combo)])
T_xray = table([("A2: X-ray model, SPARE 9 patients", a2), ("A2-TCIA: same model, TCIA 82 scans", a2t), ("Single-CT best (for comparison)", old_f)])
md = f"""# Slide content: single-CT lung motion (24 Sep – 11 Oct 2026)

Simple slide-by-slide content. All errors are DIR-Lab landmark error (TRE300, mm; lower is better) unless noted. Figures in `figs/` (made by `make_slides.py` from the saved results).

---

## Slide 1 · Where we started (around 24 Sep)

- **Goal:** predict lung breathing motion from **one planning CT**.
- **Test:** DIR-Lab, 10 patients, 300 expert landmarks each; score = landmark error (TRE) in mm.
- **First model (TCIA3):** about 4.7 mm. "Make it move more" fixes (AmpHead, MagFT, MagMatch) gave small or no gains.

![overview](figs/01_overview.png)

---

## Slide 2 · What was wrong: the DIR-Lab scorer

*Applies to every TCIA-trained model scored on DIR-Lab before 29 Sep (TCIA3, TCIA3.5 and the fine-tunes).*

- **Found in a code audit (29 Sep).** Warning sign: predicted motion was **too small** (0.65 of the real size; C08 0.44), and "move more" fixes (MagFT, MagMatch, AmpHead) did not help.
- DIR-Lab CTs were **squeezed or stretched into the model's 160³ box** instead of resampled: voxels 1.5–3.1 mm and uneven instead of 2 mm; chests up to about 35% narrower (C06–C10) or wider (C01–C05) than in training.
- **Brightness was scaled differently** from training (DIR-Lab came out much brighter).
- So the model was tested on **distorted images** it had never seen. The model and its training data were fine.

![input old vs fixed](figs/20_scorer_input_old_vs_new.png)

*Black edges on the fixed C01 = air outside the scan (C01 covers ~250 mm of the 320 mm box). C08 (scanned ~500 mm wide) is cropped at the arms; all lung and all 300 landmarks stay inside.*

---

## Slide 3 · Fixing the scorer

- New scorer: **resample to 2 mm, centre on the lung, same brightness as training**.
- **Verified 4 ways:** Elastix round trip (within 0.002 mm), motion direction agreement, image match, inverse check.
- **Same model: 4.70 → 4.25 mm**, the biggest single gain. Every earlier conclusion was re-checked.

![scorer fix](figs/03_scorer_fix.png)

---

## Slide 4 · What the fixed scorer revealed

- Same model, same measurement, all 10 cases: **direction 0.80 → 0.84** (already mostly right), **motion size 0.65 → 0.78** (this is what the fix changed).
- C01 (shallow breather) goes from 1.03 to 1.37 of the real size (too big); C08 (deep breather) stays at 0.39–0.44 (too small).

![arrows C08](figs/21_arrows_c08_old_vs_new.png)

![arrows C01](figs/21_arrows_c01_old_vs_new.png)

- The main error is **breath size**: shallow breathers get too much motion, deep breathers too little.
- With each patient's true breath depth: about **3.6 mm**. But **one CT cannot show breath depth** (lung size and shape do not predict it).

![scale test](figs/04_scale_test.png)

---

## Slide 5 · Why TCIA-lite (a fast test bench)

- A full run took **35–42 h**: too slow to test ideas.
- **TCIA-lite:** 65 scans, 4 patients held out, 40 epochs, **about 11 h**.
- Used to test one change at a time (network size, view, losses, labels).

![lite variants](figs/05_lite_variants.png)

---

## Slide 5b · Two training setups: "big" and "small"

Same network (UNetCRBDecoder, 1.07 M parameters) and base loss, different training.

| | **Big model (TCIA3.5)** | **Small model (TCIA-lite)** |
|---|---|---|
| Patients / scans | 20 / **82** | 16 / **65** (4 patients held out) |
| What it sees in training | **64³ patches** (16 per pair) | **whole lung 160³** (1 per pair) |
| Training | 100 epochs, constant learning rate | 40 epochs, learning rate lowered (cosine) |
| Augmentation | none | flip, shift, zoom, brightness, noise |
| Time per run | about 35 h | about 11 h |
| DIR-Lab (best) | **4.07** (old loss + flip, 2 seeds) | about 4.1–4.2 |

- Big model: more patients, best overall and on POPI. Small model: better on the deepest breather (C8) and stable across seeds (4.17 vs 4.16).
- They make **different mistakes**, so combining them helps (3.95 mm).

| Big model (64³ patches, old loss) | Small model (whole lung, hybrid) |
|---|---|
| ![big](figs/10_big_model_example.png) | ![small](figs/11_small_model_example.png) |

---

## Slide 5c · Bigger or smaller network?

| Network (small setup, 30 Sep) | Change | DIR-Lab |
|---|---|---|
| Baseline, 64³ patches | — | {np.mean(lite['A2']):.2f} |
| Half width (0.27 M) | smaller | {np.mean(lite['A2_small']):.2f} (worse) |
| One extra level (128³) | deeper | {np.mean(lite['A2_deep128']):.2f} |
| Whole lung 160³ (same 1.07 M) | sees more lung | **{np.mean(lite['A2_full160']):.2f}** |

- **1.07 M is the right size.** Smaller learns less; deeper only helped because it saw more lung.
- Not underfitting: training error reached the label-noise floor. **What the model sees matters more than its size.**

**Case by case (early models):**

{T_early}

---

## Slide 5d · The four small-setup networks, in detail

All four use the **same U-Net (UNetCRBDecoder)**: a plain encoder that sees only the CT, and a decoder whose blocks are told the two breathing phases (conditional residual blocks); the output is the 3-channel motion field. Same data (65 scans / 16 patients, validation 17 scans / 4 new patients), loss (lung-masked L1), Adam lr 1e-4 constant, no augmentation, seed 20260929.

| | A′ (baseline) | A′-small | A′-deep128 | A′-full160 |
|---|---|---|---|---|
| What changed | — | every channel halved | +1 level down/up, and 128³ crops | whole 160³ volume |
| Levels (image halvings) | 5 (4 halvings) | 5 (4) | **6 (5)** | 5 (4) |
| Channels, top → bottom | 16-32-32-64-64 | **8-16-16-32-32** | 16-32-32-64-64-**128** | 16-32-32-64-64 |
| Parameters | 1.07 M | **0.27 M** | **2.20 M** | 1.07 M |
| Training view | 64³ crops, 4 per pair | 64³ crops, 4 per pair | **128³ crop, 1 per pair** | **whole 160³, 1 per pair** |
| Smallest map inside the net | 4³ | 4³ | 4³ | 10³ |
| Epochs | 80 | 80 | 80 | 40 |
| Training time | 6 h 20 min | 4 h 37 min | 13 h 37 min | 10 h 24 min |
| GPU memory | — | 0.2 GB | 2.4 GB | 4.7 GB |
| Final train / val error | {fin['A2'][0]:.3f} / {fin['A2'][1]:.3f} | {fin['A2_small'][0]:.3f} / {fin['A2_small'][1]:.3f} | {fin['A2_deep128'][0]:.3f} / {fin['A2_deep128'][1]:.3f} | {fin['A2_full160'][0]:.3f} / {fin['A2_full160'][1]:.3f} |
| **DIR-Lab TRE300** (last 5 epochs) | {np.mean(lite['A2']):.2f} | {np.mean(lite['A2_small']):.2f} (worse) | {np.mean(lite['A2_deep128']):.2f} | **{np.mean(lite['A2_full160']):.2f}** |

- **Why deep128 needed 128³ crops:** each level halves the image. A 64³ crop after 5 halvings is only 2³, almost nothing to look at; 128³ keeps it at 4³, like the baseline.
- **Why full160 kept the baseline net:** to separate "sees more" from "is deeper". It matched deep128 without the extra level and was better on C08 (8.68 vs 9.71), so the whole-lung view was kept.
- **Reading the curves:** training error keeps falling while validation error is flat from about epoch 5 (A′ 0.48–0.51), so most later training is memorising. The small net memorises less (smaller gap) but does not score better on DIR-Lab. Validation numbers are measured on different views (64³ crops / one 128³ crop / whole volume), so compare their shapes, not their levels.

![training curves](figs/05b_lite_training_curves.png)

| A′ | A′-small |
|---|---|
| ![A′](figs/05c_curve_A2.png) | ![small](figs/05c_curve_A2_small.png) |
| **A′-deep128** | **A′-full160** |
| ![deep128](figs/05c_curve_A2_deep128.png) | ![full160](figs/05c_curve_A2_full160.png) |

---

## Slide 6 · Ideas that did not work

- Push-style labels, re-made labels, brightness matching, EPE loss, combining old models: each **less than 0.1 mm**.
- Conclusion: the model learns **Elastix very well, including Elastix's mistakes** (training error at the label-noise floor).

![scoreboard](figs/06_scoreboard.png)

---

## Slide 7 · Why the hybrid loss

- **Problem:** the model only copies Elastix, so it copies Elastix's errors.
- **Idea from the literature** (RMSim; noisy-label work): also check **whether the images actually line up**.
- **Hybrid loss = motion error + image match + smoothness.** The image signal does not share Elastix's mistakes.

![hybrid loss](figs/07_hybrid_loss.png)

---

## Slide 8 · How the weights were chosen (an educated guess)

- **Loss = motion + 10 × image + 0.1 × smoothness.**
- Rule of thumb: make each part about the same size (image ≈ 0.02 × 10 ≈ 0.2, close to motion ≈ 0.25; smoothness a light touch).
- **Why 10: the parts are in different units**, so they need balancing (not a correction of any error):

| Loss part | What it measures | Units | Typical size during training |
|---|---|---|---|
| Motion error (L1) | predicted motion vs Elastix | voxels (2 mm each) | about 0.25 |
| Image error (L1) | warped CT vs real CT | image brightness, scaled 0–1 | about 0.02 |

**Where these two numbers come from (measured, not assumed):**

- **Motion error ≈ 0.25** = the training motion error the old-loss main model (TCIA3.5) settles at: mean over epochs 90–100 = **0.248**.
  - Definition (`DVFLoss`): inside the lung mask, the mean of |Elastix label − prediction| over all voxels and the 3 components, in 2 mm voxels. So 0.25 ≈ 0.5 mm per component.
  - For scale: predicting no motion gives 0.685; the measured label-noise floor is about 0.165.
- **Image error ≈ 0.02** = the image term's size in practice: mean |real CT − warped CT| in the lung mask (dilated 2 voxels), on CT brightness min-max scaled to 0–1.
  - Logged values: 0.019–0.023 during small-model hybrid training; 0.027 with the Elastix labels on 64³ patches (big-model safety check).
- So 0.25 / 0.02 ≈ 12, rounded to **10**.

- The image error is naturally about 10× smaller; added as is it would be almost invisible to training. × 10 brings it to about 0.2, so **both parts have about equal influence**.
- **Fixed before training, never tuned on DIR-Lab** (tuning on the test set would inflate the result).
- Checked later: **weight 30 was worse** (4.34 vs 4.17), so 10 stayed.

![loss balance](figs/08_loss_balance.png)

---

## Slide 9 · Why first-order smoothness, not second-order

| | **First-order** (used) | **Second-order (bending energy)** |
|---|---|---|
| Name | **diffusion penalty** (squared gradient, as in VoxelMorph) | bending energy |
| Penalises | sudden changes between neighbours | bending / curving of the motion |
| Even stretch (lung expanding) | penalised a little | free |
| Typical in | deep-learning registration (VoxelMorph) | B-spline registration (Elastix) |

- Our term: mean of squared differences between neighbouring motion vectors along z, y, x (forward differences), over the **whole volume**, weight 0.1.
- Chosen as a **simple, standard, stable default** with a **light weight (0.1)**, only to stop jagged motion.
- **Not tested:** second-order might suit breathing (smooth stretching) slightly better; expected effect small.

![first vs second order](figs/09_first_vs_second_order.png)

---

---

## Slide 10 · Hybrid loss on the small model: it works, twice

- Small model (whole lung): **4.47 → 4.17 mm** (seed 1) and **4.16 mm** (seed 2), 7/10 and 9/10 cases better.
- First change in the project to pass the fixed rule (> 0.1 mm better, ≥ 7/10 cases).
- But on POPI the small hybrid was **not** better (4.80 / 5.05 vs 4.73 old loss).

![small hybrid seeds](figs/13_small_hybrid_seeds.png)

---

## Slide 11 · Hybrid loss on the big model: one lucky seed

- Seed 1 looked great: **3.87 mm** with flip. Seed 2: **4.37 mm**.
- Averaged over seeds: hybrid 4.12 vs old loss **4.07**, no improvement (16-patient test +0.02 mm, p = 0.46).
- Big-model hybrid seeds differ by **0.5 mm**; old loss only 0.1 mm. Likely causes: 64³ patches (image check sees little lung, padded edges) and a constant learning rate.
- **Lesson:** always run a second seed before claiming a gain.

![big seeds](figs/14_big_seeds.png)

---

## Slide 12 · Making the result stable

- Averaging the last 10 epochs' weights: helps the old loss a little (−0.08 mm), does not fix the hybrid seed gap.
- **Combining 2 small + 2 big models** (fixed before scoring): **3.95 mm DIR-Lab, 4.53 mm POPI**. Small and big models make different mistakes.

![combination](figs/15_combination.png)

---

## Slide 13 · Local NCC: promising on paper, tiny in practice

- Check 1: local NCC notices motion errors much better than plain image difference (rank correlation 0.84–0.89 vs 0.50–0.65), but is noisy.
- Check 2: moving toward the true landmark motion lowers local NCC by **40%** vs 8.5% for plain difference.
- Check 3 (5-epoch top-up vs matched control): **−0.02 mm DIR-Lab, 0.00 POPI**. Not kept.

![local NCC](figs/16_local_ncc.png)

---

## Slide 14 · Label quality matters

- Adding the 4 held-out patients (82 scans) made the small model **worse** (4.16 → 4.50).
- Their Elastix labels are noisier (forward-backward error +17%); one scan (S29) is clearly bad (folding +4.2 SD).
- **Removing 4 flagged scans** (78 scans): 4.27 DIR-Lab and the **best small-model POPI (4.73)**.

![label cleaning](figs/17_label_cleaning.png)

---

## Slide 15 · X-ray models (VoxelMap): better data helps, but not enough

- **A2:** VoxelMap trained on 9 SPARE patients, tested zero-shot: **7.12 mm** (weights found and re-verified).
- **A2-TCIA:** same network and settings, trained on 82 TCIA scans: **5.14 mm**.
- Still worse than the single-CT model (**4.07**) on every case, most on the big scans (C6–10: 7.13 vs 5.52).
- Likely reason: this variant only sees the two X-rays, **never the CT**, so it cannot use anatomy or scan size.

![xray models](figs/18_xray_models.png)

---

## Slide 16 · Two CTs and the literature

- With **two CTs**, scaled Elastix (no training) gives **2.0 mm** (2.35 on C6–10), beating the published physics model (Fuerst et al., 3.88).
- So knowing breath depth takes the error from about 4–5 mm to about 2 mm: **missing depth is the main cost of one CT**.
- Literature: no method found that predicts from **one CT without a breathing signal** and reports DIR-Lab/POPI landmark error. Closest: Kanamuro et al. 2025 (one CT, private data, no landmarks).

![two CT](figs/19_two_ct.png)

---

## Slide 17 · Summary

- **Best single model:** 4.07 mm DIR-Lab, 4.57 mm POPI (2 seeds). **Pre-set combination:** 3.95 / 4.53 mm.
- About **half of the motion error removed** from one CT, in the range of methods that need a breathing signal (3.3–4.2 mm).
- What helped: fixing the scorer, flip averaging, combining models, clean labels. Loss changes: no reliable gain.
- **Limit: breath depth is invisible in one CT.** Deep breathers (C8) carry most of the error.

## Appendix A · Small model, case by case (epochs 36–40)

{T_small}

![small training](figs/12_small_training_curves.png)

## Appendix B · Big model and final results, case by case (epochs 96–100, 2-seed mean)

{T_big}

POPI (6 patients, 2-seed mean): old loss + flip **{np.mean([(np.mean(list(popi['old_s1|flip']['popi'].values())) + np.mean(list(popi['old_s2|flip']['popi'].values()))) / 2]):.2f}** mm; hybrid + flip {np.mean([(np.mean(list(popi['new_s1|flip']['popi'].values())) + np.mean(list(popi['new_s2|flip']['popi'].values()))) / 2]):.2f} mm.

## Appendix C · X-ray (VoxelMap) models vs single CT, case by case

{T_xray}

A2-TCIA = epochs 46–50 mean, zero-shot, same network and settings as A2 (concatenated, sees only the two X-rays, not the CT).
"""
(OUT / "SLIDES_CONTENT.md").write_text(md); print("written", OUT / "SLIDES_CONTENT.md")
