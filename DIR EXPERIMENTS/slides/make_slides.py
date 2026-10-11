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

## Slide 2 · What was wrong: the scorer

- DIR-Lab CTs were **squeezed into the model's 160³ box** instead of resampled: voxels 1.5–3.1 mm and uneven instead of 2 mm; chests about 35% narrower than in training.
- **Brightness was scaled differently** from training.
- So the model was tested on **distorted images** it had never seen.

![orientation](figs/02_orientation_tcia_vs_dir.png)

---

## Slide 3 · Fixing the scorer

- New scorer: **resample to 2 mm, centre on the lung, same brightness as training**.
- **Verified 4 ways:** Elastix round trip (within 0.002 mm), motion direction agreement, image match, inverse check.
- **Same model: 4.70 → 4.25 mm**, the biggest single gain. Every earlier conclusion was re-checked.

![scorer fix](figs/03_scorer_fix.png)

---

## Slide 4 · What the fixed scorer revealed

- Motion **direction** was mostly right (agreement with Elastix 0.32 → 0.85).
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
- Rule of thumb: make each part about the same size at the start (image ≈ 0.02 × 10 ≈ 0.2, close to motion ≈ 0.25; smoothness a light touch).
- **Fixed before training, never tuned on DIR-Lab** (tuning on the test set would inflate the result).
- Checked later: **weight 30 was worse** (4.34 vs 4.17), so 10 stayed.

![loss balance](figs/08_loss_balance.png)

---

## Slide 9 · Why first-order smoothness, not second-order

| | **First-order** (used) | **Second-order (bending energy)** |
|---|---|---|
| Penalises | sudden changes between neighbours | bending / curving of the motion |
| Even stretch (lung expanding) | penalised a little | free |
| Typical in | deep-learning registration (VoxelMorph) | B-spline registration (Elastix) |

- Chosen as a **simple, standard, stable default** with a **light weight (0.1)**, only to stop jagged motion.
- **Not tested:** second-order might suit breathing (smooth stretching) slightly better; expected effect small.

![first vs second order](figs/09_first_vs_second_order.png)

---

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
