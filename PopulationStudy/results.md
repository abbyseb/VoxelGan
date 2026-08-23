# PopulationStudy — what went wrong (and what didn’t)

Leave-**patient**-out CRB MSE on SPARE MC (P1–P9). Same generators as Dan 2.0 (Encoder / Decoder / Both CRB), **no discriminator, no FiLM**. Metric = lung-masked L1 vs Elastix on 128³ (1 voxel = 1 mm). Lower is better.

**Short answer:** the runs are not broken. Population generalization is weak compared with Dan 2.0’s *same-patient* leave-phase-out. Experiment 1 shows some transfer on large-motion hold-outs. Experiment 2 mostly ties “do nothing” because P7/P9 barely move — raw L1 looks better than it is.

---

## 1. Two different claims

| | Dan 2.0 | PopulationStudy |
|--|---------|-----------------|
| Split | Leave-**phase**-out (same patient, P1) | Leave-**patient**-out |
| What is unseen | Phase pairs of a known anatomy | Entire new patients |
| Difficulty | Moderate | Much harder |

Dan 2.0 hold-out L1 was **~0.20–0.25** (best: Decoder FiLM 0.196 on 5&9). Cosine vs Elastix was **~0.95**. That model already knew P1’s lungs.

PopulationStudy trains on other patients’ DVFs only. Conditioning is still just `(t_ref, t_tgt)` — **no patient ID**.

---

## 2. Splits

**Experiment 1** (seed `20260817`): train P1, P2, P6, P7, P8, P9 (600 pairs) · test **P3, P4, P5** (300).

**Experiment 2** (seed `20260818`): train P1, P3, P4, P5 (400 pairs) · test **P7, P9** (200).

Identity pairs included. Val = 10% of *train-patient* pairs only. Hold-out patients never enter train/val.

---

## 3. Headline numbers (directed pairs, final ckpt)

| Study | Model | L1 | Zero-DVF L1 | L1 / zero | Beat zero | cos |
|-------|-------|----|-------------|-----------|-----------|-----|
| Dan 2.0 (P1 phases) | Decoder FiLM | **~0.20** | ≫ L1 | clearly &lt; 1 | ~all | **~0.95** |
| **E1** | Encoder | 0.390 | 0.547 | 0.79 | 86% | 0.712 |
| **E1** | **Decoder** | **0.370** | 0.547 | **0.75** | **94%** | **0.759** |
| **E2** | **Encoder** | **0.300** | 0.336 | **1.02** | **56%** | 0.781 |
| **E2** | Decoder | 0.321 | 0.336 | 1.06 | 54% | **0.786** |
| **E2** | Both | 0.322 | 0.336 | 1.09 | 51% | 0.762 |

“Zero” = lung-masked L1 of a **zero DVF** (identity warp). If pred ≈ zero, the model is not synthesizing motion.

### Per hold-out patient (E1 Decoder / E2 Encoder)

| Patient | Zero L1 (motion size) | Pred L1 | L1 / zero | Beat zero | cos |
|---------|----------------------|---------|-----------|-----------|-----|
| E1 P3 | 0.586 | 0.370 | 0.63 | 100% | 0.77 |
| E1 P4 | **0.684** | 0.401 | **0.59** | 100% | **0.83** |
| E1 P5 | 0.371 | 0.339 | 0.91 | 82% | 0.69 |
| E2 P7 | **0.302** | 0.304 | **1.01** | 47% | 0.77 |
| E2 P9 | 0.370 | 0.297 | 0.80 | 66% | 0.79 |

---

## 4. The issue: raw L1 is misleading on E2

E2 Encoder L1 **0.30** looks better than E1 Decoder **0.37**. That is mostly **smaller motion** on P7/P9 (zero L1 **0.34** vs **0.55** on P3–P5).

Relative to doing nothing:

- **E1 Decoder** is actually useful: ~25% better than zero, beats identity on **94%** of directed pairs. Strong on P3/P4 (large motion); weak on P5 (small motion).
- **E2 all three models sit at ~1× zero.** Encoder/Decoder/Both are coin-flips vs identity (~51–56% beat zero). Cosine ~0.76–0.79 means **direction is often right, magnitude is not** (under-predicted / too smooth).

So E2 is **not** a better population model. It is a **low-motion test set** plus weak transfer.

---

## 5. Val MSE ≠ hold-out DVF quality

On E2, **BothCRB** had the **best val MSE (0.030)** vs Decoder 0.033 / Encoder 0.037, but the **worst** hold-out L1 (0.322) and cosine (0.762).

Val is 40 pairs from the *same* four train patients. Best val can mean **overfit to train patients**, not better unseen-patient DVFs.

E1 BothCRB finished (best val MSE **0.022**) but hold-out QC was not run; E1 Encoder/Decoder final QC is in `Experiment1/*/plots/qc_holdout_final/`.

---

## 6. Is PopulationStudy “bad”?

| Interpretation | Verdict |
|----------------|---------|
| Training crashed / wrong data mixing | **No.** Curves converged; identity residual is small; matching is patient-prefixed. |
| As good as Dan 2.0 | **No.** ~2× worse L1; cosine 0.76 vs ~0.95. Unseen anatomy is the gap. |
| E1 failed | **No.** Decoder beats zero on 94% of P3–P5 pairs; P3/P4 look like real (under-magnitude) motion. |
| E2 failed | **Mostly yes as a generalization test.** Pred ≈ zero warp on P7; only a modest gain on P9. |
| Architecture ranking (E1 vs E2) | **Not stable.** E1: Decoder best. E2: Encoder slightly best. BothCRB does not win hold-out. |

**Scientific read:** phase-conditioned CRB + MSE on 4–6 SPARE patients does **not** freely transfer. It sometimes helps **large-motion** unseen cases (E1 P3/P4) and mostly **does not** help **small-motion** ones (E1 P5, E2 P7/P9). That is a real limitation of this setup (few patients, no patient embedding, Elastix-MSE only), not a bug in the Experiment folders.

---

## 7. Where the plots live

| | Path |
|--|------|
| E1 Decoder/Encoder final QC | `Experiment1/{Decoder,Encoder}CRB/plots/qc_holdout_final/` |
| E2 Encoder/Decoder/Both final QC | `Experiment2/{Encoder,Decoder,Both}CRB/plots/qc_holdout_final/` |
| Train curves | `*/plots/crb_*_mse_pop_e{1,2}.png` |
| Dan 2.0 table | `Dan2.0/results.md` |
