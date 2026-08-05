# Voxel_GAN Diary

Living lab notebook. Newest entries at the **bottom**.  
Timestamps are local time (AEST / UTC+10) unless noted.

---

## How to use

- After each experiment or design decision: add a dated entry — **what we did**, **what happened**, **why**, **what next**.
- Point to artifacts (`results/…`, `plots/…`, `weights/…`) instead of pasting huge tables.
- Keep it honest when something failed; failed runs are the reason for the next change.

---

## 2026-08-04 — Project day 1 (backfilled)

### ~09:00 — Framing

**What:** Locked scope for the capstone: **patient-specific** phase-conditioned DVF synthesis (single ref CT + φ_ref, φ_tgt → diffeomorphic DVF), supervised by Elastix, light 3D PatchGAN regularizer. Eval = **leave-phase-out**, not inter-patient.

**Why:** SPARE MC P1 gives a full 10-phase 4DCT; claim is intra-patient generalization over unseen phase pairs.

**Artifacts:** `plan.md`, `Code_Practices.md`

---

### ~09:20–10:30 — Code skeleton

**What:** Built layout per house style: `utilities/{generator,discriminator,film,svf,losses,dataset}.py`, single `train_dvf_gan.py`.

**Choices locked early:**
- Generator: U-Net + FiLM at bottleneck + every decoder scale after skip fusion → SVF → scaling-and-squaring (`int_steps=6`)
- Discriminator: 4-ch `[ref CT, DVF]`, PatchGAN → 8³ logits, spectral norm
- Losses: lung-masked DVF L1 + lung-masked NCC + smoothness + LSGAN (real label 0.9)
- Train knobs: `λ_img=0.5`, `λ_smooth=0.1`, `λ_adv=0.05`, `d_update_freq=2`, `adv_warmup_epochs=5`

---

### ~10:30–12:00 — Data prep (Elastix)

**What:** Lung-bbox crop → **128³**; **masked B-spline Elastix** (not multi-B-spline sliding). 90 directed pairs + 10 identity. Leave-out phases **5 & 9** → train 64 / val 36 pairs. `Mask_Lung.npy` in train/val.

**QC:** ~0% negative Jacobian on pairs. Ribs poorly matched on large jumps (expected with lung mask) → lung-masked losses required.

**Artifacts:** `scripts/prepare_elastix_pairs.py`, `data/spare/{train,val}/`, `configs/elastix_bspline_masked.txt`

---

### ~11:00–11:15 — Sparse train (1 crop/pair) + discrete phase Embedding

**What:** First full 100-epoch run (`plots/train_full.log`, filename `spare_mc_p1_dvf_gan`).

**Result:** Train and val both moved; best val G ≈ **0.46**. Usable smoke of the stack, but under-sampled spatially.

**Why sparse:** Fast iteration while proving the loop.

---

### ~12:30–13:05 — Dense patches + discrete Embedding

**What:** `patches_per_pair_train=16`, val=8 (`plots/train_dense.log`).

**Result:** Train loss dropped; **val stuck ~0.93**. Classic overfit / broken generalization.

**Why we faced it:** Discrete `nn.Embedding` for phases. Held-out phases (5, 9) never appear in training, so their embedding rows stay random → leave-phase-out pairs are effectively out-of-distribution for FiLM. Dense crops made the train-side fit look better while exposing the conditioning bug.

**Decision:** Replace discrete embeddings with a **continuous phase MLP** (`φ / n_phases` → small MLP → 8-D codes). Unseen phases still land on a smooth manifold.

---

### ~13:10–14:12 — Dense + phase-MLP FiLM (baseline)

**What:** Retrain with continuous FiLM (`filename=spare_mc_p1_dvf_gan_phase_mlp`). Stopped ~epoch **88/100**.

**Result:**
| Metric | Value |
|--------|--------|
| Best val G | **~0.177** @ epoch 85 |
| Late D loss | ~**0** (adversary collapsed / saturated) |
| TEST leave-out | L1 ~0.18–0.24 vs zero-field ~1.0–1.4; cosine ~0.97–0.99 |
| Identity 01→01 | L1 ~0.027, mean\|pred\| ~0.056 |

**Verdict:** Working **supervised** baseline. GAN term is mostly decorative — D dies after warmup.

**Artifacts:** `results/baseline_phase_mlp/` (weights, curve, `qc_test/`)

---

### ~14:29 — Freeze baseline

**What:** Copied weights / plots / TEST QC into `results/baseline_phase_mlp/` so later scenarios cannot overwrite the story.

---

### ~14:30–16:15 — Three “make GAN stronger” scenarios

Shared loop: `utilities/train_loop.py`. Thin scripts:
- `train_scenario_weak_d.py` — `d_base_channels=16` (half-width D)
- `train_scenario_more_g.py` — `d_update_freq=5` (update D less often)
- `train_scenario_higher_adv.py` — `λ_adv=0.15`

All: phase-MLP FiLM, dense patches, 100 epochs. Archived under `results/scenario_*`.

**Results:**

| Run | Best val G | Late D loss | Read |
|-----|------------|-------------|------|
| baseline phase-MLP | **0.177** @85 | ~0.0006 | strong supervised; dead D |
| weak D | **0.173** @100 | ~0.0002 | val similar; D still dead |
| more G steps | 0.183 @99 | ~0.006 | slightly healthier D |
| higher λ_adv | 0.222 @99 | **~0.019** | only living adversarial signal; supervised val worse |

**Why we faced “dead D”:**
1. Strong lung-masked L1 pulls fakes toward Elastix fast.
2. 4-ch D `[ref, DVF]` can win on easy residual cues (or saturate) without teaching motion plausibility vs anatomy.
3. Small `λ_adv=0.05` barely forces G to fight back → game ends; D loss → 0.
4. Weakening D capacity alone does **not** fix this (weak_d still collapsed).

**What higher_adv taught us:** Raising `λ_adv` keeps D awake, but val G (no adv) gets worse — expected tradeoff until the *discrimination task* itself is more meaningful.

---

### ~16:23 — Diagnosis: how to activate D (decision)

**What we decided next (not trained yet):**

1. **Harder D input (highest leverage):** move to **5-ch** `[ref CT, target CT, DVF]` so D judges consistency with *both* volumes (as preferred in `plan.md`), not just DVF texture vs Elastix.
2. Keep / schedule a non-trivial `λ_adv` (ramp after warmup), since that was the only knob that kept D alive.
3. Optional later: R1 on reals, feature matching from mid-D layers.

**Why not “bigger D” or another weak-D try:** Capacity knobs alone did not revive the game; the task for D is too easy / poorly aligned with the supervised objective.

---

### ~16:24 — Diary started

**What:** Created this file to record narrative + results with timestamps from now on.

**Rule going forward:** every train / QC / architecture change gets an entry here before we move on.

---

## 2026-08-05 ~09:28 — Results check (no new train)

Re-read archived logs/curves. Nothing new trained overnight. Status unchanged:

| Run | Best val G | Late D | Notes |
|-----|------------|--------|-------|
| baseline phase-MLP | **0.177** @85 (stopped 88) | ~0 | best supervised; only run with TEST QC |
| weak_d | **0.173** @100 | ~0 | best val number; D still dead |
| more_g | 0.183 @99 | ~0.006 | tiny D pulse |
| higher_adv | 0.222 @99 | **~0.018** | only healthy-ish D; worse supervised val |

**Gap:** scenarios have weights + curves only — **no leave-phase-out TEST QC** yet (baseline has `results/baseline_phase_mlp/qc_test/`).

---

## 2026-08-05 ~09:44–09:47 — External diagnosis (shortcut cue)

Pasted the dead-D problem prompt elsewhere. Takeaways we accept:

**Cause (not “D too strong”):** D wins on a **cheap shortcut** — Elastix B-spline DVFs vs SVF-integrated generator DVFs look different in frequency/texture. 4-ch `[ref, DVF]` never has to judge motion plausibility. That explains why weak-D / more-G barely helped, and why `λ_adv=0.15` “revives” D while **hurting** supervised val (~25%): G is chasing the wrong criterion.

**#1 fix to try:** feed D **warped image space** — `[warp(ref, DVF), target_CT]` (real = Elastix warp, fake = G warp). Removes the raw-DVF spectral cheat. Prefer this **before** 5-ch `[ref, target, DVF]`.

**Also ranked later:** matched blur on DVF-to-D, feature matching, instance noise, TTUR (`lr_d`↓), RaLSGAN.  
**Stop spending on:** more D capacity / `d_update_freq` / higher `λ_adv` alone.

**Base model for the first warp-D run:** **phase-MLP baseline recipe** (same as `train_dvf_gan.py` / `spare_mc_p1_dvf_gan_phase_mlp`) — continuous FiLM, dense patches, `λ_adv=0.05`, D width 32, freq=2.  
**Not** higher_adv weights (would confound). **Not** overwrite baseline — new scenario file + new `filename`.

---

## 2026-08-05 ~09:50 — Start warp-D scenario

**Q:** Will warp-vs-target catch subtle changes?  
**A:** Better chance than raw DVF. PatchGAN on `[warped, target]` sees local CT mismatch (edges, vessels, fissures) — the kind of subtle anatomy error global L1/NCC under-weight. It will *not* magically invent lobe-sliding physics; if Elastix “real” warps are blurry/wrong at ribs, D learns that too. Watch late D loss staying >0 and TEST Jacobian / boundary diffs vs baseline.

**What:** Implemented `d_input_mode='warp'` in `utilities/train_loop.py` (2-ch D), `train_scenario_warp_d.py`, generalized `GANLoss` pair API. Launching 100-epoch train from scratch on phase-MLP knobs. Artifacts: `plots/train_scenario_warp_d.log`, `weights/spare_mc_p1_scenario_warp_d_*.pth`.

---

## 2026-08-05 ~10:25 — Warp-D mid-run check (epoch 50/100)

**Not the same issue.** D stays alive: post-warmup mean D ≈ **0.074**, late ≈ **0.06** (baseline was ~0; higher_adv was ~0.018). Best val so far **0.235** @49 (baseline was 0.177 — still halfway; train/val tracking). Shortcut fix looks real; finish run then TEST QC.

---

## 2026-08-05 ~11:03 — Warp-D finished (100/100)

| Run | Best val G | Late D | Verdict |
|-----|------------|--------|---------|
| baseline phase-MLP | **0.177** | ~0 | dead D; best supervised |
| higher_adv | 0.222 | ~0.018 | living D, worse val |
| **warp-D** | **0.202** @96 | **~0.057** | **living D**; val between baseline and higher_adv |

Archived: `results/scenario_warp_d/`. Next useful step: leave-phase-out TEST QC vs baseline (L1/cosine/identity + optional Jacobian).

---

## 2026-08-05 ~11:08 — Leave-phase-out TEST QC (baseline vs warp-D)

Script: `scripts/qc_leave_phase_out.py`. Plots: `plots/qc_test_phase_mlp/`, `plots/qc_test_warp_d/`. Log: `plots/qc_baseline_vs_warp_d.txt`.

| Pair | Baseline L1 | Warp-D L1 | Δ | cos B / W |
|------|-------------|-----------|---|-----------|
| 01→05 | 0.241 | **0.190** | −0.051 | 0.987 / **0.992** |
| 02→05 | 0.225 | **0.172** | −0.053 | 0.986 / **0.991** |
| 05→01 | **0.213** | 0.243 | +0.029 | 0.991 / 0.990 |
| 05→09 | 0.214 | **0.200** | −0.015 | 0.984 / **0.987** |
| 09→05 | **0.187** | 0.203 | +0.015 | 0.986 / 0.988 |
| 09→06 | **0.180** | 0.194 | +0.014 | 0.990 / 0.989 |
| 01→01 | 0.027 | **0.025** | −0.002 | — |

Mean leave-out L1: baseline **0.210** → warp-D **0.200** (slightly better). Cosine high both (~0.98–0.99). Neg Jacobian ≈ **0%** both. Identity near zero both.

**Read:** Warp-D kept a living adversary **and** matched/beat baseline on TEST for several held-out pairs (esp. →05). Val G during train was higher than baseline, but leave-out L1 is not worse overall — adv may be helping generalization even when combined val loss looks softer.

---

## Open threads

- [x] Implement **warp-D** scenario on phase-MLP baseline knobs (`train_scenario_warp_d.py`)
- [x] Finish warp-D train + compare D health / val G vs baseline
- [x] Leave-phase-out **TEST QC** warp-D vs baseline
- [ ] Optional: finer PatchGAN / feature matching / R1 if boundaries still soft in panels
- [ ] Defer 5-ch `[ref,tgt,DVF]` unless warp-D QC plateaus
- [ ] Report framing: patient-specific leave-phase-out; GAN as regularizer, not the main loss

---

## Artifact map (quick)

| Name | Path |
|------|------|
| Baseline weights + TEST QC | `results/baseline_phase_mlp/` |
| Scenario runs | `results/scenario_{weak_d,more_g,higher_adv}/` |
| Live curves / logs | `plots/spare_mc_p1_*.png`, `plots/train_*.log` |
| Train entry (baseline script) | `train_dvf_gan.py` |
| Shared train API | `utilities/train_loop.py` |
| Scenario launchers | `train_scenario_*.py` |
