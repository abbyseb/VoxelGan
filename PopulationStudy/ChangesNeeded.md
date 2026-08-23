# PopulationStudy — Changes Needed for Leave-Patient-Out Generalization

Plan from Opus review of E1–E5 (agent `57d42157-1813-4435-8e63-14851251cd24`).  
Canvas mirror: `~/.cursor/projects/home-abhishek-Voxel-GAN/canvases/population-gen-plan.canvas.tsx`

---

## TL;DR

Two findings from the data change the priorities:

1. **The dominant error is amplitude, not shape.** Define α = (mean motion of the training pool) ÷ (mean motion of the hold-out patient). Across all five hold-out patients from two different splits and two different architectures, α ranks them in *exactly* the order of their beat-zero rate (rank correlation −1). The network cannot observe how much a new patient breathes, so it regresses to the training-pool mean amplitude — Bayes-optimal under MSE, and fatal whenever the test patient moves less than the training mean.
2. **The preprocessing injects a per-patient, per-axis spatial scale factor.** Every patient is cropped to its own lung bbox (with hand-chosen pads of 4–30 voxels) and resampled to 128³, so one "voxel" is 1.5–2.8 mm depending on patient *and* axis. `results.md`'s "1 voxel = 1 mm" is wrong, cross-patient L1 is not comparable, and the anisotropy rotates DVF vectors differently per patient.

A third finding reframes what "success" has meant so far: a **no-learning population atlas** (mean DVF of the other 8 patients on the shared grid) reaches cos 0.77–0.92 on P3/P4/P5, versus 0.69–0.83 for the E1 Decoder. The CNN is roughly reproducing a phase-modulated population mean field.

Recommended sequence: one **zero-training diagnostic + data regrid** (Run 0), then **two training runs** on the corrected data testing amplitude decoupling and patient-level anatomy conditioning, evaluated on *both* splits.

---

## 1. Diagnosis — ranked hypotheses

### H1 (highest confidence). Amplitude is unobservable, so the model predicts the training-pool mean

Per-patient mean ‖u‖ over all 90 directed pairs, lung-masked (current resampled-voxel units):

| Patient | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9 |
|---|---|---|---|---|---|---|---|---|---|
| mean ‖u‖ | 1.19 | 0.86 | **1.28** | **1.52** | 0.82 | 0.74 | **0.68** | 1.16 | 0.82 |

E1 train pool mean = 0.907; E2 train pool mean = 1.202. So α = train-mean ÷ patient:

| Hold-out | Split | α | Predicted behaviour | Observed L1/zero | Observed beat-zero |
|---|---|---|---|---|---|
| P4 | E1 | 0.60 | under-predict | 0.59 | **100%** |
| P3 | E1 | 0.71 | under-predict | 0.63 | **100%** |
| P5 | E1 | 1.11 | slight over-predict | 0.91 | 82% |
| P9 | E2 | 1.47 | over-predict | 0.80 | 66% |
| P7 | E2 | 1.78 | strong over-predict | 1.01 | **47%** |

Sort by α and beat-zero falls monotonically: 100, 100, 82, 66, 47. This single scalar — computable without touching the model — explains the entire E1-vs-E2 story.

The asymmetry is not a coincidence. If a prediction has cosine *c* with the truth, the L2-optimal thing to do is *shrink* it to c·‖u_gt‖. So under-predicting (α < 1) sits near-optimal and looks like success; over-predicting (α > 1) is pure added error. **E1 "worked" because its hold-out patients happen to move more than its training pool.** Swap the split and the same model loses to identity.

Supporting signal: the observed L1/zero of E1 P3/P4 (0.63, 0.59) sits essentially at the direction-limited floor implied by their cosines (√(1−cos²) ≈ 0.64, 0.57), whereas E2 P7 is at 1.01 against a floor of 0.63. That is, on E1 the scale is already near-optimal and *direction* is the binding constraint; on E2 *scale* is the binding constraint. (This is an L2 heuristic applied to L1 ratios — indicative only. Run 0 measures it directly.)

### H2 (high confidence). The 128³ resample gives every patient a different voxel size, in every axis

Parsed from `PaddedLungMasks/*.mha` (all source scans are 1 mm isotropic) plus the bbox → 128³ resample:

| Patient | mask pad (vx) | bbox extent (mm, z/y/x) | mm per resampled voxel (z/y/x) |
|---|---|---|---|
| P8 | 4 | 221 / 195 / 252 | 1.73 / 1.52 / 1.97 |
| P2 | 8 | 208 / 221 / 296 | 1.62 / 1.73 / 2.31 |
| P6 | 11 | 199 / 230 / 273 | 1.55 / 1.80 / 2.13 |
| P1 | 20 | 270 / 236 / 360 | 2.11 / 1.84 / 2.81 |
| P3 | 20 | 276 / 235 / 331 | 2.16 / 1.84 / 2.59 |
| P4 | 20 | 277 / 236 / 331 | 2.16 / 1.84 / 2.59 |
| P5 | 24 | 270 / 210 / 314 | 2.11 / 1.64 / 2.45 |
| P7 | 24 | 244 / 236 / 283 | 1.91 / 1.84 / 2.21 |
| P9 | 30 | 270 / 236 / 335 | 2.11 / 1.84 / 2.62 |

Consequences:

- All reported L1 numbers are in resampled voxels, not mm, and the conversion differs by up to 1.4× between patients. Every cross-patient comparison in `results.md` is distorted.
- The same physical 10 mm motion is 4.6 voxels in P6 and 6.5 in P4 — an *artificial* amplitude spread layered on top of the real physiological one in H1.
- Within a patient the axes are scaled unequally (e.g. P1: 2.11 / 1.84 / 2.81), so DVF vectors are sheared differently per patient. This directly caps the achievable cross-patient cosine, and is a plausible part of why cosine plateaus around 0.76–0.83.
- The pad (4–30 voxels) was a GUI choice per patient, so an operator decision is leaking into the physical scale of the labels.

### H3 (high confidence). The CNN is approximately a phase-conditioned population mean field

Leave-one-out atlas = mean DVF of the other 8 patients on the shared 128³ grid, no learning at all. For phase pair 01→06:

| Patient | atlas cos | atlas L1/zero | atlas L1/zero after oracle rescale | oracle scale |
|---|---|---|---|---|
| P1 | 0.87 | 0.63 | 0.57 | 1.44 |
| P2 | 0.48 | 1.33 | 0.94 | 0.37 |
| P3 | 0.82 | 0.71 | 0.59 | 1.85 |
| P4 | 0.92 | 0.57 | 0.46 | 1.66 |
| P5 | 0.77 | 0.73 | 0.69 | 0.85 |
| P6 | 0.54 | 1.28 | 0.90 | 0.40 |
| P7 | 0.78 | 1.93 | 0.83 | 0.38 |
| P8 | 0.86 | 0.59 | 0.60 | 1.15 |
| P9 | 0.86 | 0.75 | 0.60 | 0.70 |

The E1 Decoder gets cos 0.77 / 0.83 / 0.69 on P3 / P4 / P5; the atlas gets 0.82 / 0.92 / 0.77 on the same patients. **The 1.07M-parameter CNN does not clearly beat an average.** Two corollaries: the oracle rescale factor spans 0.38–1.85 (H1 again, from an independent direction), and P2/P6 are genuine motion-pattern outliers (atlas cos 0.48/0.54) sitting in E1's *training* pool, diluting it.

### H4 (high confidence, cheap to fix). Model selection uses a validation set that measures the wrong thing

`train_mse.py` checkpoints on best val MSE over held-out *pairs of training patients*. E2 BothCRB had the best val (0.030) and the worst hold-out (L1 0.322, cos 0.762). E1 BothCRB reached train 0.0105 / val 0.0228 — a 2× gap, i.e. it is fitting the six training identities. E3 computes a genuine cross-patient query loss every epoch and then still selects on `val_pairs`, discarding its own best signal.

### H5 (medium-high). The anatomy pretrain, as wired, cannot carry patient identity

- For Encoder/Both, `FrozenAnatomyVec` global-average-pools the encoder bottleneck of the **random 64³ training crop**. The resulting "anatomy code" describes the patch, not the patient, and changes every sample. At QC it is computed on the full 128³ volume — a different distribution entirely.
- For Decoder-CRB the encoder is frozen LIDC weights and clearly underfits: E5 reached train 0.038 / val 0.052 versus E1's 0.011 / 0.022, i.e. LIDC reconstruction features are a poor DVF basis.
- The LIDC AE trained on 64³ crops to a recon MSE of ~7e-5, which is trivially solvable by near-identity filters and teaches little anatomy.

This explains E4's modest gain (which the α table attributes mostly to a mild amplitude shift) and E5's outright regression.

### H6 (medium). 64³ patch training / 128³ evaluation, with no position information

Training crops are lung-biased random 64³ windows with no positional encoding. Motion magnitude in the lung is strongly position-dependent (diaphragm ≫ apex), so two visually similar patches carry very different motion, and the network is asked to fit both. Evaluation then runs at 128³ where the CRB's global modulation sees different statistics.

### H7 (medium, structural). n = 9, of which 2 are outliers

Six training patients, two of which (P2, P6) have atlas cosines of 0.48/0.54. There is very little population variability to learn from, and single-split conclusions are unstable — the architecture ranking already flipped between E1 (Decoder) and E2 (Encoder), with all three within ~0.02 L1.

### H8 (minor). 10% of training samples teach "predict zero"

Identity pairs (10 per patient, ~54 of 540 train pairs) have an all-zero target, adding a small extra pull toward shrinkage on top of H1.

---

## 2. What NOT to try next

| Dead end | Why |
|---|---|
| More epochs, larger models, more CRB placements | Encoder/Decoder/Both span ~0.02 L1 and the ranking flips between splits — that is split noise, not signal. Training loss is already at 0.011 with val at 0.022; the problem is not capacity or convergence. |
| More LIDC patients / a better LIDC autoencoder, as currently wired | The failure is the interface (patch-level GAP code, 64³ vs 128³ mismatch, frozen encoder underfitting), not the pretraining corpus size. Fix the wiring before spending more on data. |
| Porting the discriminator or FiLM from Dan 2.0 | Dan 2.0 solved same-patient leave-phase-out where anatomy was known. An adversarial term makes fields look plausible; it cannot supply the patient amplitude that H1 says is missing. Expect prettier QC panels and no change in L1/zero. |
| More episodic / MAML variants before fixing selection and scale | E3 already showed the effect is ~0.01 L1. Episodic training cannot help a model that has no input carrying the quantity it needs to adapt. Revisit only after H1/H2 are fixed. |
| Reporting raw L1 across splits, or treating E2 as "better" | E2's low L1 is a low-motion test set. Only L1/zero, beat-zero and cosine are comparable, and only after H2 is fixed. |
| Any further use of the E2 split as a headline result | Its hold-out patients are the two lowest-motion in the cohort, with α = 1.78 and 1.47 — the hardest possible configuration for a mean-amplitude predictor, and unrepresentative. Keep it as a stress test, not a headline. |
| Tuning against hold-out QC | With 2–3 test patients and 0.02-level differences, a few such choices will manufacture a result that will not replicate. |

---

## 3. Prioritized interventions

### P0 — do before any further training

#### P0-A. Regrid to a common isotropic mm space

*Mechanism.* Resample every patient onto one fixed grid — e.g. 2.0 mm isotropic, 160³ FOV (320 mm) centred on the lung-mask centroid, air-padded — and convert DVF components to millimetres. Physical units become identical across patients and axes.

*Why it should help.* Removes the artificial 1.4× cross-patient amplitude spread and the per-patient vector shear of H2. It makes the population atlas and any learned population prior physically meaningful, and makes L1 genuinely mm. It should raise the cross-patient cosine ceiling on its own.

*Implementation note.* No re-registration is needed for a first pass: multiply each DVF component by that patient's mm-per-voxel for the corresponding axis (careful — the stored `.npy` last axis is ordered x, y, z per `warp.py`'s channel convention, not z, y, x), then spatially resample the field onto the common grid. Re-running Elastix on the common-grid volumes is the cleaner long-term version; do it as a later validation.

*Expected metric.* Not primarily a metric win — an interpretability and ceiling fix. Expect cosine +0.02 to +0.05 and, more importantly, cross-patient numbers that mean something. Risk: interpolation smoothing of the DVF; mitigate by checking that `warp(ref, u_regridded)` still matches the target CT.

*Effort.* ~1 day, CPU only, no retraining.

#### P0-B. Zero-training diagnostics on existing checkpoints

Four measurements, all pure post-processing on the E1–E5 predictions already on disk:

1. **Oracle-scale ablation.** For each hold-out patient, find the single scalar α* minimising ‖α·pred − gt‖ and report the rescaled L1/zero. This is the decisive test of H1.
2. **Population atlas baseline.** Full leave-one-out atlas across all 90 phase pairs and all 9 patients, unscaled and oracle-scaled. This is the "what does the network add" control that is currently missing.
3. **Amplitude ratio** r = ‖pred‖ / ‖gt‖ per patient, reported alongside every future result.
4. **GT-warp sanity floor.** Compute `|target − warp(ref, gt_dvf)|` on the QC panels. If the ground-truth DVF does not warp ref onto target near-perfectly, the direction/convention of the stored fields is wrong and caps every cosine in the study. Cheap, and worth ruling out.

*Expected outcome.* If oracle rescaling drops E2 P7 from L1/zero ≈ 1.01 to ≤ 0.75, H1 is confirmed and P1-A/B/C become the whole programme. If it barely moves, the problem is shape, and priority shifts to P1-D and P2-B.

*Effort.* ~half a day, no GPU training.

#### P0-C. Fix model selection and the evaluation protocol

*Mechanism.* Replace the same-patient validation with **leave-one-training-patient-out validation**: hold one training patient entirely out of the gradient, use its L1/zero for checkpointing. E3 already computes exactly this quantity and throws it away. Simultaneously, replace the ad-hoc E1/E2 splits with **3-fold leave-3-patients-out covering all nine** — {P3,P4,P5}, {P1,P2,P6}, {P7,P8,P9} — so every patient is tested once and no conclusion depends on one lucky split.

*Why it should help.* H4 shows val MSE is anti-correlated with hold-out quality; selecting on the wrong criterion can cost more than any architecture change. The 3-fold protocol is what makes the ≤0.02 L1 differences between architectures interpretable at all.

*Expected metric.* Selection fix alone: perhaps 0.01–0.03 L1 on hold-out, and — more valuable — error bars.

*Effort.* Small code change; 3× the runs for a full fold sweep (~12–15 GPU-hours at current speed, ~4 h with batching + AMP).

---

### P1 — the substantive interventions

#### P1-A. Decouple shape from amplitude (the primary fix for H1)

*Mechanism.* Split the prediction into a normalised shape field and an explicit scalar (or per-axis 3-vector) amplitude:

- **Train on normalised targets.** Divide each patient's DVF by a per-patient amplitude constant A_p (e.g. the 95th percentile of ‖u‖ over that patient's max-excursion pair). The network then only has to learn *shape*, which the atlas analysis says does transfer (cos 0.77–0.92 for the well-behaved patients).
- **Condition on a measured amplitude at inference.** Feed A_p (and the per-pair amplitude fraction) into the CRB conditioning vector alongside (t_ref, t_tgt). Widen `cond_dim` from 2 to 3–5.
- **Choose the amplitude source to match the deployment story.** If the whole 4D CT is available at inference (the motion-model-building use case), derive A_p from the images alone with no registration: per-phase lung volume V(t) by HU thresholding, or diaphragm dome SI position d(t). If only a planning CT is available, use the clinical breathing surrogate (RPM/bellows amplitude), which is exactly what such a scalar represents physically.

*Why it should help.* It converts the single largest error source from an unobservable latent into an input. The per-patient normalisation also multiplies the effective data available for shape learning, which matters at n = 6 training patients.

*Expected metric.* On the E2 split this should move P7 from L1/zero ≈ 1.01, beat 47% to roughly **0.70–0.80, beat 75–85%**, with cosine roughly unchanged (0.77–0.80) since shape is not the target of this change. On the E1 split expect a smaller gain (α < 1 there, so the current model is already near its optimal shrinkage) — perhaps 0.68 → 0.60–0.65 with beat-zero staying near 94%. **Always report both splits**; a change that improves E2 and degrades E1 is a scale recalibration masquerading as progress.

*Risk.* Moderate. If the image-derived amplitude surrogate correlates poorly with the DVF amplitude, the conditioning is noise. Mitigate by reporting an **oracle-amplitude** variant (A_p from the true DVF) as an upper bound alongside the measured-surrogate result — the gap between them is exactly the surrogate's quality, and it's a publishable ablation either way.

*Effort.* ~2–3 days including the surrogate extraction; one training run per split.

*Minimal design.* Decoder-CRB, regridded data, 3-fold LOPO, four arms: (i) baseline, (ii) normalised target + oracle amplitude, (iii) normalised target + measured amplitude, (iv) normalised target + training-pool-mean amplitude (isolates how much of the gain is conditioning versus normalisation).

#### P1-B. Amplitude augmentation

*Mechanism.* Synthesise new training triplets by scaling: for a pair (ref, target, u) and a factor s ∈ [0.4, 2.0], form (ref, warp(ref, s·u), s·u). This creates a continuum of breathing amplitudes for every anatomy at essentially zero data cost. Add the standard DVF-aware geometric augmentations too — L-R flip with negation of the x-component, small rotations with the corresponding rotation applied to the vectors.

*Why it should help.* Directly breaks the correlation between patient identity and amplitude in the training set, so the network *must* read the amplitude conditioning of P1-A rather than memorising a per-patient constant. Without this, P1-A's conditioning input is likely to be ignored — with 6 training patients the network can memorise 6 amplitudes from anatomy alone.

*Expected metric.* On its own, modest. In combination with P1-A, this is what makes the conditioning generalise; treat the two as one intervention.

*Risk.* Low, but pseudo-targets are interpolated images, so the image statistics drift slightly from real CT. Cap s and keep a fraction of real pairs in every batch.

*Effort.* ~1 day.

#### P1-C. Test-time amplitude adaptation (the cheapest possible version of P1-A)

*Mechanism.* At inference on an unseen patient, fit **one scalar** — a global gain on the predicted field — by minimising an image-similarity loss (NCC or masked MSE) between `warp(ref, α·pred)` and the target-phase CT. No ground-truth DVF, one parameter, seconds per patient. `utilities/warp.py` is already `grid_sample`-based and differentiable, so nothing new is needed.

*Why it should help.* It is a direct, label-free implementation of the oracle-scale ablation from P0-B. If P0-B shows oracle rescaling recovers most of the E2 loss, this recovers most of it *legitimately*.

*Expected metric.* Should reach within ~0.05 L1/zero of the oracle-scale number from P0-B.

*Risk.* Low, and the downside is bounded: it can be reported as an optional post-processing step. Note honestly that it consumes the target-phase image, so it belongs to the motion-model-building scenario, not single-CT prediction.

*Effort.* ~1 day. **Highest value per hour in the whole plan** — no retraining at all.

#### P1-D. Patient-level anatomy conditioning, done properly (fixes H5, H6)

*Mechanism.* Three coupled changes:

- Compute the anatomy code **once per patient from the full reference volume**, not per training crop, and pass it as a constant condition. This alone fixes the E4/E5 side-branch, where the code currently describes a random 64³ patch and differs between training and QC.
- Add **normalised coordinate channels** (z, y, x in lung-bbox coordinates) as extra input channels, so a patch knows whether it is at the diaphragm or the apex. This resolves the position ambiguity of H6 at near-zero cost.
- Either train at **full 128³/160³** (the model is only 1.07M parameters; with AMP and gradient checkpointing this is feasible at batch 1–2 and eliminates the train/test mismatch entirely), or keep patches and rely on the coordinate channels.

*Why it should help.* Gives the network a stable per-patient signal and removes an ambiguity that currently forces averaging over lung positions. Note this addresses *shape*, so judge it on cosine, not L1/zero.

*Expected metric.* Cosine 0.76 → 0.82–0.86 (the atlas ceiling for well-behaved patients), and L1/zero improvement mostly on P5-type failures (low motion, low cosine).

*Risk.* Moderate. The atlas result caps how much patient-specific anatomy signal is extractable from n = 6; the coordinate channels and full-volume training are the safer half of this intervention.

*Effort.* ~3–4 days including the memory work for full-volume training.

---

### P2 — larger bets, after P0/P1 report

#### P2-A. Decide the task framing, and add the honest strong baseline

The generator currently sees only the reference CT and two phase indices; `target_ct` is loaded by the dataset and never used. If the target CT *is* available at inference, the correct strong baseline is an **unsupervised population registration network** (VoxelMorph-style, NCC + diffusion regularisation, trained across patients). Such nets generalise across patients far better than DVF regression precisely because the amplitude is visible in the input. It would likely reach L1/zero ≈ 0.3–0.5 on both splits and would reframe the contribution: DVF regression is a *motion-model synthesis* method, and the registration net is its upper bound. If the target CT is *not* available, this baseline still belongs in the paper as the oracle. Either way, the ambiguity should be resolved explicitly before more architecture work.

#### P2-B. More patients

n = 9 with 2 outliers is the hard limit on H7. The 4D-Lung (TCIA) and DIR-Lab COPDgene/4DCT cohorts provide 4D CT with phases, and DIR-Lab adds expert landmarks — which would let you report **TRE in mm**, the metric the registration community actually accepts, instead of DVF L1 against another algorithm's output. Higher effort, highest scientific return.

#### P2-C. Loss composition

Add a direction term (cosine), a Jacobian/gradient smoothness penalty, and an image NCC term on `warp(ref, pred)`. The NCC term is the one that matters most: it is patient-specific and label-free, so it is also the enabler for the test-time adaptation in P1-C and for semi-supervised training on unlabelled patients.

#### P2-D. Low-rank motion model

Classical 4D CT motion models represent per-patient motion as a small PCA basis with per-patient coefficients. Predicting *coefficients* over a population basis, rather than a dense field, builds the amplitude/shape decomposition of P1-A into the representation and is well matched to n ≈ 10 patients. Worth considering if P1-A works and you want a principled version.

---

## 4. Recommended next runs — the minimal sequence

### Run 0 — Diagnostics + regrid (no training, ~1.5 days)

P0-A + P0-B + P0-C together. Deliverables:

- A common-grid, mm-unit dataset, and a corrected units statement replacing "1 voxel = 1 mm" in `results.md`.
- A table for every existing checkpoint: L1/zero, oracle-scale L1/zero, amplitude ratio r, beat-zero, cosine — per patient.
- The LOO population atlas baseline across all 9 patients and all 90 pairs, unscaled and oracle-scaled.
- The GT-warp sanity floor.

**This run alone may change the paper's conclusions** and costs no GPU time. It also settles whether to proceed with P1-A/B/C or pivot to P1-D/P2-B.

### Run 1 — E6: amplitude-decoupled Decoder-CRB (~1–2 days GPU)

P1-A + P1-B on the regridded data, Decoder-CRB only, with LOPO checkpoint selection. Arms: baseline / normalised+oracle-amplitude / normalised+measured-amplitude / normalised+train-mean-amplitude. Evaluate on **both** the E1 and E2 hold-outs. This is the direct test of H1 and the four arms separate normalisation, conditioning, and surrogate quality.

### Run 2 — one of two, chosen by Run 1's result

- **If Run 1 confirms H1** (E2 P7 beat-zero > 70%): **E7 = P1-D**, patient-level anatomy code + coordinate channels + full-volume training, on top of Run 1's best configuration, over the 3-fold LOPO protocol. Target: cosine ≥ 0.82 on both splits.
- **If Run 1 does not confirm H1** (oracle amplitude also fails to fix E2): the failure is shape, and Run 2 becomes the **P2-A registration baseline** — establishing the achievable ceiling on this data before investing further in DVF regression.

Add **P1-C (test-time scalar adaptation)** opportunistically at any point; it needs no training run and can be evaluated on the checkpoints that already exist.

---

## 5. Success criteria

### Metrics to report, always

Primary: **L1/zero** per hold-out patient, in mm on the common grid. Secondary: **beat-zero %**, **cosine**, **amplitude ratio r = ‖pred‖/‖gt‖**, and **oracle-scale L1/zero** (which separates shape error from scale error). Report per patient, never only pooled — pooled means hide the P7-vs-P4 structure that constitutes the entire finding.

### Baselines an intervention must beat

1. Zero DVF (identity).
2. **LOO population atlas, unscaled** — the "did learning add anything" bar.
3. LOO population atlas with oracle per-patient scale — the bar for anything claiming to solve amplitude.

### Declare an intervention worked if

| Criterion | Threshold |
|---|---|
| Beats the unscaled atlas | on **both** splits, all hold-out patients |
| L1/zero | ≤ 0.65 on the E1 hold-out **and** ≤ 0.80 on the E2 hold-out |
| Beat-zero | ≥ 80% on every hold-out patient, including P7 |
| Amplitude ratio r | within 0.85–1.15 for every hold-out patient (the direct H1 check) |
| Cosine | ≥ 0.80 mean, for shape-targeted interventions (P1-D) |
| Stability | holds across the 3-fold LOPO protocol, not one split |

### Abandon an intervention if

- It improves one split while degrading the other by a comparable amount — that is amplitude recalibration, not generalisation, and the α table predicts it exactly.
- It fails to beat the unscaled LOO atlas after a full run.
- The gain is < 0.03 L1/zero, which is within the spread already observed between Encoder/Decoder/Both on a single split.
- P1-A specifically: abandon if the **oracle-amplitude** arm fails. If knowing the true amplitude does not fix E2, then H1 is wrong and no surrogate will rescue it — pivot immediately to P1-D / P2-A rather than tuning the surrogate.

### The honest negative result, if it comes to that

If, after P0 and P1, the network still does not beat the LOO atlas, that is a publishable finding in itself: *phase-conditioned DVF regression on ~10 patients learns a population mean motion field and nothing patient-specific; per-patient amplitude must be supplied as an input, and dense DVF regression without target-phase imaging is not identifiable at this cohort size.* The atlas baseline and the α table are what make that claim rigorous rather than anecdotal — which is why Run 0 is first.

---

## Headline numbers so far (directed pairs, final ckpt)

| Study | Model | L1 | Zero L1 | L1/zero | Beat zero | cos |
|-------|-------|----|---------|---------|-----------|-----|
| **E1** | **Decoder** | **0.370** | 0.547 | **0.68** | **94%** | **0.759** |
| E1 | Encoder | 0.390 | 0.547 | 0.71 | 86% | 0.712 |
| E2 | Encoder | 0.300 | 0.336 | 0.89 | 56% | 0.781 |
| E2 | Decoder | 0.321 | 0.336 | 0.95 | 54% | 0.786 |
| E2 | Both | 0.322 | 0.336 | 0.96 | 51% | 0.762 |
| E3 | Decoder | 0.309 | 0.336 | 0.92 | 54% | 0.788 |
| E3 | Encoder | 0.337 | 0.336 | 1.00 | 48% | 0.759 |
| **E4** | **Decoder** | **0.291** | 0.336 | **0.87** | **59%** | 0.766 |
| E4 | Both | 0.320 | 0.336 | 0.95 | 55% | 0.772 |
| E4 | Encoder | 0.332 | 0.336 | 0.99 | 51% | 0.773 |
| E5 | Both | 0.377 | 0.547 | 0.69 | 89% | 0.739 |
| E5 | Encoder | 0.386 | 0.547 | 0.71 | 89% | 0.728 |
| E5 | Decoder | 0.398 | 0.547 | 0.73 | 90% | 0.704 |

E1/E5 hold-out: P3,P4,P5. E2/E3/E4 hold-out: P7,P9.
