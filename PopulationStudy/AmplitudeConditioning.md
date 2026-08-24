# Amplitude conditioning — design for continuous "how much they breathe"

Design document for the amplitude input to leave-patient-out CRB DVF synthesis (E6 / Run 1 of
[`ChangesNeeded.md`](ChangesNeeded.md)). No code. Every number below §0 is measured on the
current SPARE P1–P9 library; the probes are described in §12 so they can be re-run as scripts.

Inputs: [`ChangesNeeded.md`](ChangesNeeded.md) (H1–H8, P0–P2),
[`Run0_Diagnostics/Run0.md`](Run0_Diagnostics/Run0.md) (oracle-scale, atlas),
[`DVFCharacteristics/DVFCharacteristics.md`](DVFCharacteristics/DVFCharacteristics.md) (rankings).

---

## 1. North star

At inference the model sees **one static planning CT of a new patient, a target phase
`t_tgt` anywhere on the breathing cycle, and one externally supplied number describing how
deeply that patient breathes** — nothing else from that patient's imaging. It outputs the
full DVF for `t_ref → t_tgt`, and by sweeping `t_tgt` a continuous 4D motion loop with
per-phase dwell weights suitable for 4D dose accumulation. The shape of the motion comes
from the population (the atlas result says shape transfers: cos 0.77–0.92 on well-behaved
patients); the *scale* comes from the external number, because Run 0 shows scale is what the
network cannot infer (E2 Encoder P7: L1/zero 1.09 → 0.69 under oracle rescale). The intended
deployment surrogate is the breathing signal already recorded during an RT planning scan
(RPM marker block, bellows, spirometry). 4D CT is allowed for *training*; it is forbidden at
inference on a new patient, or the argument is circular.

---

## 0. Measurements that constrain the design

Numbered 0 because it precedes every design choice below; §2 onward follow the requested
running order. All on `data/P*/all` (128³ lung-bbox crops), lung-masked, DVF components
converted to mm with the per-patient per-axis mm/voxel from `ChangesNeeded.md` H2. These
numbers are provisional until P0-A (isotropic regrid) and should be recomputed on the common
grid; the *ratios* this design actually consumes are insensitive to that (§0.3).

### 0.1 Conventions are correct, but the motion is AP-dominant

`raw/P*/*.mha` headers: `AnatomicalOrientation = RAI`, 1 mm isotropic,
`DimSize = 450 220 450` → numpy array axes are **(0: SI, 1: AP, 2: LR)**. DVF channels are
`(dx, dy, dz) = (LR, AP, SI)` as documented.

Verified, not assumed: warping `CT_ref` by all 6 channel permutations × 8 sign combinations
and scoring the lung-masked image residual against `CT_tgt`, the documented mapping
(axis0←ch2, axis1←ch1, axis2←ch0, all positive) is the **global optimum** on P3, P4, P7, P8.
No permutation bug. Residual/identity ratio 0.44–0.53 on the extreme pairs, consistent with
Run 0's 0.53–0.83 — the fields are real but explain only about half the image residual, and
amplitude labels inherit that noise floor.

However, mean |u| per channel on each patient's extreme pair is **AP-dominant**, e.g. P4
(LR 0.43, AP 2.97, SI 0.99) — AP exceeds SI by 2–3× in 8 of 9 patients. Registration-free
confirmation: the lung-air centroid excursion is largest in AP, and the caudal lung boundary
(diaphragm) moves only 0–4.2 mm. In P1, P3, P8 the *apical* AP displacement exceeds the
basal one. This cohort is therefore **not diaphragm-piston-dominated**.

Three consequences, all load-bearing:

1. Do not hard-wire a textbook SI/diaphragm surrogate. Choose the surrogate by measured
   calibration (§5), not by physiology.
2. Start with the scalar ‖u‖, which is invariant to channel permutation and sign, and becomes
   invariant to the per-axis mm scaling too once P0-A makes the grid isotropic. A 3-axis
   amplitude is only meaningful after G0 passes and would be dominated by an axis whose
   behaviour is atypical of real patients — it is a poor generalisation target here.
3. External validity is a genuine risk: a real cohort would be SI-dominant. The design must
   be defined in terms the transfer survives (a scalar, and a surrogate selected by
   calibration), which is what §2–§3 do.

### 0.2 Oracle amplitude per patient (mm)

`A_p` = percentile of ‖u‖ over the lung mask on the patient's volume-extreme pair.

| Patient | P4 | P1 | P3 | P9 | P8 | P7 | P5 | P2 | P6 |
|---|---|---|---|---|---|---|---|---|---|
| **q90 (mm)** | 14.69 | 10.47 | 11.30 | 7.50 | 7.68 | 6.38 | 6.17 | 5.33 | 4.73 |
| q95 (mm) | 16.59 | 13.15 | 12.79 | 9.85 | 9.40 | 8.86 | 7.05 | 6.41 | 5.57 |
| mean over all 45 pairs (mm) | 2.95 | 2.33 | 2.50 | 1.58 | 1.88 | 1.26 | 1.46 | 1.51 | 1.26 |
| q95/q90 | 1.13 | 1.26 | 1.13 | 1.31 | 1.22 | **1.39** | 1.14 | 1.20 | 1.18 |

Cohort mean q95 = 9.97 mm, CV 0.34, sd(log A) = 0.34. Extreme pair is `01_06` or `01_07` for
8 of 9 patients (P1: `06_10`).

**Use q90, not q95.** The q95/q90 ratio ranges 1.13–1.39 and is largest for P7 and P9 — the
two E2 hold-outs, and the two patients where the models most overshot. Their fields are
heavy-tailed (P7 extreme pair: q95 8.86 mm but mean 2.41 mm, ratio 3.7 versus 2.6 for P4), so
q95 is measuring a thin tail that is plausibly registration noise in a low-SNR, low-motion
field. Never use max.

Note this reorders the cohort relative to the voxel-space ranking: **P7 is mid-cohort on
q90-in-mm, not the lowest breather**; it is joint-lowest only on mean-over-all-pairs. The
"heavy P4,P3,P1,P8 / mid P2,P9,P5 / low P6,P7" ranking is a mean-‖u‖-in-voxels ranking and
does not survive unit correction plus a percentile definition. This does not weaken H1
(§0.6), but any claim of the form "patient X is a low breather" must name its statistic.

### 0.3 The amplitude ratio predicts what the trained models actually did

For each hold-out patient/split, compare predicted amplitude ratio `Ā_train / A_test` against
the amplitude ratio Run 0 measured on the checkpoints (`‖pred‖/‖gt‖`), n = 7:

| Amplitude statistic | Spearman | Pearson | mean \|log error\| |
|---|---|---|---|
| q95, extreme pair, mm | +0.86 | **+0.96** | **0.10** |
| q90, extreme pair, mm | +0.93 | +0.98 | 0.13 |
| mean over all pairs, mm | +0.93 | +0.98 | 0.14 |
| mean over all pairs, resampled voxels | +0.93 | +0.98 | 0.11 |

Predicted vs observed for q90: P3 0.62/0.75, P4 0.48/0.67, P5 1.14/1.16, P7 1.67/1.54,
P9 1.42/1.40. So the models really are emitting the train-pool mean amplitude, and a simple
ratio of per-patient amplitude constants predicts the resulting error to ~13%.

**Every candidate statistic works equally well.** The conditioning variable is a *ratio*, and
the ratio is robust to the percentile choice — that argument is settled; pick q90 for
stability and move on.

### 0.4 Which folds can even test this

With the 3-fold LOPO protocol from P0-C, using q95-mm and train-pool-only constants:

| Fold | Test | `Ā_train` (mm) | `r_p = A_p/Ā_train` per test patient | α = 1/r |
|---|---|---|---|---|
| F1 | P3, P4, P5 | 8.88 | 1.44, **1.87**, 0.79 | 0.69, 0.53, 1.26 |
| F2 | P1, P2, P6 | 10.76 | 1.22, **0.60**, **0.52** | 0.82, 1.68, 1.93 |
| F3 | P7, P8, P9 | 10.26 | 0.86, 0.92, 0.96 | 1.16, 1.09, 1.04 |

**F3 is amplitude-neutral (all r ≈ 0.9) and cannot demonstrate amplitude conditioning.** The
signal lives in F1 (P4 under-predicted 1.9×) and F2 (P2, P6 over-predicted 1.7–1.9×). Pooling
the three folds into one number will dilute the effect by ~3×. Report per fold, per patient,
always — and expect a near-null result on F3 rather than treating it as a failure.

### 0.5 CT-only surrogates: rank correlation is misleading, calibration is what matters

Registration-free surrogates from the 10 phase volumes, scored two ways: Spearman against
`A_p`, and **strict LOPO calibration** (gain fitted on the 6 train patients only, then
`Â_p/A_p` on the 3 held-out patients; error = |log ratio|).

| Surrogate | Spearman | median err | mean err | max err |
|---|---|---|---|---|
| **ΔV, tidal lung-air volume (mL)** | +0.82 | 21% | **31%** | 80% |
| ΔV / (SI extent × LR extent) | +0.70 | 18% | 28% | 119% |
| ΔV / V̄^(2/3) | +0.85 | 33% | 39% | 111% |
| ΔV / V̄ (fractional) | +0.65 | 31% | 44% | 126% |
| lung-air AP centroid excursion | **+0.97** | 26% | 77% | **432%** |
| *train-pool mean prior (no surrogate)* | 0 | 26% | 38% | 93% |

Two conclusions that change the plan:

- **A rank-correlation gate is not sufficient.** The AP-centroid surrogate has the best
  Spearman in the cohort (+0.97) and the worst calibration (P6 under-estimated 5×). The gate
  in §5 must be a calibration gate on log-ratio.
- **On SPARE, the best CT-only surrogate is only marginally better than using the pool mean**
  (mean error 31% vs 38%). It beats the prior on 5 of 9 patients (P2, P3, P4, P5, P6) and
  loses on 4 (P1, P7, P8, P9) — and the 4 it loses on are exactly those whose prior ratio was
  already near 1. A noisy surrogate helps the outliers and hurts the typical patients. §4.4
  fixes this with shrinkage.

### 0.6 The external-surrogate arm is not testable on SPARE

Measured on the raw 1 mm volumes: anterior chest-wall AP surface position, peak-to-trough
over the 10 phases, central 130 mm LR, 40 mm slab at mid-lung:

| P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9 |
|---|---|---|---|---|---|---|---|---|
| 0.47 | 0.11 | 4.73 | 0.13 | 5.73 | 0.12 | 0.04 | 0.00 | 0.03 mm |

Seven of nine patients show ≤ 0.5 mm of chest-wall excursion; real chest-wall AP motion is
2–10 mm. An abdominal slab gives nonsense (P6: 148 mm) because the slab leaves the
reconstructed FOV. **The body surface does not move in these Monte-Carlo volumes** — the
deformation appears confined to an internal ROI. Therefore:

- No emulated-RPM / emulated-bellows arm can be validated on SPARE. Do not build one and do
  not report one; it would measure the phantom's construction, not physiology.
- The deployment-real arm (§4.2) needs a cohort with real surface motion and, ideally,
  recorded traces. 4D-Lung (TCIA) ships per-scan respiratory signal files and is the natural
  target; DIR-Lab adds landmarks for TRE in mm but no traces.
- Also confirmed: the 9 lung masks are 9 distinct anatomies (distinct checksums), so LOPO is
  not leaking a shared base phantom.

### 0.7 Segmentation stability predicts calibration failure

Spread of ΔV across three air thresholds (0.35/0.45/0.55 × the in-mask soft-tissue level),
relative to the median:

| P3 | P2 | P1 | P5 | P4 | P6 | P8 | P7 | P9 |
|---|---|---|---|---|---|---|---|---|
| 3% | 12% | 17% | 26% | 27% | 28% | 31% | 37% | 42% |

The three least stable (P7, P8, P9) are exactly fold F3 and exactly three of the four
patients where the surrogate lost to the pool-mean prior. Surrogate instability is a
*predictive* QC, not a formality. Root cause is likely my threshold being relative
(these volumes are in linear attenuation units, not HU); calibrating μ_water from a
water-equivalent ROI and using a fixed −400 HU lung threshold should tighten this materially,
and is a prerequisite for the measured arm.

---

## 2. Amplitude definition

### 2.1 Per-patient ceiling `A_p`

```
A_p = q90( ‖u‖ ) over the lung mask, on the patient's volume-extreme pair
```

- **Statistic**: 90th percentile. Justification in §0.2 (q95 is tail-contaminated exactly in
  the low-motion patients; max is meaningless). Report q95 as a sensitivity column.
- **Units**: millimetres, on the P0-A common isotropic grid (2.0 mm iso, 160³, lung-centroid
  centred). Because the grid is isotropic, `‖u‖` needs no per-axis correction and the value is
  invariant to any residual channel-order ambiguity.
- **Domain**: the *unpadded* lung mask on the common grid, resampled with nearest neighbour.
  The padded masks used for Elastix include 4–30 voxels of chest wall (they cover 39% of the
  crop for P4) and would mix chest-wall and lung statistics.
- **Extreme pair**: defined by the **surrogate curve** (argmax and argmin of the phase
  volume), not by the DVF, so that the oracle and measured arms use the same pair and the
  definition survives when no DVF exists. On SPARE this picks `01_06`/`01_07` for 8 of 9,
  agreeing with the DVF-chosen pair for 8 of 9 (P7 differs: DVF says `01_06`, volume says
  `01_07`).
- **Both directions**: average `q90` over the two directed pairs (i→j and j→i) to suppress
  registration asymmetry.
- **Scalar only for E6.** 3-axis `A_p` is deferred behind gate G0 (§5) and, given §0.1, is
  unlikely to generalise off this cohort.

### 2.2 Per-pair fraction `f`

```
f(θ_ref, θ_tgt) = |v(θ_tgt) − v(θ_ref)|        v = normalised surrogate value in [0,1]
a(θ_ref, θ_tgt) = A_p · f(θ_ref, θ_tgt)         (mm)   — the "pair gain"
```

`v` is the surrogate waveform min-max normalised over the cycle, so `v = 0` at end-expiration
and `1` at end-inspiration and `f ∈ [0,1]` with `f = 0` on identity pairs. This is the
critical structural choice: **`f` is a continuous function of the two phase coordinates, not a
10×10 lookup table**, which is what makes continuous `t_tgt` (§8) fall out for free.

At training the oracle counterpart is `f_ij^orc = q90(‖u_ij‖) / A_p`. Measured, per-patient
Spearman between `f^orc` and `|Δv|` from the lung-volume curve:

| P9 | P6 | P4 | P3 | P5 | P2 | P7 | P8 | P1 |
|---|---|---|---|---|---|---|---|---|
| 0.99 | 0.98 | 0.95 | 0.89 | 0.89 | 0.87 | 0.84 | 0.83 | **0.62** |

Eight of nine ≥ 0.83; **P1 fails at 0.62** and is flagged. P1 is also the only patient whose
extreme pair is not anchored on phase 01 (`06_10`), consistent with an irregular waveform. So
the within-patient phase modulation is well predicted by the volume curve even though the
cross-patient amplitude is not (§0.5) — the surrogate is much better at *shape of the cycle*
than at *absolute scale*.

### 2.3 Small-motion pairs

`a → 0` makes normalised targets ill-conditioned. Number of the 45 unordered pairs with
q95 < 2 mm: P6 17, P2 10, P5 9, P7 7, P8 6, P9 6, P1 3, P3 1, P4 1.

Rule: **exclude pairs with `a < 1.5 mm` from the shape loss**, and recover small-amplitude
coverage from P1-B augmentation with `s < 1` applied to well-conditioned pairs. Identity pairs
are exact by construction under the architecture in §6 and are dropped entirely — this
disposes of H8 (10% of samples teaching "predict zero") at no cost.

---

## 3. Normalised conditioning interface

The network never sees millimetres or millilitres. It sees one dimensionless number.

```
r̂_p  =  m_p / m̄_train^(mod)                      # dimensionless relative amplitude
```

where `m_p` is the **peak-to-trough excursion** of whatever surrogate modality is available
and `m̄_train^(mod)` is the mean of that same quantity over the fold's training patients.
Peak-to-trough is used because it is offset-free — an RPM trace has an arbitrary baseline,
and this kills it by construction, leaving a single multiplicative gain to calibrate.

Any modality enters through the same door:

| Modality | `m_p` | Fold constant |
|---|---|---|
| Oracle DVF | `A_p` (q90, mm) | `Ā_train` (mm) |
| CT lung volume | tidal ΔV (mL) | `ΔV̄_train` (mL) |
| Diaphragm tracking | SI excursion (mm) | mean (mm) |
| RPM / bellows | marker peak-to-trough (mm) | mean (mm) |
| Spirometry | tidal volume (mL) | mean (mL) |

A single per-modality gain then maps to millimetres: `Â_p = Ā_train · r̂_p`. Only a
proportional relation is assumed; a two-parameter affine calibration is *not* recommended at
n = 6 training patients — it would fit noise, and §0.5 shows even the one-parameter version is
barely better than a constant.

Fed to the network as `log r̂_p`, clipped to `[log 0.4, log 2.5]`. Log because amplitude acts
multiplicatively, the distribution is right-skewed, and 0 means "population mean" so that
`r̂ = 1` is a natural neutral default for the fallback arm. "0.7× population mean" is then
literally the same input whether it came from RPM millimetres, spirometry millilitres, or the
lung-volume curve.

**Fold-safe by construction.** `Ā_train`, `m̄_train`, and the clip range are computed on the
fold's training patients only, stored in the fold manifest next to the split definition, and
never recomputed at test time. For the 3-fold protocol the constants are in §0.4 — note they
differ by 21% between F1 and F2, so a single cohort-wide constant would leak and would also
shift the meaning of `r̂ = 1` between folds.

**Shrinkage (§4.4) is part of the interface, not an afterthought**: what is fed is
`λ · log r̂_p`, with `λ` fixed per arm from the train pool.

---

## 4. Oracle path vs measured/deploy path

Keep these four things separate. Conflating them is the single easiest way to publish a
circular result.

### 4.1 Oracle path — training labels and the identifiability proof

`A_p` from the patient's own GT DVF.

- **For training patients this is legitimate and permanent.** The training corpus is a 4D CT
  cohort; using its DVFs to define labels is not leakage.
- **For the held-out patient it is leakage** and exists only as an upper bound. Label the arm
  "oracle (upper bound, not deployable)" in every table.
- Purpose: answer "if the amplitude were known exactly, is the residual error acceptable?"
  Run 0 already answers yes (P7 1.09 → 0.69). E6's oracle arm confirms it end-to-end and sets
  the ceiling every other arm is measured against. Per `ChangesNeeded` §5, if the oracle arm
  fails, abandon the whole line and pivot to P1-D / P2-A.

### 4.2 Deployment-real path — external sensor

The target: RPM / bellows / spirometry recorded during the planning scan, `m_p` = peak-to-trough
of the trace over a representative window (recommend the 90th minus 10th percentile of the
trace, not the absolute extremes, for the same tail reason as §0.2).

**Status on SPARE: not testable (§0.6).** No trace files exist in the local Monte-Carlo
distribution, and the body surface does not move in the volumes. Action items, in order:

1. Check the original SPARE release for the breathing signals used to animate the phantoms —
   if the traces exist, they are the correct external surrogate and this arm becomes real.
2. Otherwise defer this arm to 4D-Lung (TCIA), which ships respiratory signal files, and
   report the SPARE result as internal-surrogate-only.
3. In the meantime, characterise sensitivity *analytically* rather than fabricating a sensor:
   the tolerance model in §9 converts any assumed surrogate error into a predicted L1/zero,
   which is the honest version of "what if RPM is off by X%".

### 4.3 Experiment-only 4D-derived surrogate — the SPARE measured arm

Lung-air tidal volume ΔV from the phase volumes by thresholding, no registration. This is a
legitimate *measured* arm — it uses no DVF — but it needs multi-phase imaging, so it belongs
to the motion-model-building scenario, not to single-CT deployment. Say so in the paper.

Required hardening before use: HU calibration from a water-equivalent ROI, a fixed −400 HU
lung threshold, connected-component lung extraction excluding trachea and main bronchi, and
the §0.7 threshold-sensitivity report. Patients whose ΔV spread exceeds 25% (P7, P8, P9 today)
carry a flag through every results table.

### 4.4 Static single-CT proxy and the fallback prior

The only truly single-CT amplitude estimator is a regression from static anatomy:
`r̂_p = g(CT)` from lung volume at the scanned phase, body cross-sectional area, HU
histogram / emphysema index, diaphragm curvature, ribcage AP:LR ratio. **With 6 training
patients this is underpowered and must be reported as such** — a LOO ridge on at most two
standardised features, with the honest statement that n = 9 cannot establish it. Its value is
as a stated research direction and as the thing the fallback replaces.

The fallback is `r̂ = 1` (train-pool mean), which is exactly arm (iv) and the graceful-degradation
floor: the model then reproduces today's behaviour and can never be worse than the baseline.

**Precision-weighted shrinkage — feed a blend, not the raw surrogate.** Because
`log r̂_measured = log r_true + ε`, the minimum-MSE estimate is `λ · log r̂_measured` with
`λ = Var(log r_true) / (Var(log r_true) + σ²_ε)`. Measured on SPARE: `sd(log A_p) = 0.34` and
`σ_ε ≈ 0.34` (from the mean |log| error of 0.27 in §0.5), giving **λ = 0.5**, i.e. use the
geometric mean of the surrogate estimate and the pool mean. Verified under strict LOPO:

| λ | mean err | max err | sd(log) | per-patient `Â/A`, P1…P9 |
|---|---|---|---|---|
| 0.00 (pool mean) | 38% | 93% | 0.219 | 0.82, 1.68, 0.69, 0.53, 1.26, 1.93, 1.16, 1.09, 1.04 |
| 0.25 | 27% | 56% | 0.137 | 0.79, 1.30, 0.75, 0.65, 1.21, 1.56, 1.00, 1.24, 1.07 |
| **0.50** | **20%** | **40%** | **0.091** | 0.77, 1.01, 0.82, 0.79, 1.16, 1.27, 0.87, 1.40, 1.11 |
| 0.75 | 21% | 59% | 0.134 | 0.75, 0.79, 0.89, 0.96, 1.11, 1.02, 0.76, 1.59, 1.14 |
| 1.00 (raw surrogate) | 31% | 80% | 0.184 | 0.72, 0.61, 0.96, 1.16, 1.07, 0.83, 0.65, 1.80, 1.17 |

λ = 0.5 compresses every patient into `Â/A ∈ [0.77, 1.40]`, versus `[0.61, 1.80]` raw and
`[0.53, 1.93]` for the prior. It is also structurally safe: a worthless surrogate drives
`λ → 0` and the arm degrades to the prior rather than below it. `λ` is estimated on the train
pool only (it needs `sd(log A)` over train patients and the surrogate's train-pool
cross-validated error), so it is fold-safe.

---

## 5. Pre-training validation protocol

Run all gates before any GPU time. Each has a pre-committed decision.

**G0 — units and conventions.** After P0-A: isotropic 2.0 mm grid; `warp(ref, u_regridded)`
residual no worse than pre-regrid (guards interpolation damage); channel-permutation warp test
reconfirmed on the new grid; per-channel |u| profile and caudal-vs-cranial breakdown recorded
per patient. *Gate:* pass required before anything else. *If the permutation test no longer
picks the documented mapping, stop and fix the regrid.* 3-axis amplitude stays deferred
regardless.

**G1 — patient-level calibration (decides the measured arm).** Strict LOPO: fit the gain on
the fold's 6 train patients, predict `Â_p` for the 3 test patients, report `Â_p/A_p` per
patient and `|log|` summary. *Pass:* mean error ≤ 25% **and** no patient worse than 1.6× or
better than 0.6×, after λ-shrinkage. *Current status:* ΔV at λ = 0.5 gives mean 20%, max 40%
— **passes**, but only because of shrinkage; raw ΔV (mean 31%, max 80%) fails. Report both.
Do not gate on Spearman: §0.5 shows a +0.97 rank correlation coexisting with 432% calibration
error.

**G2 — pair-level waveform fidelity.** Per patient, Spearman between `f^orc` and `|Δv|`.
*Pass:* ≥ 0.80 for ≥ 8 of 9 patients. *Current:* 8 of 9 pass; **P1 = 0.62, flagged**. Carry
the flag into results and check whether P1 is the measured arm's worst hold-out.

**G3 — surrogate stability.** ΔV recomputed across three thresholds and (once available) two
segmentation methods. *Flag:* spread > 25%. *Abort the measured arm for that patient:* > 50%.
*Current:* P7 37%, P8 31%, P9 42% flagged; none aborted. Also plot all 9 volume curves and
require a single smooth minimum and maximum per cycle — a double-peaked or noisy curve means
the extreme pair is being chosen by noise.

**G4 — amplitude ratio predicts model behaviour.** Reproduce §0.3 on the regridded data
against Run 0's `‖pred‖/‖gt‖`. *Pass:* Pearson ≥ 0.9. This is the sanity check that the whole
premise survived the regrid.

**G5 — fold sensitivity.** Report `r_p` per fold before training (§0.4) and pre-register the
expectation that F3 shows little or no gain. Prevents a null on F3 being read as a failure of
the method.

Reporting for all gates: **per patient, both original splits (E1/E2), and all 3 LOPO folds.**
Never a single pooled correlation.

---

## 6. Plugging into Decoder-CRB

Current state: `UNetCRBDecoder` has `cond_dim = 2`, four `UpCRB` blocks each with a private
FC `cond_dim → 32 → 16 → 2·out_ch`, and `_phase_vec` returns
`[phase_ref/9, phase_tgt/9]` — a **linear, non-cyclic** encoding in which phase 10 and phase
01 are maximally distant despite being adjacent in the cycle.

### 6.1 Conditioning vector — `cond_dim = 7`

```
c = [ cos 2πθ_ref, sin 2πθ_ref,
      cos 2πθ_tgt, sin 2πθ_tgt,
      δ_tgt,                       # +1 inhaling, −1 exhaling (hysteresis)
      λ·log r̂_p,                   # patient amplitude, dimensionless
      f(θ_ref, θ_tgt) ]            # pair fraction in [0,1]
```

Cyclic `(cos, sin)` fixes the wrap-around defect and makes `θ` continuous. `δ_tgt` is needed
because the cycle has real hysteresis: the diaphragm trajectories show P4 at ‖u‖ 1.17 on
phase 05 versus 1.01 on phase 07 at comparable volumes, so `v` alone does not determine the
field. `f` is included even though it also multiplies the output (§6.2), because *shape*
depends on amplitude — deep breaths recruit different regions — while *scale* must not be
learned.

Reduced 5-D variant for ablation: drop `δ_tgt` and `f`. Widening 2 → 7 adds only
`5 × 32 = 160` weights per CRB (4 CRBs → 640 parameters on 1.07 M), so capacity is not a
concern; the risk is the opposite, that a 7-D input to a 32-unit FC lets the network ignore
the amplitude channel, which is what §6.4 prevents.

### 6.2 Analytic amplitude gain

```
u_pred(x) = a(θ_ref, θ_tgt) · shape_net(CT_ref, c)(x)        a = Ā_train · λ-shrunk r̂_p · f
```

The network's output head predicts a **unit-amplitude shape field** (targets normalised so
`q90(‖û‖) = 1`), and the millimetre scale is applied as an exact multiplicative constant
outside the network. This matters more than putting `log r̂` in the conditioning vector: it
makes the amplitude path non-learnable, so the network *cannot* regress to the pool mean, and
`r̂` at inference acts as a pure gain knob. Identity pairs give `a = 0` and are exact.

### 6.3 Loss: shape space or millimetre space

With an exact gain, MSE in millimetres equals `a²` × MSE in shape space — so the choice is
purely a per-sample weighting, `w = 1` (shape) versus `w = a²` (mm).

**Train in shape space (`w = 1`).** Under `w = a²` a single P4 `01_06` pair carries
`(14.7/4.7)² ≈ 10×` the gradient of a P6 pair, so the population prior is currently learned
almost entirely from the heavy breathers — which is precisely the failure mode. Equal
weighting is the point of the decomposition. Report a `w = a` (square-root) variant as a
one-line ablation, and always report the metric in millimetres regardless of training weight.

Add the direction (cosine) term from P2-C in shape space; it is scale-free and therefore
exactly aligned with what the shape network is now responsible for.

### 6.4 Interaction with P1-B amplitude augmentation

Under the §6.2 architecture the augmentation is exact and trivial: for factor `s`, the triplet
`(ref, warp(ref, s·u), s·u)` has the **same normalised shape target** and `a → s·a`, i.e.
`r̂ → s·r̂` in the conditioning. Labels scale with `s` by construction — no relabelling, no
recomputation of `A_p`.

Rules:

- Augmentation pairs always use **oracle** amplitude (`r̂ = s · A_p/Ā_train`). Do not attempt
  to synthesise a matching surrogate reading; there is no such thing for a synthetic image.
- The measured arm therefore trains on: real pairs with measured `r̂`, plus augmented pairs
  with oracle `r̂`. Report the mix fraction. Keep ≥ 50% real pairs per batch (interpolated
  pseudo-targets drift from real CT statistics).
- `s ∈ [0.4, 2.0]`, sampled log-uniform so the `log r̂` input is uniformly covered — the
  distribution of `log r̂` seen in training must cover the clip range of §3, otherwise the
  gain knob is untested at its extremes.
- This is what breaks the anatomy↔amplitude correlation. With 6 training patients the network
  can otherwise memorise 6 amplitudes from anatomy; then `r̂` is decorative and the measured
  arm is indistinguishable from the baseline. Treat P1-A and P1-B as one intervention.
- Geometric augmentation (LR flip with negation of the LR channel, small rotations with the
  vectors rotated) is orthogonal and amplitude-preserving; `A_p` is unchanged.

---

## 7. Operating modes

### Mode 1 — single CT + external amplitude (the north star)

Inputs: one CT at known `θ_ref`; `r̂_p` from the external surrogate; any `θ_tgt`.
Output: `u(θ_ref → θ_tgt)` and, by sweeping, the full loop with dwell weights.

`θ_ref` is a *required input*, not an optional one. With a trace recorded during the scan,
`v` at scan time gives it directly. Without a trace it must be assumed (mid-exhale is the
usual convention for a free-breathing planning CT) and the resulting error is a first-class
failure mode (§9). This is worth stating loudly: the design needs the surrogate for the
*phase* of the reference scan as much as for the amplitude.

### Mode 2 — refinement when any second observation appears

If a CBCT, a partial scan, or one extra phase becomes available later, refine the single
scalar with P1-C: fit `α` by minimising masked NCC between `warp(CT_ref, α·u_pred)` and the
observed image, `α` initialised at 1 (the amplitude is already approximately right, so this is
a small correction rather than a rescue). Differentiable `grid_sample` warping already exists.

Mode 2 relationship to Mode 1, stated precisely:

- Mode 2 consumes target-phase imaging, so it is **not** the north-star scenario; it is the
  motion-model-building / on-treatment-adaptation scenario.
- Its value in the paper is as the **legitimate realisation of the oracle scale**: Run 0 says
  oracle rescaling recovers most of the E2 loss, and P1-C recovers it without labels. It
  should land within ~0.05 L1/zero of the oracle-scale number.
- Because Mode 1's amplitude is already within ~20–40% after §4.4, Mode 2's job shrinks from
  "fix a 1.8× error" to "trim a 1.2× error" — and the composition should be reported: Mode 1
  alone, Mode 2 alone (i.e. `r̂ = 1` plus test-time fit), and Mode 1 + Mode 2.
- Given §0.6, on SPARE **Mode 2 is the only route to a genuinely non-circular amplitude**.
  This should be said plainly rather than hidden.

---

## 8. Continuous phase requirement

Explicit requirements the amplitude bolt-on must not break:

1. **Cyclic continuous coordinate.** `θ ∈ [0,1)` with `(cos, sin)` encoding, replacing
   `phase/9`. Both `θ_ref` and `θ_tgt` continuous; the 10 discrete phases become 10
   non-uniformly spaced samples.
2. **`θ` defined by the surrogate, not the clock.** `θ = ½v` on the inhale limb and
   `1 − ½v` on the exhale limb. So `θ` indexes the *filling state*, which is what the motion
   field depends on, and it is directly readable from an RPM trace at inference. A clock-based
   `θ` would be wrong for irregular breathing — the case that matters clinically.
3. **`f` as a continuous function.** `f = |v(θ_tgt) − v(θ_ref)|` (§2.2) is defined for
   arbitrary real `θ`, so continuous inference needs no interpolation table and the amplitude
   path stays exact at unseen phases.
4. **Continuity and consistency QC**, evaluated at 100 points around the loop on hold-out
   patients:
   - smoothness of `‖u(θ)‖` and of the field itself (no jumps between trained phases);
   - loop closure `‖u(θ=0) − u(θ=1)‖ ≈ 0`;
   - antisymmetry `u(a→b) ≈ −u(b→a)` after composition;
   - transitivity `u(a→c) ≈ u(a→b) ∘ u(b→c)`;
   - a diaphragm landmark tracing a closed hysteresis loop with the correct limb ordering;
   - Jacobian determinant > 0 everywhere (no folding), at intermediate `θ` as well as trained
     ones — untrained phases are where folding will appear first.
5. **Dose-calculation output contract.** N warped CTs at `θ_k` plus **dwell weights** from the
   surrogate's time histogram, not uniform weights — a real cycle spends more time near
   end-expiration. The weights come from the trace, so this is another thing the amplitude
   surrogate delivers beyond a scalar.
6. **Amplitude and phase must stay orthogonal.** Test explicitly: fix `θ` and sweep `r̂` over
   the clip range; the field direction should be near-constant (cosine between the fields
   > 0.99) with magnitude scaling linearly. If sweeping `r̂` rotates the field, the shape
   network has entangled the two and the gain knob is not interpretable.

---

## 9. Failure modes and graceful degradation

### 9.1 Validated tolerance model

For a prediction with cosine `c` against truth and amplitude ratio `k = ‖pred‖/‖gt‖`:

```
L1/zero  ≈  sqrt( 1 + k² − 2kc )
```

Checked against Run 0, E2 Encoder P7 (c = 0.773): `k = 1.54` → predicted 1.00, observed 1.09;
`k = 1` → predicted 0.67, observed 0.69 (oracle). Good enough to design with.

Consequences at `c = 0.78`:

- Optimal gain is `k* = c = 0.78`, **not 1.0**. Metric-optimal is deliberately under-scaled.
- The band keeping L1/zero ≤ 0.80 is `k ∈ [0.28, 1.28]`; for ≤ 0.70 it is `k ∈ [0.47, 1.09]`.
- **The curve is strongly asymmetric: underestimating amplitude by 70% is as cheap as
  overestimating by 28%.** Every design choice should therefore bias low, which is exactly
  what the λ-shrinkage of §4.4 does, and it explains why E1 "worked" and E2 did not.
- Under log-normal surrogate error with sd σ, the optimal gain is `k₀ = c · exp(−1.5σ²)`. At
  σ = 0.34 that is `0.78 × 0.84 = 0.66`.

**Predicted outcome of the measured arm with the full recipe** (λ = 0.5 shrinkage, then gain
`c·Â`), applied to the §4.4 LOPO ratios at c = 0.78: `k ∈ [0.60, 1.09]` and predicted L1/zero
**0.63–0.70 for all nine patients** — against 0.64–1.10 today. That is the pre-registered
prediction E6 should be checked against. If the measured arm lands materially above 0.75 on
any patient, either the cosine assumption or the calibration failed, and the per-patient
`k` and `cos` columns will say which.

### 9.2 Metric-optimal versus physically-correct scale

The `k = c` shrinkage minimises L1 but produces DVFs that systematically under-state motion by
~22%, which under-states motion blurring in a 4D dose calculation — the actual clinical use.
**Report both**: `k = 1` (unbiased, for dose and for ITV margins) and `k = c` (metric-optimal,
for comparability with the existing tables). Do not silently ship the shrunk field as the
product. Any reviewer who knows the L2 shrinkage argument will ask.

### 9.3 Enumerated failure modes

| Failure | Mechanism | Detection | Degradation |
|---|---|---|---|
| Surrogate systematically off by gain `g` | RPM-to-internal coupling differs from the training cohort | G1 LOPO calibration; per-patient `k` column | λ-shrinkage caps the damage; §9.1 band converts `g` to predicted L1/zero |
| Surrogate uncorrelated for a patient | chest vs abdominal breather, poor marker placement | G2 pair-level Spearman < 0.6 (P1 today) | fall back to `r̂ = 1`; report as "prior-only" |
| Segmentation instability | HU threshold sensitivity | G3 spread > 25% (P7, P8, P9 today) | flag; prefer prior; require two segmentation methods to agree within 15% |
| `θ_ref` unknown or wrong | no trace at scan time | sweep `θ_ref` and report worst case | report a `θ_ref` sensitivity band, not a point estimate |
| Amplitude conditioning ignored | anatomy↔amplitude memorised at n = 6 | sweep `r̂`, measure `d‖u‖/d log r̂`; should be ≈ 1 | P1-B augmentation is the fix, not a knob |
| Shape error dominates | cosine below the atlas ceiling | oracle-scale L1/zero stays high | pivot to P1-D per `ChangesNeeded` §5 |
| Irregular / non-stationary breathing | amplitude is not one number | trace percentile spread per cycle | condition on a per-cycle `r̂`; report the within-patient spread as irreducible |

### 9.4 Uncertainty output

Cheap and worth having: propagate the surrogate's calibration sd `σ` (from the fold's
cross-validated LOPO residual, a train-pool-only quantity) into a predicted-error band via
§9.1, and emit `u_pred` together with `[k_lo, k_hi]` at `exp(±σ)`. This costs nothing at
inference, gives clinicians a motion-envelope rather than a point field, and is the natural
place for a robust-margin argument. A learned heteroscedastic head is not justified at n = 9.

---

## 10. Minimal E6 experiment matrix

Decoder-CRB only (E1's best), P0-A regridded data, P0-C LOPO checkpoint selection, 3-fold
LOPO (F1/F2/F3) plus the legacy E1 and E2 splits for continuity. All arms share seeds,
schedule, and augmentation except where stated.

| Arm | Target | `r̂_p` at test | `f` | Purpose |
|---|---|---|---|---|
| **A0** baseline | raw DVF (mm) | — | — | reproduces today's behaviour on regridded data |
| **A1** normalise only | unit shape × `a`, `r̂ = 1` | 1 (pool mean) | oracle | isolates normalisation + shape-space loss from conditioning; the honest fallback |
| **A2** oracle amplitude | unit shape × `a` | `A_p/Ā_train` (leaks; upper bound) | oracle | identifiability proof; the ceiling |
| **A3** measured amplitude | unit shape × `a` | λ-shrunk ΔV surrogate | from `v` curve | the reportable result |
| **A3-raw** | as A3 | raw ΔV, λ = 1 | from `v` curve | quantifies the shrinkage contribution |
| **A4** oracle amp, measured `f` | unit shape × `a` | oracle | from `v` curve | separates patient-scale error from within-cycle error |

All arms with P1-B augmentation on; one extra run of A3 with augmentation off to confirm the
conditioning would otherwise be ignored (§6.4). Total: 6 arms × 3 folds + 1 ablation ≈ 19
runs; at E1's cost with AMP and batching this is roughly a day of GPU.

Metrics per patient, per fold: L1 (mm), L1/zero, beat-zero %, cosine, `k = ‖pred‖/‖gt‖`,
oracle-scale L1/zero, plus `Â_p/A_p` for the measured arms and the G3 stability flag.
Baselines in every table: zero DVF, LOO atlas unscaled, LOO atlas oracle-scaled.

Pre-registered expectations, so that a null is interpretable:

- A2 ≫ A0 on F1 and F2; A2 ≈ A0 on F3 (`r ≈ 0.9`, §0.4).
- A3 between A1 and A2, closer to A2 on P2/P4/P6 and closer to A1 on P1/P7/P8/P9 (§0.5).
- A3-raw worse than A3 on P8 specifically (raw ratio 1.80 vs 1.40 shrunk).
- A1 already better than A0 on F2, from shape-space loss reweighting alone — do not
  misattribute that gain to conditioning.

---

## 11. Success criteria

Tied to the north star; all on the P0-A common grid, in millimetres.

**Must hold for the programme to continue**

1. **A2 (oracle) reaches L1/zero ≤ 0.70 and `k ∈ [0.85, 1.15]` on every hold-out patient in
   F1 and F2.** If knowing the amplitude exactly does not fix it, H1 is wrong — pivot to
   P1-D / P2-A immediately (`ChangesNeeded` §5).
2. Every arm beats the **unscaled LOO atlas** on all 9 patients. The 1.07 M-parameter network
   must add something to an average.
3. Gains hold across F1 and F2 without degrading the other; a change that helps one split and
   hurts the other by a comparable amount is amplitude recalibration, not generalisation.

**Deployment-relevant success (Mode 1)**

4. **A3 achieves L1/zero ≤ 0.75 and beat-zero ≥ 80% on all nine patients** at the
   metric-optimal gain, i.e. no patient left in P7's current regime, using only
   single-CT-compatible inputs at test time. At the unbiased gain (`k = 1`, the dose-relevant
   one) the same recipe predicts 0.63–0.88 with the worst case at **P8 (0.88)** — the patient
   carrying the G3 instability flag. So criterion 4 is stated for the metric-optimal gain and
   P8 is the pre-identified exception; if P8 is the only patient above 0.75 at `k = 1`,
   improving its lung segmentation is the fix, not the model.
5. `k ∈ [0.6, 1.3]` for every patient — inside the §9.1 tolerance band. Predicted per-patient
   L1/zero at `k = 1` from the §4.4 ratios: P1 0.63, P2 0.67, P3 0.63, P4 0.63, P5 0.73,
   P6 0.80, P7 0.63, P8 0.88, P9 0.71, versus the A1 prior's 0.63, 1.10, 0.63, 0.67, 0.79,
   1.31, 0.73, 0.70, 0.68 — A3 better on 7 of 9, worse on P8 and P9 only.
6. A3 ≥ A1 on at least 7 of 9 patients. **If A3 does not beat A1, say so.** Given §0.5 this is
   a live possibility, and "a CT-derived amplitude surrogate adds little over the population
   mean on n = 9, whereas the oracle amplitude fixes the error entirely" is a real and
   publishable result: it localises the remaining problem in *amplitude estimation*, not in
   the conditioning mechanism, and it motivates the external-sensor cohort.

**Continuity and physical validity**

7. Loop closure and antisymmetry residuals < 10% of `‖u‖` at 100 sampled phases; Jacobian
   determinant positive everywhere, including untrained phases.
8. Amplitude/phase orthogonality: sweeping `r̂` scales `‖u‖` linearly (slope 0.9–1.1 in log-log)
   with field cosine > 0.99.
9. Both `k = 1` and `k = c` variants reported (§9.2).

**Honest-negative criteria** — declare and stop

10. A2 fails criterion 1 → amplitude is not the binding constraint; pivot.
11. A3 ≈ A1 and Mode 2 is required to reach criterion 4 → the contribution is a *motion-model
    builder*, not single-CT synthesis. Reframe the claim rather than the method.
12. Any gain < 0.03 L1/zero, which is inside the architecture spread already seen on a single
    split.

---

## 12. Probes behind §0, for scripting

Each was run ad hoc for this document and should become a script under `scripts/` with output
under `AmplitudeConditioning/`:

| Probe | Measures | §  |
|---|---|---|
| channel-permutation warp test | 6 perms × 8 signs vs image residual | 0.1 |
| per-channel and caudal/cranial \|u\| profile | axis dominance | 0.1 |
| `A_p` at q90/q95/mean, per patient, both pair directions | amplitude definition | 0.2 |
| predicted vs Run 0 observed `‖pred‖/‖gt‖` | definition validity | 0.3 |
| fold train-pool constants and `r_p` | fold sensitivity, fold-safe constants | 0.4 |
| surrogate table with strict-LOPO calibration | surrogate choice | 0.5 |
| raw-1 mm body-surface excursion | external-arm feasibility | 0.6 |
| ΔV threshold sensitivity | segmentation stability | 0.7 |
| λ sweep under LOPO | shrinkage constant | 4.4 |

Note the §0 numbers use the *padded* lung masks and per-patient anisotropic mm/voxel; all must
be recomputed on the P0-A grid with unpadded masks before E6. The design choices they support
(q90, scalar, ratio-based conditioning, λ = 0.5, shape-space loss, ΔV over AP-centroid) are
not expected to change, because each rests on a ratio or a rank rather than an absolute scale.
