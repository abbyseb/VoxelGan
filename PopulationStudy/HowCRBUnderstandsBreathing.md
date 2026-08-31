# How the CRB architecture relates to breathing — mechanism and evidence

This document explains what “understanding breathing” means in **PopulationStudy**, how **Conditional Residual Blocks (CRB)** inject respiratory phase, and what experiments support (or refute) each claim.

Related write-ups: [`results.md`](results.md), [`ChangesNeeded.md`](ChangesNeeded.md), [`AmplitudeConditioning.md`](AmplitudeConditioning.md).

---

## TL;DR

| Claim | Verdict |
|-------|---------|
| The model maps **(reference CT, phase A, phase B) → DVF** supervised by Elastix | **Yes** — that is the training objective |
| CRB lets the same anatomy produce **different motions** for different phase pairs | **Yes** — FiLM-style conditioning on `(t_ref, t_tgt)` |
| The model **understands breathing physics** (biomechanics, patient-specific amplitude) | **No** — it imitates a phase-indexed population motion prior |
| Works on **unseen patients** within SPARE | **Partly** — strong on large-motion hold-outs (E1); weak on low-motion (E2) |
| Works **clinical / DIR** zero-shot | **Mixed / weak** — clinical cos ~0.5–0.6; DIR TRE ≈ identity |

---

## 1. What “breathing” means to this model

4DCT provides **10 respiratory phases** (inhale → exhale → inhale). The network does **not** model lungs, diaphragm, or airflow. It learns:

```
(reference CT at phase t_ref,  t_ref,  t_tgt)  →  3D displacement field u(x)
```

with `u` trained to match **Elastix** B-spline DVFs on SPARE MC.

| Input | Role |
|-------|------|
| **Reference CT** | Where anatomy is *now* |
| **`ref_phase`, `target_phase`** (0–9) | Where you are on the breathing cycle and which frame to reach |
| **Output** | DVF: per-voxel displacement ref → target |

Training uses **all directed phase pairs** (100 per patient, including identity). The model sees the full breathing trajectory many times per patient.

**No patient ID** is given at inference — only phase codes, same as Dan 2.0.

---

## 2. Phase encoding

### Linear (E1, E6 Normal)

Phases are normalised to `[0, 1]` and concatenated (`cond_dim = 2`):

```python
# InitialExperiments/Experiment1/networks/generator_crb_both.py
def _phase_vec(self, ref_phase, target_phase):
    denom = float(max(self.n_phases - 1, 1))
    t_ref = ref_phase.float().unsqueeze(-1) / denom
    t_tgt = target_phase.float().unsqueeze(-1) / denom
    return torch.cat([t_ref, t_tgt], dim=1)
```

### Cyclic (E2, E3, E6 Cyclic)

`cond_dim = 4`: `[cos 2πθ_ref, sin 2πθ_ref, cos 2πθ_tgt, sin 2πθ_tgt]` so inhale and exhale are treated as **periodic**, not just two scalars in `[0, 1]`.

---

## 3. What CRB does

**CRB = Conditional Residual Block** (Sang & Ruan, Med Phys 2023; replicated in Dan 2.0 / PopulationStudy).

```python
# InitialExperiments/Experiment1/networks/generator_crb.py
def forward(self, x, cond):
    ab = self.fc(cond)           # phase pair → MLP → per-channel (a, b)
    a, b = ab.chunk(2, dim=1)
    a = a.view(-1, self.out_ch, 1, 1, 1)
    b = b.view(-1, self.out_ch, 1, 1, 1)
    y = self.conv1(x)
    y = y * a + b                # FiLM-like scale + shift
    y = F.relu(y, inplace=True)
    y = self.conv2(y)
    ...
```

- **Convolutions** extract **spatial** structure from the reference CT.
- **CRB** modulates those features by **which motion** you want (ref → target phase).
- Without conditioning, a plain U-Net would have to ignore phase or bake it into the image.

### Three architecture ablations

| Variant | Encoder | Decoder | Design intent |
|---------|---------|---------|---------------|
| **EncoderCRB** | CRB + phase | plain conv | Phase affects early features |
| **DecoderCRB** | plain conv (anatomy only) | CRB + phase | Encode anatomy first; inject phase when building DVF |
| **BothCRB** | CRB | CRB | Phase everywhere |

DecoderCRB docstring:

> Encoder + bottleneck: plain residual blocks (anatomy only, no phase).  
> Decoder: CRB after skip concat (phase codes `[t_ref, t_tgt]`).

**Source files:**

- `InitialExperiments/Experiment1/networks/generator_crb.py` — EncoderCRB  
- `InitialExperiments/Experiment1/networks/generator_crb_dec.py` — DecoderCRB  
- `InitialExperiments/Experiment1/networks/generator_crb_both.py` — BothCRB  

### Data flow (conceptual)

```
Reference CT ──► U-Net encoder ──► spatial features ──┐
                                                      ├──► CRB (y = conv(x)·a + b) ──► 3-ch DVF
Phase (t_ref, t_tgt) ──► FC ──► (a, b) ──────────────┘
```

- **CT** → *where* the tissue is  
- **Phase** → *which point on the breath* and *which target frame*  
- **CRB** → *how* feature maps are scaled/shifted for that phase pair  
- **Loss** → MSE vs Elastix on thousands of `(patient, t_ref, t_tgt)` examples  

---

## 4. Two evaluation regimes (do not confuse them)

| | Dan 2.0 | PopulationStudy |
|--|---------|-----------------|
| Split | Leave-**phase**-out (same patient P1) | Leave-**patient**-out |
| What is unseen | Phase pairs of **known** anatomy | Entire **new** patients |
| Difficulty | Moderate | Much harder |

Dan 2.0 already knows the patient’s lungs; PopulationStudy must generalise anatomy **and** motion from 4–6 training patients.

---

## 5. Evidence the architecture learns phase → motion

### 5.1 Same patient, unseen phases (Dan 2.0)

Strongest proof that **CRB + phase conditioning can represent breathing** when anatomy is fixed.

| Metric | Dan 2.0 hold-out (P1, leave-phase-out) |
|--------|----------------------------------------|
| L1 vs Elastix | **~0.20–0.25** (best Decoder FiLM **0.196**) |
| **cos** (direction) | **~0.95** |

Source: [`results.md`](results.md) §1, `Dan2.0/results.md`.

### 5.2 Beats “do nothing” on unseen patients (E1)

**Zero DVF** = identity warp (no registration). **Beat zero** = predicted L1 &lt; zero-DVF L1 on that directed pair.

SPARE leave-patient-out, directed pairs, final checkpoint ([`results.md`](results.md) §3):

| Study | Model | L1 | L1 / zero | **Beat zero** | **cos** |
|-------|-------|-----|-----------|---------------|---------|
| **E1** | Encoder | 0.390 | 0.79 | 86% | 0.712 |
| **E1** | **Decoder** | **0.370** | **0.75** | **94%** | **0.759** |
| E2 | Encoder | 0.300 | 1.02 | 56% | 0.781 |
| E2 | Decoder | 0.321 | 1.06 | 54% | 0.786 |

**E1 Decoder: 94% beat-zero** — on most ref→target pairs for hold-out P3–P5, the model beats doing nothing. That requires `target_phase` to change the output; a zero DVF ignores phase.

Per-patient (E1 Decoder):

| Patient | Zero L1 (motion size) | Pred L1 | L1 / zero | Beat zero | cos |
|---------|----------------------|---------|-----------|-----------|-----|
| P3 | 0.586 | 0.370 | 0.63 | **100%** | 0.77 |
| P4 | **0.684** | 0.401 | **0.59** | **100%** | **0.83** |
| P5 | 0.371 | 0.339 | 0.91 | 82% | 0.69 |

High **cos (~0.76–0.79)** with L1/zero ~1 on E2 means: **direction is often right; magnitude is often wrong** (under-predicted / too smooth).

### 5.3 CRB placement ablation (E1)

On population hold-out, **DecoderCRB** beat EncoderCRB:

- L1: **0.370** vs 0.390  
- Beat zero: **94%** vs 86%  
- cos: **0.759** vs 0.712  

Supports: **encode anatomy without phase, inject phase in the decoder** for better generalisation. Not proof CRB is always optimal (BothCRB did not win hold-out), but proof **where** conditioning sits matters.

QC plots: `InitialExperiments/Experiment1/{Decoder,Encoder}CRB/plots/qc_holdout_final/`.

### 5.4 Clinical zero-shot (SPARE train → Varian / Elekta)

Experiment 6 Normal: **128³ full volume**, linear phase, same CRB nets as E1.

Cohort summary ([`ClinicalExperiments/Experiment6/Normal/README.md`](ClinicalExperiments/Experiment6/Normal/README.md)):

| Arch | Norm | Varian cos | Elekta cos |
|------|------|------------|------------|
| Both | minmax | 0.547 | 0.344 |
| Both | **µ** | **0.584** | **0.478** |
| Decoder | µ | 0.588 | 0.474 |

Example per-scan gain (µ, Both): **CV_P3** cos **0.671** (E6) vs **0.623** (E1) — [`compare_e1_e6.tsv`](ClinicalExperiments/Experiment6/Normal/plots/qc_summary/compare_e1_e6.tsv).

The model applies a **learned phase→deformation prior** on unseen vendor CT, not a single memorised warp.

### 5.5 What the CNN does *not* clearly beat — population atlas

From [`ChangesNeeded.md`](ChangesNeeded.md) §1:

A **no-learning leave-one-out atlas** (mean DVF of the other 8 patients on the shared grid, indexed by phase) reaches **cos 0.77–0.92** on P3/P4/P5 for pair 01→06, versus **0.69–0.83** for the E1 Decoder on the same patients.

**Interpretation:** much of apparent “understanding breathing” is a **phase-modulated population mean field**, not patient-specific biomechanics. The 1M-parameter CNN does not clearly beat an average.

### 5.6 Amplitude is the main failure mode (H1)

The network cannot observe **how much** a new patient breathes — only phase indices.

Define **α** = (mean motion of training pool) ÷ (mean motion of hold-out patient). α ranks hold-out patients in **exact** order of beat-zero rate (rank correlation **−1**).

| Hold-out | Split | α | Beat zero | Behaviour |
|----------|-------|---|-----------|-----------|
| P4 | E1 | 0.60 | **100%** | under-predicts (train moves less than P4) |
| P3 | E1 | 0.71 | **100%** | under-predicts |
| P5 | E1 | 1.11 | 82% | slight over-predict |
| P9 | E2 | 1.47 | 66% | over-predict |
| P7 | E2 | 1.78 | **47%** | strong over-predict → pred ≈ zero |

So CRB + phase gets **shape/direction**; **magnitude** regresses to the training-pool mean under MSE. Planned fix: amplitude conditioning — [`AmplitudeConditioning.md`](AmplitudeConditioning.md), [`CRBExperiments/`](CRBExperiments/).

### 5.7 DIR zero-shot (domain shift)

DIR-Lab benchmark: E6 Both µ, T00→T50, landmark TRE ([`DIR-Experiments/benchmark/results/tre_benchmark_packed.tsv`](DIR-Experiments/benchmark/results/tre_benchmark_packed.tsv)).

Cohort mean TRE (native mm) is **~8.5 mm** for identity; E6 Both µ is **~8.9–10.2 mm** — **does not beat identity** on average.

Phase + CRB prior trained on SPARE coronal HU patches does **not** transfer cleanly to DIR geometry/intensity (µ, axial packing, different anatomy).

---

## 6. Summary table

| Question | Answer | Primary evidence |
|----------|--------|------------------|
| Does the model use phase to select motion? | **Yes** | Dan 2.0 cos ~0.95; E1 94% beat-zero |
| Does CRB help vs unconditioned conv? | **Yes (placement matters)** | E1 Decoder &gt; Encoder; Both ≠ best hold-out |
| Does it understand breathing *physics*? | **No** | Supervised Elastix; atlas nearly as good |
| Generalises to new SPARE patients? | **Partly** | P3/P4 yes; P5/E2 low-motion weak |
| Generalises clinical / DIR? | **Weak / mixed** | Clinical cos ~0.5–0.6; DIR ≈ identity TRE |

---

## 7. Where to look in the repo

| Topic | Path |
|-------|------|
| CRB network code | `InitialExperiments/Experiment1/networks/generator_crb*.py` |
| E1 population results | [`results.md`](results.md) |
| Limitations & atlas | [`ChangesNeeded.md`](ChangesNeeded.md) |
| Amplitude plan | [`AmplitudeConditioning.md`](AmplitudeConditioning.md) |
| E1 hold-out QC | `InitialExperiments/Experiment1/*/plots/qc_holdout_final/` |
| E6 clinical QC | `ClinicalExperiments/Experiment6/Normal/plots/qc_summary/` |
| E1 vs E6 comparison | `ClinicalExperiments/Experiment6/Normal/plots/qc_summary/compare_e1_e6.tsv` |
| DIR TRE benchmark | `DIR-Experiments/benchmark/results/tre_benchmark_packed*.tsv` |
| Dan 2.0 baseline | `Dan2.0/results.md` |

---

## 8. Honest bottom line

**CRB is how a U-Net selects different deformation fields from the same lung appearance** by conditioning convolutional features on `(t_ref, t_tgt)`. That is “understanding breathing” **in this codebase**: supervised imitation of phase-indexed motion, not explicit respiratory modelling.

- **Strong proof:** within-patient phase generalisation (Dan 2.0).  
- **Moderate proof:** across SPARE patients when motion amplitude is similar to the training pool (E1 P3/P4).  
- **Limited proof:** clinical zero-shot (direction OK, amplitude and domain gap remain).  
- **Negative proof:** DIR zero-shot; population atlas without learning.

The gap is mostly **amplitude and anatomy/domain**, not “CRB cannot represent phase.” Next experiments (isotropic grid, amplitude conditioning, measured surrogates) target exactly that — see [`ChangesNeeded.md`](ChangesNeeded.md).
