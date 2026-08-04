# Voxel-GAN Capstone Plan — Formal Spec

Greenfield repo. Scope: **patient-specific** 4DCT motion model (intra-patient leave-phase-out), not inter-patient generalization unless DIR-Lab is added later.

---

## 1. Objective & Framing

| Item | Decision |
|------|----------|
| Task | Predict a stationary velocity field (SVF) → integrate to DVF, conditioned on (ref phase, target phase) and a reference CT |
| Data | Single patient, 10 respiratory phases |
| Claim | Intra-patient generalization over **unseen phase pairs** (leave-phase-out) |
| Non-claim | Inter-patient generalization (unless DIR-Lab is folded in) |

---

## 2. Network Architecture

### 2.1 Input / Output

| | Spec |
|--|------|
| Input anatomy | Single-channel CT, training shape **64³ or 96³** patches (prefer patches over full 128³ for memory + sample count) |
| Conditioning | Discrete phases φ ∈ {0…9}: reference + target |
| Output | 3-channel DVF via SVF + scaling-and-squaring (5–7 steps) |

Full-volume 128³ remains the design target for inference if memory allows; training uses patches.

### 2.2 Encoder (unconditioned — no FiLM)

Anatomy-only path: encode “what is the anatomy.”

| Stage | Resolution | Channels | Op |
|-------|------------|----------|-----|
| Stem | \(R^3\) | 16 | Conv3×3×3 → IN → LeakyReLU (×2) |
| Enc1 | \(R^3\)→\(R/2\) | 16→32 | Strided conv (s=2) + conv block |
| Enc2 | →\(R/4\) | 32→64 | Strided conv + conv block |
| Enc3 | →\(R/8\) | 64→128 | Strided conv + conv block |
| Enc4 | →\(R/16\) | 128→256 | Strided conv + conv block |

**Bottleneck:** \(R/16\)³ × 256, two conv blocks, **FiLM injected here** (coarsest / most global conditioning — e.g. lung-volume change).

### 2.3 Decoder (FiLM at every scale)

Each block: **upsample → concat skip → conv block → FiLM → conv block**.

FiLM **after** skip fusion so phase modulates fused anatomy + context, not only the upsampled path.

| Stage | Resolution | Channels | Skip | FiLM |
|-------|------------|----------|------|------|
| Dec4 | \(R/16\)→\(R/8\) | 256→128 | Enc3 | yes |
| Dec3 | →\(R/4\) | 128→64 | Enc2 | yes |
| Dec2 | →\(R/2\) | 64→32 | Enc1 | yes |
| Dec1 | →\(R\) | 32→16 | Stem | yes |

### 2.4 Output head

```
Dec1 (16 ch) → 1×1×1 conv → 3 ch SVF
             → scaling-and-squaring (N=5..7)
             → DVF (3 × R³)
```

### 2.5 FiLM conditioning subnetwork (spatial-free)

1. **Phase embedding:** learned embedding table (size 10) preferred over sinusoidal — phases are discrete.
2. **Context input:** `[e_ref, e_target, e_target − e_ref]` (direction + magnitude of phase change).
3. **Shared trunk MLP:** concat → 64 → 128 → 64 (LeakyReLU) → context vector `c`.
4. **Per-scale FiLM heads:** `c → (2 × C_level)` → split `(Δγ, β)`; use  
   `γ = 1 + tanh(Δγ)` (residual around identity) for early-training stability.
5. **Apply:** broadcast over spatial dims: `x' = γ ⊙ x + β`.

Capacity bias: FiLM stays small (≲ few ×10⁵ params); capacity lives in the 3D U-Net.

### 2.6 Discriminator (PatchGAN, training-only)

| Item | Spec |
|------|------|
| Input | Prefer **5-ch:** `[ref CT, target CT, DVF]` (or 4-ch `[ref, DVF]` minimum) |
| Arch | Small 3D CNN, strided convs, **no** U-Net / skips |
| Output | Low-res real/fake logit grid (local receptive fields) |
| Role | Plausibility of DVF **for this anatomy**, not generic smoothness |

---

## 3. Data Pipeline

### 3.1 Pairing (highest leverage)

| Pair type | Count | DVF |
|-----------|-------|-----|
| Directed pairs \(i \to j\), \(i \neq j\) | 90 | Elastix DVF |
| Identity \(i \to i\) | 10 | Zero field |

**Do not** train only “phase 1 as reference.” Generator must see arbitrary (ref, target) combinations at train time.

### 3.2 Elastix QC (gate before training)

For each of 90 registrations:

1. Visual: checkerboard / overlay (warped ref vs target)
2. Jacobian: reject / flag negative-det regions (folding)
3. Forward–backward: \(T_{i\to j} \circ T_{j\to i} \approx \mathrm{Id}\)
4. Optional TRE if landmarks available (DIR-Lab-style)

Elastix error → network target error; QC is mandatory.

### 3.3 Preprocessing (per volume, before reg + train)

1. Resample all phases to common **isotropic** spacing (register in physical space)
2. Clip HU ≈ [−1000, 200], normalize
3. Lung mask (similarity loss restriction + patch sampling bias)
4. Crop to **union** lung bounding box across 10 phases, then resize / patch

### 3.4 Patch extraction

| Choice | Rationale |
|--------|-----------|
| Patch size 64³ or 96³ | Memory + multiplies samples from 90 pairs |
| Overlapping / sliding window | Thousands of patch tuples |
| Motion-biased sampling | Prefer diaphragm, lower lobes, pleural sliding; avoid apex oversampling |
| Alignment | Identical crop on ref CT, target CT, DVF |

### 3.5 Splits — leave-phase-out (not random pairs)

- Hold out 1–2 **entire phases**; exclude any pair involving them from train.
- Test only on pairs that involve held-out phases.
- Report **k-fold leave-phase-out** (rotate held-out phase) for capstone robustness.
- Avoid random pair splits (anatomy leakage across phases).

---

## 4. Training Loop

### 4.1 Per-iteration protocol

1. Sample patch: `(ref CT, target CT, φ_ref, φ_tgt, Elastix DVF)`
2. **G** forward → fake DVF
3. **D** update only: real `[CT…, real DVF]` vs fake `[CT…, fake.detach()]`
4. **G** update only: adversarial (non-detached) + supervised losses  
   Separate Adam optimizers; no weight sharing.

### 4.2 Loss stack (generator)

| Term | Role | Suggested weight |
|------|------|------------------|
| Supervised DVF (e.g. L1/L2 vs Elastix) | Primary signal | high |
| Image similarity (warped ref vs target), **lung-masked** | Anatomy consistency | medium–high |
| Smoothness / diffusion regularizer on DVF or SVF | Physical plausibility | medium |
| Adversarial | Light regularizer | **λ_adv ≈ 0.01–0.1** |

### 4.3 Low-data stability

1. Supervised **warm-up** (several epochs) before enabling D
2. Update D less often than G (e.g. 1 D step / 2–3 G steps)
3. Keep λ_adv small so D cannot dominate

### 4.4 Memory / compute defaults

- AMP (mixed precision)
- Gradient checkpointing if needed
- Batch size 1–2 on patches; avoid full 128³ train unless hardware supports it

---

## 5. Evaluation

| Metric | Use |
|--------|-----|
| DVF error vs Elastix (held-out pairs) | Direct supervised score |
| Landmark TRE (if available) | Independent of Elastix |
| Jacobian stats (% folding) | Physical validity |
| Image similarity after warp (lung ROI) | Downstream usefulness |
| Identity pairs → near-zero DVF | Sanity check |
| Leave-phase-out k-fold summary | Capstone headline number |

Framing sentence for the report: *patient-specific respiratory motion model evaluated under leave-phase-out; not inter-patient generalization.*

---

## 6. Suggested Module Layout (implementation order)

```
Voxel_GAN/
  data/           # preprocess, Elastix I/O, pairing, patches, splits
  models/         # encoder/decoder, FiLM, SVF integrate, discriminator
  losses/         # DVF, NCC/MSE masked, smoothness, adversarial
  train/          # G/D loop, warm-up schedule, checkpointing
  eval/           # leave-phase-out, Jacobian, TRE hooks
  configs/        # patch size, λs, phases held out
```

**Build order**

1. Preprocess + Elastix 90+10 pairs + QC filters  
2. Patch dataset + leave-phase-out sampler  
3. Generator (U-Net + FiLM + scaling-and-squaring) — supervised only  
4. Losses + train loop (no D)  
5. Discriminator + adv warm-up schedule  
6. Eval suite + k-fold reporting  

---

## 7. Open decisions (defaults recommended)

| Decision | Default |
|----------|---------|
| Train resolution | **96³** patches if VRAM allows, else **64³** |
| Upsample | Trilinear + conv (safer than transposed alone) |
| Phase embed dim | 8–16 per phase; table of size 10 |
| Scaling-and-squaring steps | **7** |
| D input channels | **5** (ref + target + DVF) |
| Held-out phases (first fold) | e.g. mid-inspiration + mid-expiration (5 and 9) |
| DIR-Lab | Optional extension for TRE + inter-patient claim |

---

## 8. Success criteria (capstone)

- Stable train: identity pairs ≈ 0; no widespread negative Jacobians on held-out pairs  
- Leave-phase-out DVF / similarity competitive with Elastix self-consistency baseline  
- Ablations reportable: no-FiLM difference embedding; no adv; whole-volume vs patch; with/without motion-biased sampling  
- Explicit patient-specific framing in write-up  
