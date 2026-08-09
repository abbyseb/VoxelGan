# Loss Function (in detail)

Training minimizes a **generator loss** (main) and separately a **discriminator loss**. Same recipe for Phase-MLP, Warp-D, and Dan (only D’s *inputs* change).

Code: `utilities/losses.py` · wiring: `utilities/train_loop.py` (and Dan’s `train_crb_gan.py`).

---

## Generator objective

\[
\mathcal{L}_G =
\underbrace{\mathcal{L}_{\mathrm{DVF}}}_{\text{match Elastix}}
+ \lambda_{\mathrm{img}}\,\underbrace{\mathcal{L}_{\mathrm{img}}}_{\text{warp looks like target}}
+ \lambda_{\mathrm{smooth}}\,\underbrace{\mathcal{L}_{\mathrm{smooth}}}_{\text{no wild fields}}
+ \lambda_{\mathrm{adv}}\,\underbrace{\mathcal{L}_{\mathrm{adv}}}_{\text{fool D (after warmup)}}
\]

| Weight | Value | Role |
|--------|-------|------|
| \(\lambda_{\mathrm{img}}\) | **0.5** | image similarity |
| \(\lambda_{\mathrm{smooth}}\) | **0.1** | smoothness |
| \(\lambda_{\mathrm{adv}}\) | **0.05** | adversarial (small) |

Adv starts after **warmup** (epoch > 5). Val uses only the first three terms (no GAN).

---

## 1. \(\mathcal{L}_{\mathrm{DVF}}\) — lung-masked L1 on the field

**What:** predicted DVF vs Elastix DVF, only inside lung.

\[
\mathcal{L}_{\mathrm{DVF}}
= \frac{\sum |u_{\mathrm{pred}} - u_{\mathrm{Elastix}}| \cdot m}{\sum m}
\]

- \(u\): 3 channels \((u_x,u_y,u_z)\)
- \(m\): lung mask broadcast over 3 channels
- Outside lung ignored (ribs/chest-wall Elastix junk won’t dominate)

**Why:** main supervised signal — “same motion as registration.”

---

## 2. \(\mathcal{L}_{\mathrm{img}}\) — warp the CT, match target (NCC)

**What:** warp reference CT with predicted DVF, compare to target CT in lung.

\[
\text{warped} = \mathrm{warp}(\mathrm{CT}_{\mathrm{ref}},\, u_{\mathrm{pred}})
\]

Default metric = **masked global NCC**, returned as **\(1 - \mathrm{NCC}\)** (lower better):

\[
\mathcal{L}_{\mathrm{img}} = 1 - \mathrm{NCC}_{\mathrm{lung}}(\text{warped},\, \mathrm{CT}_{\mathrm{tgt}})
\]

(NCC = correlation of intensities after subtracting means, only where mask=1.)

**Why:** even if DVF isn’t identical to Elastix voxel-wise, the **warped image** should look like the target. Softens pure vector matching.

---

## 3. \(\mathcal{L}_{\mathrm{smooth}}\) — diffusion / TV-ish on DVF

**What:** penalize squared jumps between neighbor voxels (finite differences):

\[
\mathcal{L}_{\mathrm{smooth}}
= \frac{1}{3}\Big(
\mathbb{E}[(\Delta_x u)^2] +
\mathbb{E}[(\Delta_y u)^2] +
\mathbb{E}[(\Delta_z u)^2]
\Big)
\]

**Why:** discourage folds / noisy fields. Dan relies on this more (no SVF); FiLM+SVF still uses it.

---

## 4. Adversarial — LSGAN on PatchGAN

### Discriminator loss (D step)

Two forwards:

- **Real** → push scores toward **0.9** (label smoothing)
- **Fake** (G detached) → push scores toward **0.0**

\[
\mathcal{L}_D = \tfrac{1}{2}\Big(
\mathrm{MSE}(D(\mathrm{real}),\, 0.9)
+ \mathrm{MSE}(D(\mathrm{fake}),\, 0)
\Big)
\]

### Generator adversarial term (G step)

\[
\mathcal{L}_{\mathrm{adv}} = \mathrm{MSE}\big(D(\mathrm{fake}),\, 1\big)
\]

G wants D to call fake “real.”

### What is fed to D

| Mode | Real | Fake |
|------|------|------|
| **DVF (Phase-MLP)** | `[ref CT, Elastix DVF]` 4-ch | `[ref CT, pred DVF]` 4-ch |
| **Warp-D** | `[warp(ref,Elastix), target]` 2-ch | `[warp(ref,pred), target]` 2-ch |

D updated every **2** G steps (`d_update_freq=2`).

---

## One training step (mental order)

```text
1. G predicts DVF from (ref CT, φ_ref, φ_tgt)
2. Sometimes: update D with real vs fake (LSGAN)
3. Update G with:
      L_DVF + 0.5·L_img + 0.1·L_smooth  [+ 0.05·L_adv if past warmup]
```

---

## Who does what (roles)

| Loss | Teaches G to… |
|------|----------------|
| \(\mathcal{L}_{\mathrm{DVF}}\) | Match Elastix vectors in lung |
| \(\mathcal{L}_{\mathrm{img}}\) | Make warped CT look like target |
| \(\mathcal{L}_{\mathrm{smooth}}\) | Keep field spatially smooth |
| \(\mathcal{L}_{\mathrm{adv}}\) | Fool D (local realism PatchGAN cares about) |

**Main driver = \(\mathcal{L}_{\mathrm{DVF}}\)**; image + smooth regularize; adv is a light regularizer (`λ=0.05`), not the primary objective — that’s why Warp-D can stay alive without wrecking accuracy.
