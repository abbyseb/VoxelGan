# DVF sign scatter — how to read

Lung-masked **SI** displacement, phase 06→01 (T50→T00), ModelTraining 128³ grids.

## Axes

| Axis | Meaning |
|------|---------|
| **X** | Elastix \(u_{SI}\) [mm] — landmark / TRE convention |
| **Y** | Synth SI — raw pull \(u\), or \(-\,u\) (Elastix convention) |

## Guide lines

- Dashed **y = x** — perfect agreement with Elastix  
- Dotted **y = −x** — perfect *opposite* of Elastix (pure sign flip)

## Panels

- **Left (red) — raw \(u\)**: cloud hugs **y = −x**, Pearson **r ≈ −0.7**. Synth stores the breath the other way around.  
- **Right (green) — \(-\,u\)**: same cloud on **y = x**, **r ≈ +0.7**. Convention now matches Elastix.

Hex color = voxel count (log). Denser = more of the lung.

## What this is *not*

Not TRE. Spread off the diagonal **after** the flip is real teacher/model error (why synth oracle ~6 mm vs Elastix ~2 mm). Negation fixes **direction**, not amplitude/shape.

## Files

| File | Content |
|------|---------|
| `DIR_C01_dvf_si_elastix_vs_synth_sign_scatter.png` | Case 1 only |
| `dvf_si_elastix_vs_synth_sign_scatter.png` | C01–C10 pooled |

Regen: `python scripts/plot_dvf_sign_scatter.py` from `DIR EXPERIMENTS/`.
