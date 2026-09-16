# DVF sign convention: pull (G160) vs push (Elastix / DIR TRE)

**Why this note exists:** A3 synth DVF TRE looked *worse than identity* (~12 mm)
until the field was negated (~6 mm). That was not a free TRE knob — it is a
**language mismatch** between two displacement conventions in this repo.

| Role | Symbol | Meaning of the arrow at voxel `x` |
|------|--------|-----------------------------------|
| **Pull** (G160 `warp`) | `u` | “To fill output `x`, **sample** the input at `x + u(x)`.” |
| **Push** (Elastix / ITK) | `d` | “Tissue at fixed `x` **moves to** `x + d(x)`.” |

For the same breathing motion, those arrows point **roughly opposite**:

\[
\mathbf{d}_{\mathrm{Elastix}} \approx -\mathbf{u}_{\mathrm{G160}}
\]

(small-strain / first-order; exact DF inverse ≠ exactly `-d`).

---

## 1. G160 gives a **pull** field

Warping is PyTorch `grid_sample` on `identity + flow`:

```python
# PopulationStudy/ClinicalExperiments/Grid160/Experiment1/utilities/warp.py

def _flow_to_grid(flow):
    ..., d, h, w = flow.shape
    grid = _meshgrid((d, h, w), flow.device, flow.dtype)  # identity in [-1, 1]
    scale = torch.tensor(
        [2.0 / max(w - 1, 1), 2.0 / max(h - 1, 1), 2.0 / max(d - 1, 1)],
        ...
    ).view(1, 3, 1, 1, 1)
    disp = flow * scale
    sample_grid = (grid + disp).permute(0, 2, 3, 4, 1)  # ← sample at x + u
    return sample_grid.expand(flow.size(0), -1, -1, -1, -1)


def warp(vol, flow):
    """Warp a (B, C, D, H, W) volume by voxel-space flow (B, 3, D, H, W)."""
    grid = _flow_to_grid(flow)
    return F.grid_sample(
        vol, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
```

In words:

\[
I_{\mathrm{tgt}}(\mathbf{x}) = I_{06}(\mathbf{x} + \mathbf{u}(\mathbf{x}))
\]

So `u` is a **sampling / pull** field on the **output** grid.  
**Synth CT QA uses `+u` as-is** — warper and field speak the same language.

---

## 2. Elastix / DIR TRE expect a **push**-style field

A1 Elastix: `fixed = sub_CT_06` (T50), `moving =` earlier phase (e.g. T00).  
ITK / Transformix displacement:

\[
\mathbf{x}_{\mathrm{moving}} \approx \mathbf{x}_{\mathrm{fixed}} + \mathbf{d}(\mathbf{x}_{\mathrm{fixed}})
\]

The TRE harness documents that explicitly:

```python
# DIR EXPERIMENTS/scripts/eval_a1_tre.py (header)

# Convention (Elastix fixed=T50=phase06, moving=T00=phase01; LEARN unit-voxel DVF):
#   ITK: moving ≈ fixed + disp(fixed)
#   T00→T50: pred = lm00_pack + (-disp) sampled at lm00
#   T50→T00: pred = lm50_pack + (+disp) sampled at lm50
```

Low-level landmark displace helper (push form `pred = src + dvf(src)`):

```python
# DIR EXPERIMENTS/scripts/dirlab_tre.py

def evaluate_dvf(case, dvf, src_phase="T00", dst_phase="T50", which="75"):
    """TRE after displacing the source landmarks by a dense field.

    The field must map SOURCE positions to TARGET positions directly:
    pred = src + dvf(src). If your field is defined the other way round (the
    warp that pulls the target back to the source), invert it before calling
    this -- getting that backwards typically inflates TRE by ~1-2 mm on the
    large-motion cases, which looks like a mediocre result rather than a bug.
    """
    ...
    pred = src + sample_dvf(dvf, src)
    return stats(tre_mm(pred, dst, spacing)), stats(tre_mm(src, dst, spacing))
```

---

## 3. Where we apply the **minus**

`prepare_a3` writes VoxelMap / TRE labels in **Elastix convention** by default
(`DVF_sub = -u`). Use `pull` only if you intentionally keep the raw G160 field.

```python
# DIR EXPERIMENTS/scripts/prepare_a3_dir_case.py — write_synth_dvfs()

def write_synth_dvfs(..., convention: str = "elastix") -> int:
    """Write DVF_sub_*.mha labels for VoxelMap / TRE.

    G160 ``warp`` uses a pull field: I_tgt(x) = I_06(x + u(x)).
    A1 Elastix / eval_a1_tre expect fixed→moving d with x_mov ≈ x_fix + d(x_fix).
    First-order: d ≈ -u. Default ``convention='elastix'`` writes -u.
    Use ``convention='pull'`` only to keep raw G160 sampling vectors.
    """
    ...
        if convention == "elastix":
            dvf_zyx = -dvf_zyx
        out = train / f"DVF_sub_{phase:02d}.mha"
        write_vector_dvf_mha(dvf_zyx, ref, out)
```

`synth_meta.json` records which convention was written:

```json
"dvf_label_convention": "elastix",
"dvf_label_note": "elastix: DVF_sub = -u (G160 pull→ITK fixed→moving); pull: DVF_sub = u as used by warp(I_06, u)"
```

---

## 4. Empirical check (not optional folklore)

On R3 DIR C1–C10 (75-pt T00→T50), log  
`arms/A3_synth_conditioned/logs/signflip_all_cases.log`:

| Field used in TRE path | Cohort mean TRE (mm) | vs identity 8.69 |
|------------------------|---------------------:|-----------------:|
| raw synth `u` (as-is)  | **12.06**           | worse            |
| **`-u`**               | **6.13**            | better (10/10)   |
| A1 Elastix             | **2.08**            | teacher          |

SI component: `corr(Elastix, raw)` ≈ **−0.83**, `corr(Elastix, −synth)` ≈ **+0.83**.

Re-run the table:

```bash
cd "DIR EXPERIMENTS"
# see scripts/eval_a3_synth_vs_elastix.py  (--r3, as-is vs negate)
```

---

## 5. What we do **not** negate

| Action | Field |
|--------|--------|
| Warp synth CT with G160 `warp` | keep **`+u`** |
| Score landmarks / compare to Elastix like A1 | use **`-u`** (or baked `DVF_sub`) |
| Train VoxelMap against A1-style labels | labels should be **Elastix convention** |

Varian / Elekta clinical CRB work usually trained or compared **in one convention**
(model ↔ Elastix) or warped with the **model’s own** `warp(u)`. Mixing G160 pull
`u` into the DIR Elastix TRE recipe is what surfaces the minus.

---

## 6. One-line rule

> **Same motion, opposite arrow language.**  
> Warp CT with G160 → `+u`. Talk to Elastix / DIR TRE → `-u`.

Do **not** treat the sign as a free hyperparameter to chase TRE. Bake Elastix
convention into `prepare_a3` (`--dvf-convention elastix`, the default) and keep
raw pull fields only when you are calling `warp`.
