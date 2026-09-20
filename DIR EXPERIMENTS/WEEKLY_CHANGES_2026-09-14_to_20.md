# DIR EXPERIMENTS — what changed (14–20 Sep 2026)

Short narrative of the two big fixes (**DRR orbit**, **DVF sign**), **why** each was needed, why SPARE A3 is **bounded by the SPARE oracle**, and how **TCIA2** starts to break that bound on hard cases.

Detail and plots: `diary.md`, `docs/DVF_pull_vs_push_sign.md`.

---

## 1. Problem before the fixes

We had two separate issues that both made A3 (and early A1 VoxelMap) look worse than they were:

| Issue | Symptom | What was wrong |
|-------|---------|----------------|
| **Bad DRR orbit** | DIR DRRs didn’t match SPARE chest↔spine orbit; anatomy swung off-detector | DIR SI was on **itk Z**, volumes **corner-origin**; RTK orbits **world Y** around isocentre. SPARE has SI on Y and centred volumes. |
| **Wrong DVF sign for A3** | Synth CT looked fine, but TRE ≫ identity | G160 outputs a **pull** field \(I(x+u)\); Elastix / our TRE path expects **fixed→moving** \(\approx -u\). |

Old wrong-orbit runs were moved to `arms/Incorrect DRR/` (do not use for reporting).

---

## 2. Why the DRR was wrong

RTK / SPARE circular DRR always orbits **world Y** (patient SI should be that axis), around the **volume isocentre**.

What DIR had instead:

1. **SI axis mismatch** — SPARE GTVol puts SI on **itk Y** (thin axis). Native DIR packs put SI on **itk Z**. So with SPARE’s `Geometry.xml` unchanged, the gantry was spinning around the wrong patient axis → chest↔spine orbit became a mess / anatomy left the detector.
2. **Origin mismatch** — SPARE volumes are **centred** on isocentre. DIR used `origin=(0,0,0)` (corner). RTK still orbits world origin → the orbit centre sat near a corner of the box, roughly **outside** the patient.
3. **Permute alone is not enough** — mapping SI→Y without recentre still fails. After permute we also needed **AP flip** and **SI flip** so the patient is upright and matches SPARE’s chest↔spine video (**R3**).

**Important:** we did **not** “fix” this by editing projection angles in `Geometry.xml`. The XML (SPARE, OffsetY **−2**) was already correct for SPARE-shaped volumes. DIR volumes had to be brought into that frame.

**Why VoxelMap cared more than Elastix TRE:** Elastix TRE is scored on **volume landmarks** (can look fine even if DRRs are wrong). VoxelMap learns motion from **projections**. Wrong orbit ⇒ the network never saw the right 2D↔3D relationship ⇒ inflated A1 VM TRE until R3 re-prep + retrain.

---

## 3. Fix A — DRR / geometry (R3), ~14–16 Sep

**Locked recipe (R3):**
- Reorient DIR CT: SI→Y + AP flip + SI flip, centre on isocentre  
  (`arr.transpose(1,0,2)[::-1,::-1,:]`, origin = −(size−1)×spacing/2).
- Keep **`Geometry_SPARE.xml`** (OffsetY **−2**). Do **not** edit orbit matrices.
- Bake into `prepare_a1_case.py`; TRE with `eval_a1_tre.py --r3`.

**How TRE changed (after full A1 re-prep + retrain):**

| Arm | Before (wrong orbit) | After R3 | Notes |
|-----|---------------------:|---------:|-------|
| A1 Elastix | ~2.08 mm | **~2.08 mm** | Volume landmarks ≈ unchanged |
| A1 VoxelMap | ~3.2 mm | **~2.32 mm** | Large gain — projections finally match anatomy |
| A2 zero-shot | slightly **worse** than identity | **beats identity by ~1.4 mm** (mean TRE ~7.3) | Still far from A1 |

Per-case R3 A1 VM: C01 1.40 … C08 4.48; cohort mean **2.32 ± 1.01** mm (TRE75 T00→T50).

---

## 4. Why the negative (−u) is needed

Two **different displacement conventions** were being mixed:

### Elastix / ITK (what A1 labels and our TRE harness use)

- Fixed = mid-exhale T50 (`sub_CT_06`), moving = inhale phase (e.g. T00).
- Transformix DF: \(x_{\mathrm{moving}} \approx x_{\mathrm{fixed}} + d(x_{\mathrm{fixed}})\).
- So \(d\) is **fixed → moving** on the fixed grid (T50→T00).
- TRE: T00→T50 uses `pred = lm00 − disp` with that DF.

### G160 synthesizer (what `warp` actually does)

- \(I_{\mathrm{tgt}}(x) = I_{06}(x + u(x))\) via `grid_sample` (`identity + flow`).
- \(u\) is a **pull / sampling** field on the **output** grid: “where to read in the reference.”
- That is **tgt → ref** (~ opposite transport sense to Elastix \(d\)).

**First-order inverse:** \(d_{\mathrm{Elastix}} \approx -u_{\mathrm{synth}}\).

Empirically: corr(Elastix SI, raw \(u\)) ≈ **−0.83**; corr(Elastix SI, \(-\,u\)) ≈ **+0.83**.

### Why CT QA does not catch this

Warping CT with raw \(u\) is **internally consistent** — synth CT can look excellent. That only proves the warp matches the field the synthesizer was trained with. It does **not** prove the field is in the same convention as Elastix labels / TRE.

Feeding raw \(u\) into the Elastix TRE / VoxelMap-label path is the **wrong sign** → TRE worse than identity (cohort **12.1** mm vs id **8.7**). Negating fixes the convention (cohort **6.1** mm), not a free “TRE knob.”

Bake **`DVF_sub = -u`** in `prepare_a3_dir_case.py` (`--dvf-convention elastix`, default). See `docs/DVF_pull_vs_push_sign.md`.

---

## 5. Fix B — DVF sign baked in, ~16 Sep

Smoke (synth DVF vs Elastix TRE, no VoxelMap yet):

| Convention | Cohort TRE75 mean |
|------------|------------------:|
| Raw synth \(u\) (as-is) | **12.06 mm** (worse than identity 8.69) |
| **\(-\,u\)** (Elastix convention) | **6.13 mm** (beats identity 10/10) |
| A1 Elastix | 2.08 mm |

Then A3 full VoxelMap train (SPARE G160 teacher + correct R3 DRRs + \(-\,u\) labels):

| Case | A3 VoxelMap TRE75 (SPARE G160) |
|-----:|-------------------------------:|
| C01 | 2.16 |
| C02 | 2.75 |
| C03 | 3.91 |
| C04 | 7.30 |
| C05 | 4.51 |
| C06 | 8.94 |
| C07 | 9.33 |
| C08 | 13.61 |
| C09 | 6.18 |
| C10 | 6.16 |
| **mean** | **6.49 ± 3.48** |

---

## 6. Why SPARE A3 is bounded by the SPARE oracle — and why TCIA

### The bound

A3 VoxelMap is trained on **synthetic 4D** whose motion teacher is the **G160 synthesizer**.  
If that teacher’s DIR oracle TRE (synth DVF scored like Elastix, no VoxelMap) is already **~6.1 mm**, VoxelMap cannot magically invent better motion than it was shown. Empirically:

| Stage | Cohort TRE75 |
|-------|-------------:|
| SPARE G160 **oracle** (synth alone) | **~6.13 mm** |
| A3 VoxelMap taught by SPARE G160 | **~6.49 mm** |

A3 sits **at / slightly above** the SPARE oracle — as expected for a student of that teacher (plus DRR/train noise). Easy cases (C01 ~2.2) are already near the teacher; **hard cases** (C08 ~13–14) are where the teacher under-moves / misses DIR-like amplitude.

P3 semi-oracle (synth DRRs + **real A1 Elastix labels**) already showed C01 ~2.5 mm: **appearance was OK**; the **motion teacher** was the ceiling.

### Breaking the bound → better teacher (TCIA2)

To push A3 **below** the SPARE oracle, the synthesizer itself must improve on DIR (especially large-motion patients). That is what **TCIA2** (4D-Lung, R3+µ, A1 FOV Decoder) is for.

**Synthesizer alone (DIR oracle TRE75):**

| Model | Mean ± SD | C01 | C08 (tough) |
|-------|----------:|----:|------------:|
| SPARE G160-A1 | 6.13 ± 3.15 | 2.15 | **13.04** |
| **TCIA2 best** (final val) | **5.17 ± 2.51** | 2.13 | **10.07** |
| TCIA2 ep100 latest | 5.28 ± 2.61 | 2.35 | 10.61 |

TCIA2 improves the **cohort** (~1 mm) and, more importantly, **cuts the hard-case oracle** (C08 **13.0 → 10.1**). Soft cases were already near Elastix; the gap was in large SI motion.

**A3 VoxelMap with that teacher** (student follows the new ceiling):

| Case | SPARE G160 A3 | TCIA2 A3 | Δ |
|-----:|--------------:|---------:|--:|
| C01 (easy) | 2.16 | **1.97** | −0.19 |
| C08 (tough) | 13.61 | **11.45** | **−2.16** |
| C02–C07, C09–C10 | SPARE done | TCIA2 pipeline in progress | |

So: SPARE A3 ≈ SPARE oracle bound; TCIA2 lowers the oracle, and A3 on tough cases moves with it (C08 ~2 mm better so far). Remaining gap to A1 Elastix / A1 VM is still synthesizer quality + domain, not DRR/sign bugs.

---

## 7. Tooling added alongside

- **TRE viewer** (`tools/tre_viewer`): napari overlays, DRR/RTK pages, Phase Performance (T50→phase).
- Merged to `main` / `TRE-VIZ` (PR #2 verify/UI, PR #3 phase graphs).
- Defaults: green truth / red cross pred; TRE rings off; stronger error arrows.

---

## 8. One-line takeaway

1. **DRR was wrong** because DIR SI/origin didn’t match SPARE’s RTK frame — VoxelMap (projections) broke; Elastix volume TRE hid it.  
2. **\(-\,u\) is required** because G160 \(u\) is pull and Elastix/TRE \(d\) is fixed→moving — CT QA can’t see the mismatch.  
3. **SPARE A3 ≈ SPARE oracle (~6 mm)**; to go lower need a better teacher → **TCIA2**, which mainly helps **tough cases** (C08 oracle 13→10; A3 13.6→11.5).

---

## Where to look

| Topic | Path |
|-------|------|
| Full diary | `DIR EXPERIMENTS/diary.md` |
| DVF pull vs push | `DIR EXPERIMENTS/docs/DVF_pull_vs_push_sign.md` |
| Older Results snapshot | `DIR EXPERIMENTS/Results.md` (pre-R3 / partial) |
| Incorrect-orbit archive | `arms/Incorrect DRR/` |
| A1/A2 R3 cohort | `arms/*/results/cohort_tre_r3_final.json` |
| A3 SPARE runs | `arms/A3_synth_conditioned/runs/DIR_C0N/` |
| A3 TCIA2 runs | `arms/A3_synth_conditioned/runs/DIR_C0N_tcia2/` |
| TCIA2 oracle JSON | `…/Grid160/TCIA2/DecoderCRB/plots/qc_dir_oracle/tre75_final_best_vs_ep100_vs_spare.json` |

*Written / updated 2026-09-20.*
