# Plan: Unsupervised training (branch `Unsupervised-Training`)

## Goal
Remove the supervised DVF term (lung-masked L1 against the **Elastix** DVF labels) completely.
Train only from the images: the target CT acts as the signal, through a warp.

Today (`TCIA_lite/run_A2_full160_aug_hybrid*/scripts/train_ft.py`):

    loss = L1_dvf(elastix, pred) + 10 * L1(tgt, warp(ref_geo, pred)) + 0.1 * mean|grad pred|^2   [+ λ_lncc * LNCC]

New:

    loss = LNCC_9(tgt, warp(ref_geo, pred))  [lung mask dilated 2 vox]
         + λ_reg * bending energy (or diffusion) of pred
         (+ optional λ_img * masked L1 / MSE as a secondary image term)

Why: the model can only be as good as Elastix if Elastix is the teacher (official 1.49 mm, 1.96 mm on our scorer).
An image loss does not share Elastix's errors.

Note: the network sees only `ref CT + ref/target phase`. It never sees the target CT, so it is a
population motion model. The target CT is used **only inside the loss**. That stays the same.

## Phase 0: Setup (no training)
1. Make a NEW COPY of the hybrid run folder as `run_A2_full160_aug_unsup/`, following the existing rule
   "new copy, original untouched". Same data, net (UNetCRBDecoder 1.07M), aug, cosine lr, 40 epochs, seed 20260929.
2. Add `unsup_loss()` with `LAMBDA_DVF = 0` hard-wired. Reuse `lncc_term`, `warp`, `dilate`, `ref_geo`.
3. Add `bending_term(pred)` (2nd-order finite differences) next to the existing `smooth_term` (1st order).
4. Logging: keep **val DVF MAE vs Elastix** as a *diagnostic only*. It is no longer the target and is expected to rise.
   Add val LNCC, the % of lung voxels with det J ≤ 0, and the std of log det J.
5. Checkpoint selection must NOT use the DVF MAE. Use val LNCC, or simply take the last epoch
   (fixed rule, decided before training).

## Phase 1: Gate (stops if any check fails)
a) The identity field gives a worse loss than the Elastix label on the val pairs. This checks the loss points the right way,
   with no aug and with forced flip/shift/zoom.
b) The regulariser and LNCC both send a non-zero gradient into the net.
c) Overfit 1 pair for ~200 steps: LNCC rises, det J ≤ 0 stays < 1 %, and that pair's TRE improves vs identity.
d) Confirm DIR-Lab/POPI are not in training (no leakage). The unsupervised loss makes it tempting to add them later.

## Phase 2: Pick λ_reg (small sweep, TCIA val only, not DIR-Lab)
- λ_reg ∈ {0.3, 1, 3} for bending energy; 10 epochs each, 1 seed.
- Choose on **val LNCC subject to folding < 0.5 %**. DIR-Lab TRE stays untouched until Phase 3.
  This avoids tuning on the test set.

## Phase 3: Main arms (40 epochs, 2 seeds, same scorer)
| Arm | Init | Loss |
|---|---|---|
| A  baseline | scratch | current hybrid (existing results, rerun not needed) |
| U1 unsup-scratch | scratch | LNCC + λ_reg·bending |
| U2 unsup-warmstart | hybrid s1 ep40 | LNCC + λ_reg·bending, lr 1e-5, 10 epochs |

Score: DIR-Lab + POPI TRE300 on the fixed 2 mm scorer (`eval_dir_tcia3_iso2mm_v2`), last-5-epoch mean, 2 seeds,
plus lung NCC and folding %.
Keep rule (set now): an arm wins if TRE300 is > 0.1 mm better than A on BOTH datasets AND ≥ 7/10 DIR-Lab cases improve,
with folding < 0.5 %.

## Phase 4: Only if U1 or U2 wins
- Unsupervised training no longer needs Elastix labels, so add more TCIA 4DCT pairs (all phases, not only T00/T50).
- Try a symmetric / inverse-consistent loss and an LNCC window of 5 vs 9.

## Risks to watch
- **Folding.** The image term alone rewards non-physical fields. Watch det J.
- **Sliding at the lung boundary.** The global regulariser fights it. The mask is dilated only 2 vox; a mask-weighted
  regulariser is a possible fix.
- **HU change with breathing.** LNCC handles it better than L1. This is why L1 drops to an optional extra.
- **Weak earlier evidence.** The 5-epoch LNCC top-up gave only −0.02 mm on DIR-Lab and 0 on POPI. That was with the DVF term still present
  and a tiny budget, so U1/U2 are the real test.

## Deliverables
- `run_A2_full160_aug_unsup/` (U1), `run_A2_full160_aug_hybrid_ft_unsup/` (U2), the gate JSON, sweep JSON, score JSON
- A row in `Results.md` and one slide comparing A / U1 / U2
