# Plan: X-ray conditioned unsupervised model (branch `Unsupervised-Training`)

## Where we are (12–13 Oct 2026)
| Model | Sees | DIR-Lab TRE300 (2 mm scorer) |
|---|---|---|
| CT-only, LNCC only (`run_A2_full160_aug_unsup_lncc`, last 5 of ep 33–37) | ref CT + phase number | **4.03** (mirror 3.98), POPI 4.98 |
| CT-only, hybrid (Elastix labels) | ref CT + phase number | 4.17, POPI 4.80 |
| A2-TCIA VoxelMap (zero-shot) | two X-rays only, **never the CT** | 5.14 (C1–5 3.15, C6–10 7.13) |
| A1 VoxelMap patient oracle | X-rays, trained on that patient's own 4D | 1.95 |
| Elastix (sees full target CT) | both full CTs | ~1.5–2.0 |

The CT-only model knows the anatomy but has to **guess** the breath. The X-ray-only model sees the breath but
**not the anatomy**. Neither is enough alone → give one network both.
Bending is parked: folding is already ~0.01 % without any regulariser.

## Core idea
    ref CT (3D)  +  X-ray(s) of the target moment (2D)  →  network  →  DVF   (ref → target)
    loss = LNCC_3D( target CT , warp(ref CT, DVF) )                          (as now, training only)
         + λ_p · LNCC_2D( X-ray , DRR(warp(ref CT, DVF)) )                    (new: projection consistency)

The second term needs only the X-ray, not the target CT. So it also works **at treatment time** (see stage X5).

## Use the X-ray to the fullest: 6 ingredients
1. **Geometrically correct back-projection, not a flat broadcast.** Smear each X-ray pixel along its **real
   cone-beam ray** into the 160³ CT grid, using the projection matrices we already have
   (`tools/tre_viewer/drr.py`, RTK `Geometry.xml`, MC/Varian half-fan). Each voxel then receives the X-ray value
   of the ray that passes through it, so the anatomy and the X-ray line up in 3D. Extra input channel(s).
2. **Give the reference X-ray too, and the difference.** Back-project DRR(ref CT) at the same angle and feed
   `target X-ray − ref DRR`. The difference image is ≈0 where nothing moved and lights up at the diaphragm and
   lung edges, which is the "how much did it move" signal, already in the right place.
3. **Global breath code (FiLM).** A small 2D encoder → a vector that scales and shifts the decoder features. It carries
   "how deep is this breath" globally, which is exactly what the CT-only model is missing. Combine with (1): local + global.
4. **Angle as input, random angles in training.** Real CBCT X-rays come from any gantry angle. During training, render
   DRRs on the fly at a random angle and pass the angle (sin, cos) in. The model then works at any angle and at 1 or 2 views
   (view dropout: train with 0/1/2 views, so it degrades gracefully and the 0-view case = today's model).
5. **Projection-consistency loss.** Make a DRR of the *warped* ref CT and compare it with the input X-ray (2D LNCC).
   This forces the predicted 3D motion to explain what the X-ray shows. It needs a differentiable DRR in torch
   (ray-marching with `grid_sample`).
6. **Test-time refinement (the biggest potential win).** At treatment we have planning CT + real X-ray, not the
   target CT. Run the network, then take a few gradient steps on the network output (or a small residual)
   to maximise term (5) for *this* patient's X-ray, with the field kept smooth. This is what makes the A1
   "patient oracle" good (1.95 mm), but without needing the patient's own 4D CT.

## Stages (one change per stage; every stage scored on DIR-Lab + POPI with the same 2 mm scorer)
**X0 Plumbing (no training)**
- Torch differentiable DRR (cone-beam, same geometry as RTK, downsampled to 128²) + its back-projection.
- Gate: torch DRR vs RTK DRR of the same CT, correlation > 0.95; landmark projects onto the same pixel (±1);
  DRR(T50) − DRR(T00) shows the diaphragm band on DIR-Lab C01.
- DIR-Lab test inputs: DRR of T50 at fixed angles (0° AP, 90° lateral), made with the SAME code as training
  (the old "Incorrect DRR" orbit bug came from train/test DRR mismatch).

**X1 CT + 1 view (AP), back-projected, LNCC loss**: init from the LNCC-only checkpoint (warm start).
- Control A: same model with the X-ray channel **zeroed** → must equal today's ~4.0. Control B: **shuffled** X-ray (other
  patient/phase, = A4 mismatched conditioning) → TRE must get worse. If B ≈ X1, the net ignores the X-ray.

**X2 + difference image + FiLM breath code**

**X3 + random angle + view dropout (0/1/2 views)**: report TRE vs number of views and vs angle.

**X4 + projection-consistency loss**: λ_p ∈ {0.1, 1}, picked on TCIA val.

**X5 Test-time refinement** on DIR-Lab/POPI: 0 / 10 / 50 steps; report TRE and folding vs steps.

**X6 Realism** (if X1–X5 work): noise, scatter and the real 128² compressed detector pipeline on the test DRRs, then real
CBCT projections (SPARE), which is the A5 domain-adaptation arm.

## What we report per stage
DIR-Lab TRE300 (plain + mirror, per case, last-5-epoch mean), POPI TRE, folding %, std log|det J|, TRE vs
#views/angle, zeroed/shuffled-X-ray controls. Expected ordering: CT-only 4.0 > X1 > … > X5 → toward Elastix 1.5–2.0.

## Keep rule (fixed now)
A stage is kept if DIR-Lab last-5 TRE300 improves > 0.1 mm over the previous kept stage AND >= 7/10 cases improve
AND the shuffled-X-ray control is worse than the real X-ray by > 0.3 mm. POPI is reported but not part of the rule
(user decision, 13 Oct 2026).

## Risks
- **Train/test DRR mismatch.** Same code and geometry for TCIA and DIR-Lab DRRs, checked in X0.
- **Shortcut.** The DRR comes from the same target CT the loss uses. A 2D projection cannot give the 3D answer, but
  the shuffled control (X1) proves the net uses the X-ray, not a leak.
- **Compute.** On-the-fly cone-beam DRR on 160³ every step costs time; fall back to 128² detector and precomputed
  angle sets. One full-volume lite run at a time (RAM).
- **Depth.** One view is weak front–back → X3 (two views) and X5 (test-time) address it.
