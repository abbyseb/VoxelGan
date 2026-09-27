# This week's update (21–27 September 2026)

The numbers to keep are the finished TCIA3 VoxelMap runs on all 10 DIR-Lab cases. Later tries (deeper training views, a scale on the motion, a lung crop of the X-ray) did not beat them. C08 stays near 12 mm.

## Result

Inhale to exhale, real X-rays, R3. TCIA3 epoch 100 makes the motion. VoxelMap is trained on those synthetic X-rays and scored on the real ones.

75 landmarks:

| Case | TCIA3 | Do nothing |
|---|---:|---:|
| C01 | 2.13 mm | 3.91 mm |
| C02 | 1.93 mm | 4.65 mm |
| C03 | 3.38 mm | 7.25 mm |
| C04 | 5.43 mm | 9.69 mm |
| C05 | 3.70 mm | 7.41 mm |
| C06 | 6.15 mm | 11.77 mm |
| C07 | 7.35 mm | 10.71 mm |
| C08 | 11.82 mm | 16.00 mm |
| C09 | 5.21 mm | 7.16 mm |
| C10 | 5.44 mm | 8.33 mm |
| **Mean** | **5.26 mm** | |

300 landmarks: mean **5.03 mm**. C08 is **10.99 mm**.

C01 and C02 are near 2 mm. C06 and C07 are already 6–7 mm. C08 is the worst, not the only miss. A registration on the same C08 landmarks is 3.77 mm (75 points) and 3.32 mm (300 points).

## What we tried on C08

Each of these kept the TCIA3 motion as the thing being copied. None of them moved the landmark error off about 11–12 mm.

- **Four motion sizes in training** (0.8, 1.0, 1.3, 2.0). On the synthetic X-rays the network followed the size. On the real C08 X-rays it still answered a shallow breath. Final error: 11.83 mm (75) and 11.15 mm (300).
- **A scale on top of the frozen TCIA3 field.** On the TCIA patients the scale stayed at 1, because the field already matches those breaths. Where the field points the wrong way, a positive scale learns to shrink it.
- **Lung crop, then 128×128.** The lungs already filled the picture from top to bottom, so the up-down breath did not get bigger. Only the empty side was cut. C01 finished at 2.26 mm (75) against 2.13 mm before. C08 at epoch 20 was 11.81 mm (75) and 10.98 mm (300), the same as the finished run.

## Why C08 stays there

On the 82 TCIA patients, a big breath stays big. For the largest 01→06 breaths the TCIA3 field is about 93% of the real one (biggest: 13.3 mm real, 12.3 mm predicted).

C08 was not one of those patients. The network guesses a breath from that CT. The guess is too small and points the wrong way. VoxelMap is already copying that guess, so a clearer X-ray cannot get under it. On the 75 C08 landmarks the TCIA3 field itself is about 10.7 mm.

## Left running

C08 and C03 lung-crop training were still going when this was written. They do not replace the table above. TCIA3.1 (the oversampled MAE run) was paused during epoch 75 so C08 could use the GPU. Resume from `TCIA3.1/DecoderCRB/checkpoints/interrupt.pt`.
