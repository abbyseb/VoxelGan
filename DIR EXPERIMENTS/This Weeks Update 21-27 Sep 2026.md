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

## What we tried, and what failed

Baseline for the motion network is TCIA3 epoch 100: synth-oracle mean **4.94 mm** on 75 landmarks, C08 **10.65 mm**. The VoxelMap table above is the deployable number (real X-rays). Nothing this week beat either one by a useful amount.

| Try | What it was | Result |
|---|---|---|
| AmpHead | One scale from the phase number only, TCIA-trained, frozen TCIA3 | One scale for every DIR case (1.04). Cohort 4.94 → 4.86 mm. Not a real gain. |
| FeatAmpHead | Scale from the CT plus the size of the predicted motion | Scale stuck at the floor, 0.80, on all 10 cases. Cohort got worse: 5.50 mm. |
| MagFT | Fine-tune TCIA3 with an under-move penalty. Did not overwrite TCIA3. | Best snapshot −0.18 mm (4.76 at epoch 12). The saved best epoch was 4.89. C08 10.65 → 9.80 only at that early snapshot. Not the 1 mm we wanted. |
| MagMatch | Fine-tune so the motion size matches the real TCIA field, including the head-foot part | Killed at epoch 10. Cohort 5.13 mm, worse than 4.94. |
| Four depths (amp4) | Train VoxelMap on C08 at motion scales 0.8, 1.0, 1.3, 2.0 | It followed the size on synthetic X-rays. On real C08 X-rays the predicted breath stayed shallow (about 3 mm against a 13 mm label). Final landmarks: 11.83 mm (75) and 11.15 mm (300). |
| Scale map | A small network on the frozen TCIA3 field, free to multiply the motion by 0.5–2.5 | On TCIA patients the multiplier stayed at 1. The field already fits those patients. |
| Scale map on DIR | Same head, trained to match DIR registrations. Oracle only, not a result. | Where the real breath was larger, the multiplier went down. Shrinking a badly aimed arrow reduces the training error. |
| Brightness match | Force the real C08 X-rays to look like the synthetic ones, then predict | Predicted breath moved from 2.84 to 2.95. The network already rescales each X-ray on its own. |
| Lung crop, then 128×128 | Cut to the lungs before the shrink, same TCIA3 motion | The lungs already filled the picture top to bottom, so the up-down breath did not get bigger (1.00×). C01 finished 2.26 mm against 2.13 mm. C08 at epoch 20 was 11.81 / 10.98 mm, the same as the finished run. |
| TCIA3.5 | Same MAE training as TCIA3, with the field-of-view augmentation turned off | Finished 100 epochs. A check near epoch 88 was already behind: about 5.15 mm mean, C08 11.35 mm. Not adopted. |
| TCIA3.1 | Same MAE, with extra copies of the large breaths | At epoch 35 the oracle was 5.39–5.59 mm, behind 4.94. Paused during epoch 75 so C08 could use the GPU. Not a result. |

Two ideas were written down and not run. Measuring the diaphragm on the real X-rays and stretching the field to match was rejected: that uses the test scan to build the motion. Pasting measured TCIA breathing onto C01 and C08 was rejected for the same reason.

A separate check, not a trained model: if C08 were given the true motion size but kept our direction, the 75-landmark error would still be about 7.2 mm. The true direction with our size would still be about 9.8 mm. Registration on that case is 3.8 mm. The arrows are too small and they point the wrong way. A single multiplier cannot fix both.

## Why C08 stays there

On the 82 TCIA patients, a big breath stays big. For the largest 01→06 breaths the TCIA3 field is about 93% of the real one (biggest: 13.3 mm real, 12.3 mm predicted).

C08 was not one of those patients. The network guesses a breath from that CT. The guess is too small and points the wrong way. VoxelMap is already copying that guess, so a clearer X-ray cannot get under it. On the 75 C08 landmarks the TCIA3 field itself is about 10.7 mm.

## Left running

C08 and C03 lung-crop training were still going when this was written. They do not replace the table above. TCIA3.1 (the oversampled MAE run) was paused during epoch 75 so C08 could use the GPU. Resume from `TCIA3.1/DecoderCRB/checkpoints/interrupt.pt`.
