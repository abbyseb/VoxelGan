# This week's update (21–27 September 2026)

The numbers to keep are the finished TCIA3 VoxelMap runs on all 10 DIR-Lab cases. Later tries (deeper training views, a scale on the motion, a lung crop of the X-ray) did not beat them. C08 stays near 12 mm.

## Result

Inhale to exhale, real X-rays, R3. A3 TCIA3 (MAE), epoch 100, makes the motion. VoxelMap is trained on those synthetic X-rays and scored on the real ones.

75 landmarks:

| Case | A3 TCIA3 (MAE) | Do nothing |
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
| **Mean ± SD** | **5.26 ± 2.89 mm** | **8.69 ± 3.55 mm** |

300 landmarks:

| Case | A3 TCIA3 (MAE) | Do nothing |
|---|---:|---:|
| C01 | 1.87 mm | 3.89 mm |
| C02 | 1.84 mm | 4.34 mm |
| C03 | 3.19 mm | 6.94 mm |
| C04 | 5.30 mm | 9.83 mm |
| C05 | 3.57 mm | 7.48 mm |
| C06 | 5.93 mm | 10.89 mm |
| C07 | 7.41 mm | 11.03 mm |
| C08 | 10.99 mm | 14.99 mm |
| C09 | 5.47 mm | 7.92 mm |
| C10 | 4.76 mm | 7.30 mm |
| **Mean ± SD** | **5.03 ± 2.75 mm** | **8.46 ± 3.33 mm** |

The ± is the standard deviation across the 10 cases.

C01 and C02 are near 2 mm. C06 and C07 are already 6–7 mm. C08 is the worst, not the only miss. A registration on the same C08 landmarks is 3.77 mm (75 points) and 3.32 mm (300 points).

## TCIA2, next to the other finished arms

Same test: real X-rays, inhale to exhale. A3 TCIA2 (MSE) and A3 TCIA3 (MAE) are VoxelMap trained on synthetic X-rays from those motion networks. A3 SPARE is the same pipeline with the original SPARE motion. A2 Voxel is the generic SPARE prior, with no patient-specific motion. A1 Voxel is trained on that patient’s own real 4D scan. Identity is no motion.

75 landmarks:

| Case | A3 TCIA2 (MSE) | A3 TCIA3 (MAE) | A3 SPARE | Identity | A2 Voxel | A1 Voxel |
|---|---:|---:|---:|---:|---:|---:|
| C01 | 1.97 | 2.13 | 2.16 | 3.91 | 3.85 | 1.40 |
| C02 | 2.20 | 1.93 | 2.75 | 4.65 | 4.65 | 1.40 |
| C03 | 3.97 | 3.38 | 3.91 | 7.25 | 6.79 | 1.44 |
| C04 | 6.06 | 5.43 | 7.30 | 9.69 | 7.86 | 1.88 |
| C05 | 4.03 | 3.70 | 4.51 | 7.41 | 6.15 | 2.19 |
| C06 | 7.46 | 6.15 | 8.94 | 11.77 | 9.21 | 3.56 |
| C07 | 8.16 | 7.35 | 9.33 | 10.71 | 8.65 | 2.63 |
| C08 | 11.45 | 11.82 | 13.61 | 16.00 | 13.66 | 4.48 |
| C09 | 4.67 | 5.21 | 6.18 | 7.16 | 6.62 | 2.08 |
| C10 | 5.41 | 5.44 | 6.16 | 8.33 | 5.22 | 2.12 |
| **Mean ± SD** | **5.54 ± 2.89** | **5.26 ± 2.89** | **6.49 ± 3.48** | **8.69 ± 3.55** | **7.27 ± 2.82** | **2.32 ± 1.01** |

300 landmarks:

| Case | A3 TCIA2 (MSE) | A3 TCIA3 (MAE) | A3 SPARE | Identity | A2 Voxel | A1 Voxel |
|---|---:|---:|---:|---:|---:|---:|
| C01 | 1.83 | 1.87 | 2.14 | 3.89 | 3.78 | 1.29 |
| C02 | 2.06 | 1.84 | 2.63 | 4.34 | 4.36 | 1.31 |
| C03 | 3.72 | 3.19 | 3.61 | 6.94 | 6.61 | 1.40 |
| C04 | 6.01 | 5.30 | 7.33 | 9.83 | 8.09 | 1.77 |
| C05 | 3.88 | 3.57 | 4.52 | 7.48 | 6.07 | 2.09 |
| C06 | 6.91 | 5.93 | 8.35 | 10.89 | 8.46 | 3.47 |
| C07 | 8.29 | 7.41 | 9.60 | 11.03 | 8.85 | 2.83 |
| C08 | 10.58 | 10.99 | 12.74 | 14.99 | 12.89 | 4.04 |
| C09 | 5.15 | 5.47 | 6.67 | 7.92 | 7.23 | 1.96 |
| C10 | 4.78 | 4.76 | 5.42 | 7.30 | 4.54 | 2.14 |
| **Mean ± SD** | **5.32 ± 2.73** | **5.03 ± 2.75** | **6.30 ± 3.33** | **8.46 ± 3.33** | **7.09 ± 2.71** | **2.23 ± 0.94** |

A3 TCIA2 beats A3 SPARE and A2. A3 TCIA3 beats A3 TCIA2 by about 0.3 mm. A1 Voxel is still about 3 mm better than A3 TCIA3, because it sees that patient’s real scan.

## What we tried, and what failed

Baseline for the motion network is TCIA3 epoch 100: synth-oracle mean **4.94 mm** on 75 landmarks, C08 **10.65 mm**. The VoxelMap table above is the deployable number (real X-rays). Nothing this week beat either one by a useful amount.

| Try | What it was | Result |
|---|---|---|
| AmpHead | One scale from the phase number only, TCIA-trained, frozen TCIA3 | One scale for every DIR case (1.04). Oracle 4.94 → 4.86 mm (75) and 4.70 → 4.62 mm (300). Not a real gain. See below. |
| FeatAmpHead | Scale from the CT plus the size of the predicted motion | Scale stuck at the floor, 0.80, on all 10 cases. Cohort got worse: 5.50 mm. |
| MagFT | Fine-tune TCIA3 with an under-move penalty. Did not overwrite TCIA3. | Best snapshot −0.18 mm (4.76 at epoch 12). The saved best epoch was 4.89. C08 10.65 → 9.80 only at that early snapshot. Not the 1 mm we wanted. |
| MagMatch | Fine-tune so the motion size matches the real TCIA field, including the head-foot part | Killed at epoch 10. Cohort 5.13 mm, worse than 4.94. |
| Four depths (amp4) | Train VoxelMap on C08 at motion scales 0.8, 1.0, 1.3, 2.0 | It followed the size on synthetic X-rays. On real C08 X-rays the predicted breath stayed shallow (about 3 mm against a 13 mm label). Final landmarks: 11.83 mm (75) and 11.15 mm (300). |
| Scale map | A small network on the frozen TCIA3 field, free to multiply the motion by 0.5–2.5 | On TCIA patients the multiplier stayed at 1. The field already fits those patients. |
| Scale map on DIR | Same head, trained to match DIR registrations. Oracle only, not a result. | Where the real breath was larger, the multiplier went down. Shrinking a badly aimed arrow reduces the training error. |
| Brightness match | Force the real C08 X-rays to look like the synthetic ones, then predict | Predicted breath moved from 2.84 to 2.95. The network already rescales each X-ray on its own. |
| Lung crop, then 128×128 | Cut to the lungs before the shrink, same TCIA3 motion | The lungs already filled the picture top to bottom, so the up-down breath did not get bigger (1.00×). Finished cases (75 / 300 mm), next to the TCIA3 table: C01 2.26/1.93 (was 2.13/1.87), C02 1.94/1.83 (1.93/1.84), C03 3.40/3.18 (3.38/3.19), C04 5.25/5.13 (5.43/5.30), C05 3.69/3.61 (3.70/3.57), C06 5.88/5.66 (6.15/5.93), C08 11.67/10.79 (11.82/10.99). A few tenths either way. C07, C09, and C10 were still training on 28 September. |
| TCIA3.5 | Same MAE training as TCIA3, with the field-of-view augmentation turned off | Finished 100 epochs. Best validation is epoch 92: mean 5.02 mm, C08 11.07 mm. Epoch 100 is worse: mean 5.34 mm, C08 11.30 mm. Both behind TCIA3 (4.94 / 10.65). Not adopted. |
| TCIA3.1 | Same MAE, with extra copies of the large breaths | At epoch 35 the oracle was 5.39–5.59 mm, behind 4.94. Paused during epoch 75 so C08 could use the GPU. Not a result. |

### AmpHead

TCIA3 epoch 100 stays frozen. A small network sits on top of it. The only inputs are the two phase numbers. It outputs one scale, kept between 0.8 and 2.5, and that scale multiplies the whole motion field. It does not see the CT, so it cannot give C08 a different scale from C01.

Training used TCIA breathing pairs only, for 1000 epochs. For each pair the target scale is the one that matches the real TCIA lung-motion size. DIR cases were not in that training. On the inhale-to-exhale pair the learned scale is 1.04, and that same 1.04 is applied to every DIR case.

The numbers below are the oracle test: the TCIA3 field itself, sampled on the DIR landmarks. VoxelMap was not retrained, so the real-X-ray table (5.26 mm) is unchanged.

| | TRE75 | TRE300 |
|---|---:|---:|
| TCIA3 oracle | 4.94 ± 2.54 mm | 4.70 ± 2.40 mm |
| AmpHead × 1.04 | 4.86 ± 2.47 mm | 4.62 ± 2.34 mm |

C08 moves from 10.65 to 10.47 mm (75) and from 9.80 to 9.62 mm (300). About 0.08 mm on the cohort. A single scale cannot fix a field that also points the wrong way.

Two ideas were written down and not run. Measuring the diaphragm on the real X-rays and stretching the field to match was rejected: that uses the test scan to build the motion. Pasting measured TCIA breathing onto C01 and C08 was rejected for the same reason.

A separate check, not a trained model: if C08 were given the true motion size but kept our direction, the 75-landmark error would still be about 7.2 mm. The true direction with our size would still be about 9.8 mm. Registration on that case is 3.8 mm. The arrows are too small and they point the wrong way. A single multiplier cannot fix both.

## Why C08 stays there

On the 82 TCIA patients, a big breath stays big. For the largest 01→06 breaths the TCIA3 field is about 93% of the real one (biggest: 13.3 mm real, 12.3 mm predicted).

C08 was not one of those patients. The network guesses a breath from that CT. The guess is too small and points the wrong way. VoxelMap is already copying that guess, so a clearer X-ray cannot get under it. On the 75 C08 landmarks the TCIA3 field itself is about 10.7 mm.

## Left running

C07 and C09 lung-crop training were still going on 28 September, with C10 queued after C07. The finished cases do not replace the table above. TCIA3.1 (the oversampled MAE run) is still paused at epoch 75. Resume from `TCIA3.1/DecoderCRB/checkpoints/interrupt.pt`.
