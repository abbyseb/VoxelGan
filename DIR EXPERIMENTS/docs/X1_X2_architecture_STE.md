# X1 and X2: model architecture

Language: ASD-STE100 Simplified Technical English.
Branch: `Unsupervised-Training`.
Code:
- X1: `PopulationStudy/ClinicalExperiments/Grid160/TCIA_lite/run_A2_full160_aug_xray1/scripts/train_xray1.py`
- X2: `PopulationStudy/ClinicalExperiments/Grid160/TCIA_lite/run_A2_full160_aug_xray2/scripts/train_xray2.py`
- X-ray projector: `PopulationStudy/ClinicalExperiments/Grid160/TCIA_lite/xray/drr_torch.py`

---

## 1. Purpose

The models calculate the 3D motion of the lungs.

The motion is between two breathing phases.

The models use two inputs:

- The planning CT. This CT shows the lungs at the reference phase.
- One live X-ray. This X-ray shows the patient at the target phase.

The output is a deformation field. The deformation field gives one 3D vector for each voxel.

Each vector tells where the tissue at that voxel moves.

The models do not use the target CT at test time.

The models use one set of weights for all patients. They do not use training on the test patient.

---

## 2. Data

### 2.1 Grid

| Item | Value |
|---|---|
| Grid size | 160 × 160 × 160 voxels |
| Voxel size | 2 mm (isotropic) |
| Field of view | 320 mm |
| Grid centre | Lung centroid of the reference CT |
| Axis order | (AP, SI, LR) |

### 2.2 Training data

| Item | Value |
|---|---|
| Source | TCIA 4D-Lung |
| Training scans | 65 scans (16 patients) |
| Validation scans | 17 scans (4 patients) |
| Phase pairs | 10 × 10 pairs for each scan |
| Training pairs | 6500 |
| Validation pairs | 1700 |

A validation patient is not a training patient.

### 2.3 Test data

The test data is DIR-Lab 4D-CT (10 cases).

The input CT is phase T50 (exhale).

The target phase is T00 (inhale).

The error metric is TRE. TRE is the mean distance between the predicted and the true positions of 300 landmarks.

---

## 3. The X-ray (DRR)

The models use a simulated X-ray. This X-ray is a digitally reconstructed radiograph (DRR).

The DRR is made from the target CT during training.

### 3.1 Geometry

| Item | Value |
|---|---|
| Beam type | Cone beam |
| Source to isocentre distance | 1000 mm |
| Source to detector distance | 1500 mm |
| Detector size | 128 × 128 pixels |
| Pixel size | 4 mm |
| Gantry angle | 0° (front view) |

### 3.2 Projection

For each detector pixel, a ray goes from the source through the CT.

The projector adds the CT values along the ray. The step size is 1 mm.

The projector is differentiable. It agrees with the RTK Joseph projector (correlation more than 0.99999).

### 3.3 Standardisation

The models standardise each X-ray image.

Standardisation subtracts the mean and divides by the standard deviation.

This removes the effect of a different X-ray brightness.

### 3.4 Back-projection

A 2D X-ray does not have the same shape as the 3D CT.

Back-projection changes the 2D X-ray into a 3D volume.

For each voxel, back-projection finds the ray through the voxel. Then it copies the X-ray value of that ray into the voxel.

All voxels on one ray get the same value.

Back-projection gives a volume with the same shape as the CT.

Back-projection does not give the depth of a structure.

---

## 4. X1

### 4.1 Function

X1 is the first X-ray model.

X1 gives the CT and one back-projected X-ray to the network.

### 4.2 Inputs

| Channel | Content |
|---|---|
| 1 | Reference CT (min–max normalised) |
| 2 | Back-projected, standardised front X-ray of the target |

The input has 2 channels.

### 4.3 Network

The network is a 3D U-Net with 5 levels.

| Part | Layers | Output channels |
|---|---|---|
| Encoder level 1 | Residual block (2 → 16) | 16 |
| Encoder level 2 | Average pool, residual block | 32 |
| Encoder level 3 | Average pool, residual block | 32 |
| Encoder level 4 | Average pool, residual block | 64 |
| Bottleneck | Average pool, residual block | 64 |
| Decoder level 4 | Upsample, skip, conditional block | 64 |
| Decoder level 3 | Upsample, skip, conditional block | 32 |
| Decoder level 2 | Upsample, skip, conditional block | 32 |
| Decoder level 1 | Upsample, skip, conditional block | 16 |
| Output | 3 × 3 × 3 convolution | 3 |

A residual block has two 3 × 3 × 3 convolutions and a skip connection.

The network has no normalisation layers.

The number of parameters is approximately 1.07 million.

### 4.4 Phase conditioning

The decoder uses two numbers:

- The reference phase.
- The target phase.

Each conditional block has a small fully connected network. This network uses the two numbers. It gives a scale and a shift for each feature channel.

The block multiplies each feature channel by its scale. Then it adds the shift.

This method is FiLM.

### 4.5 Output

The output has 3 channels. They are the x, y and z parts of the displacement, in voxels.

The field is a pull field:

`target(p) = reference(p + u(p))`

### 4.6 Loss

The loss does not use labels.

1. The training script moves the reference CT with the predicted field.
2. The script compares the moved CT with the target CT.
3. The comparison is local normalised cross-correlation (LNCC). The window is 9 × 9 × 9 voxels.
4. The script uses the lung mask with a 2-voxel dilation.

`loss = 1 − mean LNCC (in the lung)`

### 4.7 Training

| Item | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 1e-4 to 1e-6 (cosine) |
| Epochs | 40 |
| Batch size | 1 |
| Augmentation | Flip, shift, zoom, brightness, noise |
| X-ray during training | Perfect DRR |

### 4.8 Results (DIR-Lab, epochs 36–40)

| X-ray | TRE |
|---|---|
| Perfect DRR | 3.77 mm |
| Perfect DRR, mirror average | 3.71 mm |
| Realistic X-ray (scatter, noise, blur) | 3.94 mm |
| No X-ray (zero input) | 5.00 mm |
| X-ray of a different patient | 4.21 mm |

The CT-only model (no X-ray) has a TRE of 4.03 mm.

X1 is better than the CT-only model.

X1 is robust to realistic X-rays.

---

## 5. X2

### 5.1 Function

X2 is X1 with two more inputs:

- A difference image ("what moved").
- A breath code.

### 5.2 Difference image

1. Make the DRR of the target CT: `Dt`.
2. Make the DRR of the reference CT at the same angle: `Dr`.
3. Standardise each DRR.
4. Subtract: `d = std(Dt) − std(Dr)`.

Structures that do not move cancel in `d`. Examples are the spine and the ribs.

The motion stays in `d`. Examples are the diaphragm band and the change in lung density.

If the reference and target CTs are the same, `d` is zero.

### 5.3 Inputs

| Channel | Content |
|---|---|
| 1 | Reference CT |
| 2 | Back-projected `std(Dt)` |
| 3 | Back-projected difference image `d` |

The input has 3 channels.

### 5.4 Breath code

A small 2D network reads the difference image `d` (128 × 128).

| Layer | Content |
|---|---|
| 1–4 | 3 × 3 convolution, stride 2, ReLU (channels 1 → 16 → 32 → 32 → 32) |
| 5 | Global average |
| 6 | Fully connected layer (32 → 8) |

The output is 8 numbers. These numbers are the breath code.

The breath code gives a summary of the breath. An example is the depth of the breath.

### 5.5 Conditioning

The decoder uses 10 numbers:

- 2 phase numbers.
- 8 breath-code numbers.

Each conditional block uses these 10 numbers for FiLM (section 4.4).

The breath code changes all decoder levels.

### 5.6 Network, loss and training

The network is the X1 network with these changes:

- Encoder level 1 has 3 input channels.
- The conditional blocks use 10 numbers.
- The breath-code network is added.

The number of parameters is approximately 1.10 million.

The loss and the training are the same as X1 (sections 4.6 and 4.7).

X2 uses perfect DRRs during training.

### 5.7 Results (DIR-Lab, epochs 36–40, perfect DRR)

| Case | X1 | X2 |
|---|---|---|
| C01 | 1.89 | 1.68 |
| C02 | 2.30 | 1.53 |
| C03 | 2.46 | 2.07 |
| C04 | 3.36 | 2.96 |
| C05 | 4.27 | 2.78 |
| C06 | 3.92 | 3.04 |
| C07 | 6.51 | 4.21 |
| C08 | 6.37 | 5.10 |
| C09 | 3.51 | 2.93 |
| C10 | 3.08 | 2.99 |
| Mean | 3.77 | 2.93 |

X2 is better than X1 for all 10 cases.

### 5.8 Limit

X2 is not robust to scatter.

| X-ray | X2 TRE |
|---|---|
| Perfect DRR | 2.93 mm |
| Noise only | 3.23 mm |
| Blur only | 3.35 mm |
| Scatter (SPR 0.3) | 10.85 mm |
| Realistic X-ray | 9.32–9.62 mm |
| Realistic X-ray with the X-ray cleaner | 4.06 mm |

Scatter changes the brightness of the X-ray smoothly.

The difference image then shows changes that are not motion.

X2 uses these changes as motion. The result is incorrect.

The variant X2R uses realistic X-rays during training. X2R is robust, but less accurate (approximately 3.7 mm).

---

## 6. X2 B-spline

X2 B-spline is X2 with a different output.

All inputs, the network trunk, the loss and the training are the same as X2.

### 6.1 Output

The decoder stops at the 20 × 20 × 20 level.

A 1 × 1 × 1 convolution gives 3 numbers for each control point.

The control point spacing is 8 voxels (16 mm).

A fixed cubic B-spline changes the control points into the full 160 × 160 × 160 field.

The B-spline has no weights.

### 6.2 Results (DIR-Lab, epochs 36–40, perfect DRR)

| Case | X2 | X2 B-spline |
|---|---|---|
| C01 | 1.68 | 1.53 |
| C02 | 1.53 | 1.77 |
| C03 | 2.07 | 2.41 |
| C04 | 2.96 | 3.13 |
| C05 | 2.78 | 2.80 |
| C06 | 3.04 | 3.26 |
| C07 | 4.21 | 4.82 |
| C08 | 5.10 | 5.45 |
| C09 | 2.93 | 3.21 |
| C10 | 2.99 | 3.16 |
| Mean | 2.93 | 3.15 |

The B-spline field is always smooth. The folding is approximately 0 %.

The B-spline field has less local detail. The TRE is 0.22 mm larger.

X2 B-spline is not robust to scatter (realistic X-ray: 12.10 mm).

---

## 7. Summary

| Model | Inputs | Perfect DRR | Realistic X-ray |
|---|---|---|---|
| CT only | CT | 4.03 mm | 4.03 mm |
| X1 | CT + X-ray | 3.77 mm | 3.94 mm |
| X2 | CT + X-ray + difference + breath code | 2.93 mm | 9.32 mm |
| X2 B-spline | As X2, B-spline output | 3.15 mm | 12.10 mm |

X1 is simple and robust.

X2 is the most accurate model with clean X-rays.

X2 needs X-rays without scatter.
