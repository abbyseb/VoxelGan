"""
Reorient a DIR-Lab 4DCT volume into the SPARE GTVol world frame for RTK DRR generation.

DIR-Lab native (LPS, identity direction, origin at the volume corner):
    itk x = patient Left, itk y = Posterior, itk z = Superior
SPARE GTVol (identity direction, volume centred on the isocenter):
    itk x = patient Left, itk y = Superior, itk z = Anterior

RTK's ThreeDCircularProjectionGeometry orbits the gantry about world Y, so S-I must sit
on itk y and the patient must sit on the world origin, otherwise the gantry circles a
point outside the patient and the anatomy swings off the detector.

Use with Geometry_SPARE.xml (680 projections) and MC_VARIAN_DRR_OPTS
(detector 1024x768 @ 0.388 mm, origin (-200, -150, 0)).
"""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk

HU_LO, HU_HI = -1000.0, 3000.0

# SPARE MC/Varian reconstruction FOV (mm) in the transaxial plane
SPARE_FOV_AP_MM = 461.9
SPARE_FOV_LR_MM = 441.4


def reorient_for_spare_orbit(
    image: sitk.Image,
    *,
    pad_to_spare_fov: bool = False,
    clip_hu: bool = True,
) -> sitk.Image:
    """
    Args:
        image: DIR-Lab CT in native LPS layout, size (x, y, z) = (256, 256, 94).
        pad_to_spare_fov: air-pad the transaxial plane out to the SPARE recon FOV so the
            detector framing matches A2. Costs ~3.5x the voxels.
        clip_hu: clamp to the SPARE GTVol dynamic range.

    Returns:
        Volume with size (x, y, z) = (256, 94, 256), spacing (LR, SI, AP),
        identity direction and the grid centre on (0, 0, 0).
        Includes SI flip so head matches SPARE upright (R3; R1 was 180° upside-down).
    """
    arr = sitk.GetArrayFromImage(image).astype(np.float32)  # numpy (SI, Post, Left)
    sp_lr, sp_ap, sp_si = image.GetSpacing()

    # numpy (SI, Post, Left) -> (Post, SI, Left)
    # flip AP (axis 0) for handedness after odd permute; flip SI (axis 1) so head is up like SPARE
    # (R1 without SI flip was chest↔spine but upside-down vs SPARE reference)
    out = arr.transpose(1, 0, 2)[::-1, ::-1, :]

    if pad_to_spare_fov:
        pa = max(0, int(round(SPARE_FOV_AP_MM / sp_ap)) - out.shape[0]) // 2
        pl = max(0, int(round(SPARE_FOV_LR_MM / sp_lr)) - out.shape[2]) // 2
        out = np.pad(out, ((pa, pa), (0, 0), (pl, pl)), constant_values=HU_LO)

    if clip_hu:
        out = np.clip(out, HU_LO, HU_HI)

    img = sitk.GetImageFromArray(np.ascontiguousarray(out))
    spacing = (sp_lr, sp_si, sp_ap)  # itk (x, y, z) = (Left, Superior, Anterior)
    img.SetSpacing(spacing)
    img.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    img.SetOrigin(tuple(-(n - 1) * s / 2.0 for n, s in zip(img.GetSize(), spacing)))
    return img
