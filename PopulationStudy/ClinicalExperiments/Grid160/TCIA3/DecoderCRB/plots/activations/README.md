# TCIA3 activation maps (C01 and C08)

Decoder checkpoint: `DecoderCRB/checkpoints/epoch_100.pt` (A3 TCIA3, MAE).

Input is the real DIR phase-06 CT. The network predicts the motion from phase 06 to phase 01. Hotter colour means a higher value. Each picture shows three mid-slices of the 160³ volume, labeled View 1, View 2, and View 3. View 2 is the axial slice. View 3 is the sagittal slice, turned 90° clockwise. The top row is the CT. The bottom row is the map drawn on that CT.

## Files in each case folder

| File | What it shows |
|---|---|
| `enc1_meanabs.png` | First encoder block. Average absolute activation across channels. This is where the CT first lights the network up: body and lung edges, not yet the motion. |
| `bottleneck_meanabs.png` | Deepest layer, at the lowest resolution. A coarse summary of the whole scan before the decoder expands it back. |
| `up1_meanabs.png` | Last decoder block, just before the 3-channel motion head. Closest feature picture to the predicted motion. |
| `gradcam_up1.png` | Grad-CAM on that last decoder block. The score being explained is the mean size of the predicted motion. Hot spots are the places that push that score up. |
| `gradcam_enc_bottleneck_decoder.png` | The same Grad-CAM, for the encoder, the bottleneck, and the decoder, on one figure. |
| `pred_mag_u.png` | Size of the predicted motion itself (`‖u‖`), not an attention map. |

Grad-CAM and the mean-activation pictures are different. Mean activation is “where are the features large?”. Grad-CAM is “where did those features contribute to the motion size?”.
