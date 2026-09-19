#!/usr/bin/env bash
# Copy Iso-E2 checkpoints into G160-A0 / A0h names under Experiment0/weights.
set -euo pipefail
E0="$(cd "$(dirname "$0")/.." && pwd)"
ISO="$(cd "$E0/../../../IsoExperiments/Experiment2" && pwd)"

copy() { cp -a "$1" "$2"; echo "copied $(basename "$2")"; }

copy "$ISO/EncoderCRB/weights/crb_enc_mse_iso_e2_full_generator.pth" \
  "$E0/EncoderCRB/weights/crb_enc_mse_iso_g160_a0_full_generator.pth"
copy "$ISO/DecoderCRB/weights/crb_dec_mse_iso_e2_full_generator.pth" \
  "$E0/DecoderCRB/weights/crb_dec_mse_iso_g160_a0_full_generator.pth"
copy "$ISO/BothCRB/weights/crb_both_mse_iso_e2_full_generator.pth" \
  "$E0/BothCRB/weights/crb_both_mse_iso_g160_a0_full_generator.pth"
copy "$ISO/EncoderCRB/weights/crb_enc_mse_iso_e2_generator.pth" \
  "$E0/EncoderCRB/weights/crb_enc_mse_iso_g160_a0h_generator.pth"
copy "$ISO/DecoderCRB/weights/crb_dec_mse_iso_e2_generator.pth" \
  "$E0/DecoderCRB/weights/crb_dec_mse_iso_g160_a0h_generator.pth"
if [ -f "$ISO/BothCRB/weights/crb_both_mse_iso_e2_generator.pth" ]; then
  copy "$ISO/BothCRB/weights/crb_both_mse_iso_e2_generator.pth" \
    "$E0/BothCRB/weights/crb_both_mse_iso_g160_a0h_generator.pth"
else
  echo "NOTE: both A0h not in Iso-E2 — train separately if needed"
fi
