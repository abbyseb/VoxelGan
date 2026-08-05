"""Scenario: warp-space discriminator (phase-MLP baseline knobs).

Baseline D: 4-ch [ref CT, DVF] — can cheat on Elastix vs SVF spectral cues.
This run:   2-ch [warp(ref, DVF), target CT] — D judges image-space realism.

All other knobs match spare_mc_p1_dvf_gan_phase_mlp (λ_adv=0.05, D width=32,
d_update_freq=2, dense patches). Train from scratch; do not overwrite baseline.
"""

from utilities.train_loop import run_training

if __name__ == '__main__':
    run_training(
        filename='spare_mc_p1_scenario_warp_d',
        # --- scenario change ---
        d_input_mode='warp',
        # --- phase-MLP baseline defaults ---
        lambda_adv=0.05,
        d_update_freq=2,
        d_base_channels=32,
    )
