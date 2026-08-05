"""Scenario: weaker discriminator (half channel width).

Baseline: PatchDiscriminator base_channels=32 (32→64→128→256).
This run:     base_channels=16 (16→32→64→128) so D is easier for G to keep up with.
All other knobs match baseline phase-MLP (lambda_adv=0.05, d_update_freq=2).
"""

from utilities.train_loop import run_training

if __name__ == '__main__':
    run_training(
        filename='spare_mc_p1_scenario_weak_d',
        # --- scenario change ---
        d_base_channels=16,   # weaker D (baseline uses 32)
        # --- baseline defaults ---
        lambda_adv=0.05,
        d_update_freq=2,
    )
