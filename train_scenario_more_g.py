"""Scenario: more generator steps relative to the discriminator.

Baseline: d_update_freq=2  → update D once every 2 G steps.
This run: d_update_freq=5  → update D once every 5 G steps (more G pressure).
All other knobs match baseline phase-MLP (lambda_adv=0.05, D width=32).
"""

from utilities.train_loop import run_training

if __name__ == '__main__':
    run_training(
        filename='spare_mc_p1_scenario_more_g',
        # --- scenario change ---
        d_update_freq=5,      # more G steps per D step (baseline uses 2)
        # --- baseline defaults ---
        lambda_adv=0.05,
        d_base_channels=32,
    )
