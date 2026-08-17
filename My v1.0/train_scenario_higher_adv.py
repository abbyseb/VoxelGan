"""Scenario: slightly higher adversarial weight on the generator.

Baseline: lambda_adv=0.05.
This run: lambda_adv=0.15  → stronger push for G to fool D.
All other knobs match baseline phase-MLP (d_update_freq=2, D width=32).
"""

from utilities.train_loop import run_training

if __name__ == '__main__':
    run_training(
        filename='spare_mc_p1_scenario_higher_adv',
        # --- scenario change ---
        lambda_adv=0.15,      # higher adv weight (baseline uses 0.05)
        # --- baseline defaults ---
        d_update_freq=2,
        d_base_channels=32,
    )
