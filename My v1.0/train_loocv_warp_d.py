"""Leave-one-phase-out LOOCV with warp-D recipe (sequential folds).

For each held-out phase k in {0…9}:
  train on pairs that do not touch k
  validate on pairs that touch k
Uses data/spare/all (full 100-pair pool). Warp-space PatchGAN, phase-MLP FiLM.
"""

import os
from datetime import datetime

from utilities.train_loop import run_training


def main():
    from pathlib import Path
    data_dir = str(Path(__file__).resolve().parents[1] / 'data' / 'spare' / 'all')
    os.makedirs('plots/loocv_warp_d', exist_ok=True)
    summary_path = 'plots/loocv_warp_d/summary.log'
    with open(summary_path, 'a') as f:
        f.write(f'\n=== LOOCV warp-D start {datetime.now().isoformat()} ===\n')

    # 0-indexed phase fold ids
    for phase0 in range(10):
        phase1 = phase0 + 1
        filename = f'spare_mc_p1_loocv_warp_d_hold{phase1:02d}'
        log_path = f'plots/loocv_warp_d/fold_hold{phase1:02d}.log'
        print(f'\n######## LOOCV fold hold-out phase {phase1:02d} ########\n', flush=True)
        with open(log_path, 'a') as f:
            f.write(f'start {datetime.now().isoformat()} hold_out_phase={phase1}\n')

        best = run_training(
            filename=filename,
            d_input_mode='warp',
            lambda_adv=0.05,
            d_update_freq=2,
            d_base_channels=32,
            train_dir=data_dir,
            val_dir=data_dir,
            held_out_phases=[phase0],
            log_path=log_path,
        )
        line = f'fold hold{phase1:02d} best_val={best:.6f}\n'
        print(line, flush=True)
        with open(summary_path, 'a') as f:
            f.write(line)

    with open(summary_path, 'a') as f:
        f.write(f'=== LOOCV warp-D done {datetime.now().isoformat()} ===\n')


if __name__ == '__main__':
    main()
