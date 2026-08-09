"""Train FiLM warp-D on ALL P1 pairs (no leave-phase-out).

For P2 transfer: zero-shot then fine-tune from these weights.
Train/val both use data/spare/all (checkpoint by best supervised val).
"""

from utilities.train_loop import run_training

if __name__ == '__main__':
    run_training(
        filename='spare_mc_p1_full_warp_d',
        d_input_mode='warp',
        lambda_adv=0.05,
        d_update_freq=2,
        d_base_channels=32,
        train_dir='data/spare/all',
        val_dir='data/spare/all',
        held_out_phases=None,
        epoch_num=100,
        log_path='plots/train_full_p1_warp_d.log',
    )
