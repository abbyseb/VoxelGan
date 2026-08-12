"""QC for LeaveOut_3_6_8 — delegates to Dan2.0 QC with local ckpt/out."""
import runpy, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DAN20 = ROOT.parent / 'Dan2.0'
sys.argv = [
    'qc_crb_pairs.py',
    '--ckpt', str(ROOT / 'weights' / 'dans_crb_mse_lo_3_6_8_generator.pth'),
    '--out_dir', str(ROOT / 'plots' / 'qc_test_mse'),
    '--view_config', str(DAN20 / 'configs' / 'dvf_view_config.json'),
]
sys.path.insert(0, str(DAN20))
runpy.run_path(str(DAN20 / 'scripts' / 'qc_crb_pairs.py'), run_name='__main__')
