"""Analytical translations/affine fields: phase identity, frames, and missingness."""
from dataclasses import replace
import json

import numpy as np
import pytest

from tre_viewer import data, phases


@pytest.fixture
def phase_run(tmp_path, monkeypatch):
    root = tmp_path / "DIR_C01_muhist"
    train = root / "DIR_C01" / "train"
    train.mkdir(parents=True)
    monkeypatch.setitem(data.CASE_INFO, 1, ((32, 24, 16), (.7, 1.3, 2.5)))
    src = np.tile([12., 10., 8.], (75, 1))
    src[:, 0] += np.linspace(-1, 1, 75)
    monkeypatch.setattr(data, "landmarks_75", lambda case, phase: src.copy())
    return data.RunRef("test", 1, "DIR_C01", root, "native", ("identity",)), src, train


def raw_field(run, train, raw, phase="T00"):
    nx, ny, nz = data.CASE_INFO[run.case][0]
    shape = (ny, nz, nx) if run.r3 else (nz, ny, nx)
    (run.run_root / "synth_meta.json").write_text(json.dumps({"native_shape": shape}))
    np.save(train / f"_synth_dvf_infer_{data.phase_to_train_idx(phase):02d}.npy", raw)


@pytest.mark.parametrize("frame", ["native", "r3"])
def test_synth_inverse_translation_all_components_mm(phase_run, monkeypatch, frame):
    run, src, train = phase_run
    run = replace(run, frame=frame)
    ev = data.require_evaluator()
    nx, ny, nz = data.CASE_INFO[1][0]
    src_pack = ev.official_to_pack(src, nz)
    src_g = ev.pack_to_r3(src_pack, ny, nz) if run.r3 else src_pack
    shift = np.array([1., -.6, .8])
    pred_g = src_g - shift  # raw field is target→reference; inverse subtracts a translation
    pred_pack = ev.r3_to_pack(pred_g, ny, nz) if run.r3 else pred_g
    truth = ev.pack_to_official(pred_pack, nz)
    monkeypatch.setattr(data, "landmarks_75", lambda case, phase: src if phase == 'T50' else truth)
    native_zyx = np.array((ny, nz, nx) if run.r3 else (nz, ny, nx))
    raw = np.broadcast_to((shift / (native_zyx / 8))[:, None, None, None], (3, 8, 8, 8)).copy()
    raw_field(run, train, raw)
    result = phases.evaluate_phase(run, 'synth', 'T00')
    assert result.status == 'ok', result.reason
    assert result.metric('mean') < .01
    np.testing.assert_allclose(result.predicted_mm, (truth-src) * data.CASE_INFO[1][1], atol=.01)
    assert result.metric('improvement') > 1


def test_nonconstant_pull_requires_actual_inverse(phase_run, monkeypatch):
    run, src, train = phase_run
    # Native pull u_x(q)=0.25*q_x+0.4. Inverse q=(p-0.4)/1.25.
    raw = np.zeros((3, 8, 8, 8))
    raw[0] = (.25 * np.linspace(0, 31, 8)[None, None, :] + .4) / (16/8)
    raw_field(run, train, raw)
    truth = src.copy()
    truth[:, 0] = (src[:, 0] - .4) / 1.25
    monkeypatch.setattr(data, 'landmarks_75', lambda case, phase: src if phase == 'T50' else truth)
    result = phases.evaluate_phase(run, 'synth', 'T00')
    assert result.status == 'ok', result.reason
    assert result.metric('mean') < .01
    # Simply negating u(p) gives >0.4 mm error here.
    assert np.mean(np.abs(src[:, 0] - (.25*src[:, 0]+.4) - truth[:, 0])) * .7 > .4


def test_inverse_outside_grid_not_reported_as_zero(phase_run):
    run, src, train = phase_run
    raw_field(run, train, np.full((3, 8, 8, 8), 100.))
    result = phases.evaluate_phase(run, 'synth', 'T00')
    assert result.status == 'error'
    assert result.landmarks is None
    assert 'inverse invalid' in result.reason


def test_missing_annotations_and_reference_not_conflated(phase_run, monkeypatch):
    run, src, _ = phase_run
    def points(case, phase):
        if phase == 'T70':
            raise FileNotFoundError('T70 has no 75-point landmarks')
        return src
    monkeypatch.setattr(data, 'landmarks_75', points)
    missing = phases.evaluate_phase(run, 'synth', 'T70')
    reference = phases.evaluate_phase(run, 'synth', 'T50')
    assert missing.status == 'unavailable'
    assert missing.landmarks is None and np.isnan(missing.metric('mean'))
    assert missing.to_dict()['mean_mm'] is None
    assert reference.status == 'reference' and reference.metric('mean') == 0


def test_training_labels_are_never_direct_synth_output(phase_run):
    run, _, train = phase_run
    (train / 'DVF_sub_01.mha').write_text('not a raw synth output')
    result = phases.evaluate_phase(run, 'synth', 'T00')
    assert result.status == 'unavailable'
    assert 'raw pull' in result.reason
    assert result.observed_mm is not None


def test_phase_specific_voxelmap_paths_and_no_phase01_fallback(phase_run, monkeypatch):
    run, src, _ = phase_run
    p = phases.phase_cache_path(run, 'T00')
    p.parent.mkdir()
    p.write_text('stub')
    monkeypatch.setattr(data, '_load_zyx_npy', lambda p: np.zeros((128, 128, 128, 3)))
    assert phases.evaluate_phase(run, 'voxelmap', 'T00').status == 'ok'
    missing = phases.evaluate_phase(run, 'voxelmap', 'T10')
    assert missing.status == 'unavailable'
    assert 'phase02' in missing.reason


def test_voxelmap_forward_reference_to_target(phase_run, monkeypatch):
    run, src, _ = phase_run
    shift_sub = np.array([2., -1., 3.])
    field = np.broadcast_to(shift_sub, (128, 128, 128, 3))
    expected = src + shift_sub * np.array([32, 24, -16]) / 128
    monkeypatch.setattr(data, 'landmarks_75', lambda case, phase: src if phase == 'T50' else expected)
    monkeypatch.setattr(phases, 'voxelmap_field', lambda *args, **kw: field)
    result = phases.evaluate_phase(run, 'voxelmap', 'T20')
    assert result.status == 'ok' and result.metric('mean') < 1e-10
    assert result.landmarks.pair == 'T50_T20'


def test_bad_field_is_invalid_and_export_has_nulls(phase_run, tmp_path):
    run, _, train = phase_run
    raw_field(run, train, np.full((3, 8, 8, 8), np.nan))
    result = phases.evaluate_phase(run, 'synth', 'T00')
    assert result.status == 'error'
    path = tmp_path / 'out.json'
    phases.export_results(path, [result])
    exported = json.loads(path.read_text())['results'][0]
    assert exported['mean_mm'] is None and exported['n'] == 0
    assert 'NaN' not in path.read_text()


def test_discovery_includes_synth_variants(phase_run):
    run, _, _ = phase_run
    refs = data.discover_runs_in_folder(run.run_root.parent)
    assert len(refs) == 1
    assert refs[0].case == 1 and refs[0].scan_id == 'DIR_C01'
    assert refs[0].run_root.name == 'DIR_C01_muhist'


def test_cancelled_cohort_has_no_fake_cells(phase_run):
    run, _, _ = phase_run
    assert list(phases.evaluate_cohort([run], 'synth', cancelled=lambda: True)) == []


def test_synth_trained_voxelmap_never_infers_from_synthetic_drrs(phase_run, monkeypatch, tmp_path):
    run, _, _ = phase_run
    (run.run_root / 'synth_meta.json').write_text('{}')
    training = run.run_root / 'ModelTraining' / 'train' / run.scan_id
    training.mkdir(parents=True)
    monkeypatch.setattr(data, 'ARMS_ROOT', tmp_path / 'arms')
    with pytest.raises(FileNotFoundError, match='real A1'):
        phases._real_model_training(run)
    real = data.ARMS_ROOT / 'A1_oracle_dirlab' / 'runs' / run.scan_id / 'ModelTraining' / 'train' / run.scan_id
    real.mkdir(parents=True)
    assert phases._real_model_training(run) == real


def test_inference_routes_phase_and_real_data_then_records_provenance(phase_run, monkeypatch, tmp_path):
    from types import SimpleNamespace
    run, _, _ = phase_run
    (run.run_root / 'synth_meta.json').write_text('{}')
    ckpt = run.run_root / 'checkpoints_nofilm' / 'best.pt'
    ckpt.parent.mkdir()
    ckpt.write_text('test checkpoint')
    real_root = tmp_path / 'real_runs'
    real = real_root / run.scan_id / 'ModelTraining' / 'train' / run.scan_id
    real.mkdir(parents=True)
    calls = []
    def infer(ckpt, mt, device, stride, phase):
        calls.append((mt, device, stride, phase))
        return np.zeros((128, 128, 128, 3), np.float32), 4
    monkeypatch.setattr(data, 'require_evaluator', lambda: SimpleNamespace(infer_voxelmap_dvf_phase=infer))
    phases.voxelmap_field(run, 'T20', infer=True, stride=7, device='cpu', real_runs_dir=real_root)
    assert calls == [(real, 'cpu', 7, 3)]
    path = phases.phase_cache_path(run, 'T20')
    assert path.is_file()
    assert not phases.phase_cache_path(run, 'T00').exists()
    provenance = json.loads(path.with_suffix('.json').read_text())
    assert provenance['phase'] == 'T20'
    assert provenance['data_dir'] == str(real)
    assert provenance['n_projections'] == 4


@pytest.mark.parametrize('frame', ['native', 'r3'])
def test_image_check_matches_real_ct_without_any_landmarks(phase_run, monkeypatch, frame):
    import SimpleITK as sitk
    run, _, train = phase_run
    run = replace(run, frame=frame)
    (run.run_root / 'synth_meta.json').write_text('{}')
    real = np.arange(16*24*32, dtype=np.float32).reshape(16, 24, 32) - 1000
    # R3[z_r,y_r,x] = pack[nz-1-y_r, ny-1-z_r, x].
    synth = real[::-1, ::-1, :].transpose(1, 0, 2) if run.r3 else real
    sitk.WriteImage(sitk.GetImageFromArray(synth + 5), str(train / 'CT_08.mha'))
    monkeypatch.setattr(data, 'load_pack_volume', lambda *a: data.PackVolume(real, (2.5, 1.3, .7), train, 'T70'))
    def missing(*a):
        raise FileNotFoundError('no annotation')
    monkeypatch.setattr(data, 'landmarks_75', missing)
    _, predicted, diff, mae = phases.synth_image_check(run, 'T70')
    assert mae == 5
    np.testing.assert_allclose(diff, 5)
    np.testing.assert_allclose(predicted, real + 5)


def test_cache_sidecar_rejects_wrong_phase_or_frame(phase_run):
    run, _, _ = phase_run
    path = phases.phase_cache_path(run, 'T20')
    path.parent.mkdir()
    path.write_text('not loaded when provenance is wrong')
    path.with_suffix('.json').write_text(json.dumps({'phase': 'T00', 'reference': 'T50', 'frame': 'native'}))
    result = phases.evaluate_phase(run, 'voxelmap', 'T20')
    assert result.status == 'error' and 'provenance mismatch' in result.reason


def test_real_input_frame_mismatch_is_rejected(phase_run, tmp_path):
    run, _, _ = phase_run
    root = tmp_path / 'real_runs' / run.scan_id
    (root / 'tre').mkdir(parents=True)
    (root / 'tre' / 'tre_summary.json').write_text('{"case": 1, "frame": "r3"}')
    with pytest.raises(ValueError, match='frame r3 differs'):
        phases._real_model_training(run, root.parent)


def test_partial_export_reports_coverage(phase_run, tmp_path):
    run, _, _ = phase_run
    path = tmp_path / 'partial.json'
    phases.export_results(path, [phases.evaluate_phase(run, 'synth', 'T00')], expected_cells=10)
    report = json.loads(path.read_text())
    assert report['complete'] is False
    assert report['evaluated_cells'] == 1 and report['expected_cells'] == 10
