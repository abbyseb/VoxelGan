"""Real Qt/napari interactions using small synthetic data; no patient packs needed."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tre_viewer import app as viewer_module
from tre_viewer.data import PackVolume, PerLandmarkResult, RunRef, stats


@pytest.fixture
def viewer(tmp_path, monkeypatch):
    pytest.importorskip("napari")
    from qtpy.QtWidgets import QApplication
    qt_app = QApplication.instance() or QApplication([])
    root = tmp_path / "DIR_C01"
    root.mkdir()
    (root / "tre").mkdir()
    (root / "tre" / "tre_summary.json").write_text('{"case": 1, "frame": "native"}')
    run = RunRef("synthetic", 1, "DIR_C01", root, "native", ("identity",))
    monkeypatch.setitem(viewer_module.CASE_INFO, 1, ((12, 10, 8), (1., 1., 1.)))

    def volume(run, phase):
        values = np.arange(8 * 10 * 12, dtype=np.float32).reshape(8, 10, 12) - 900
        return PackVolume(values, (1., 1., 1.), root / phase, phase)

    def landmarks(run, field, which="75", pair="T00_T50", **kwargs):
        n = int(which)
        points = np.tile([5., 5., 4.], (n, 1))
        tre = np.ones(n)
        return PerLandmarkResult(
            run.case, field, which, pair, run.frame,
            points, points + 1, points + 1, points, points + 1, points + 1,
            np.ones_like(points), tre, tre, np.zeros(n, dtype=bool), stats(tre), stats(tre),
        )

    monkeypatch.setattr(viewer_module, "load_pack_volume", volume)
    monkeypatch.setattr(viewer_module, "per_landmark", landmarks)
    def no_warp(*args):
        raise FileNotFoundError("Synthetic test has no sub volumes")
    monkeypatch.setattr(viewer_module, "field_warp_bundle", no_warp)
    app = viewer_module.TreViewerApp(run, show=False)
    yield app
    for page in list(app._proj_pages.values()):
        page.win.close()
    app.viewer.close()
    qt_app.processEvents()


def test_visibility_and_orientation_update_without_apply(viewer):
    viewer._cb_src_pts.setChecked(True)
    assert viewer.layer_src_pts.visible
    viewer._ctrl_orient.setCurrentText("coronal")
    assert viewer._orient_name == "coronal"
    viewer._set_orient("sagittal")
    assert viewer._ctrl_orient.currentText() == "sagittal"


def test_failed_case_load_preserves_current_case_and_data(viewer, monkeypatch):
    original_run, original_data = viewer.run, viewer.layer_ct.data
    def missing(*args):
        raise FileNotFoundError("missing target volume")
    monkeypatch.setattr(viewer_module, "load_pack_volume", missing)
    assert viewer._switch_to_run(replace(viewer.run, case=2)) is False
    assert viewer.run is original_run
    assert viewer.layer_ct.data is original_data
    assert "missing target volume" in viewer._error_label.text()


def test_failed_landmark_selection_rolls_back_controls(viewer, monkeypatch):
    def missing(*args, **kwargs):
        raise ValueError("300-point landmarks unavailable")
    monkeypatch.setattr(viewer_module, "per_landmark", missing)
    viewer._ctrl_which.setCurrentText("300")
    viewer._apply_overlay_controls()
    assert viewer.which == "75"
    assert viewer._ctrl_which.currentText() == "75"
    assert len(viewer.pl.tre_mm) == 75


def test_failed_warp_removes_stale_overlay_and_restores_target(viewer):
    viewer.layer_warped.visible = True
    viewer.layer_ct.visible = False
    viewer.layer_warped.data = np.ones_like(viewer.vol_dst.data)
    viewer._refresh_dvf_overlays()
    assert not viewer.layer_warped.visible
    assert viewer.layer_ct.visible
    assert not viewer.layer_warped.data.any()
    assert viewer._bundle is None


def test_arrows_follow_slice_changes(viewer):
    viewer._bundle = {"dvf": np.ones((4, 4, 4, 3))}
    viewer.layer_dvf_arrows.visible = True
    viewer.viewer.dims.set_current_step(0, 2)
    viewer._refresh_arrows()
    np.testing.assert_allclose(viewer.layer_dvf_arrows.data[:, 0, 0], 2)
    viewer.viewer.dims.set_current_step(0, 5)
    np.testing.assert_allclose(viewer.layer_dvf_arrows.data[:, 0, 0], 5)


def test_projection_page_clears_stale_plot_and_updates_slider(viewer):
    from tre_viewer.proj_pages import ProjectionPageWindow
    page = ProjectionPageWindow(viewer, kind="drr")
    try:
        page.ax.imshow(np.ones((8, 8)))
        viewer._geom = SimpleNamespace(n_views=7)
        viewer._drr_view = 6
        page.refresh()
        assert not page.ax.images
        assert page.slider.maximum() == 6
        assert page.slider.value() == 6
        viewer._geom = SimpleNamespace(n_views=2)
        page.sync_slider()
        assert page.slider.maximum() == 1
        assert page.slider.value() == 1
    finally:
        page.win.close()


def test_phase_ct_missing_tre_clears_points_and_returns_to_kpi(viewer):
    from tre_viewer.phases import PhaseResult
    missing = PhaseResult(viewer.run, 'synth', 'T70', 'unavailable', 'No T70 annotations')
    assert viewer._switch_to_run(viewer.run, phase_record=missing)
    assert viewer.vol_dst.phase == 'T70'
    assert not len(viewer.layer_truth.data)
    assert not len(viewer.layer_pred.data)
    assert 'TRE unavailable' in viewer.summary_label.text()
    assert not viewer._ctrl_pair.isEnabled()
    viewer._update_case_tre_label()
    assert 'TRE unavailable' in viewer._case_tre_label.text()
    assert not viewer._error_label.text()
    assert not viewer.layer_warped.visible
    viewer.jump_worst()  # empty set must be safe
    viewer.jump_to_landmark(74)
    assert viewer._switch_to_run(viewer.run)
    assert viewer.pair == 'T00_T50'
    assert len(viewer.layer_truth.data) == 75
    assert viewer._ctrl_pair.isEnabled()


def test_phase_heatmap_click_links_phase_and_landmark(viewer):
    from tre_viewer.phases import PhaseResult
    panel = viewer._phase_panel
    panel.runs = [viewer.run]
    panel.patient.addItem(viewer.run.run_root.name)
    record = PhaseResult(viewer.run, 'synth', 'T20', 'ok', landmarks=viewer.pl,
                         observed_mm=np.zeros((75, 3)), predicted_mm=np.ones((75, 3)))
    panel.results[(str(viewer.run.run_root), 'T20')] = record
    panel.redraw()
    panel.landmark.setValue(7)
    panel._plot_clicked(SimpleNamespace(inaxes=panel.figures[1].axes[0],
                                       canvas=panel.canvases[1], xdata=2, ydata=0))
    assert viewer.vol_dst.phase == 'T20'
    assert viewer.layer_truth.selected_data == {7}
    assert panel.phase.currentText() == 'T20'
    viewer.jump_to_landmark(4)
    assert panel.landmark.value() == 4


def test_failed_phase_ct_load_preserves_previous_selection(viewer, monkeypatch):
    from tre_viewer.phases import PhaseResult
    original = viewer.layer_ct.data
    def missing(*args):
        raise FileNotFoundError('missing phase CT')
    monkeypatch.setattr(viewer_module, 'load_pack_volume', missing)
    record = PhaseResult(viewer.run, 'synth', 'T90', 'unavailable')
    assert not viewer._switch_to_run(viewer.run, phase_record=record)
    assert viewer.layer_ct.data is original
    assert viewer._phase_record is None
    assert viewer.pair == 'T00_T50'


def test_phase_background_evaluation_and_stage_reset(viewer, monkeypatch):
    from qtpy.QtWidgets import QApplication
    from tre_viewer import phase_panel
    from tre_viewer.phases import PhaseResult, PHASES
    import time
    def evaluate(runs, stage, **kwargs):
        for phase in PHASES:
            yield PhaseResult(runs[0], stage, phase, 'unavailable', 'test missing annotation')
    monkeypatch.setattr(phase_panel, 'evaluate_cohort', evaluate)
    panel = viewer._phase_panel
    panel.evaluate()
    deadline = time.monotonic() + 20
    while panel.worker is not None and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(.01)
    assert panel.worker is None
    assert len(panel.results) == 10
    assert '10 unavailable' in panel.status.text()
    panel.stage.setCurrentIndex(1)
    assert not panel.results
    assert not panel.image_button.isEnabled()


def test_phase_inspection_clears_open_projection_pages(viewer):
    from tre_viewer.phases import PhaseResult
    from tre_viewer.proj_pages import ProjectionPageWindow
    page = ProjectionPageWindow(viewer, kind='drr')
    try:
        page.ax.imshow(np.ones((8, 8)))
        viewer._phase_record = PhaseResult(viewer.run, 'synth', 'T70', 'unavailable')
        page.refresh()
        assert not page.ax.images
        assert viewer._drr_frame_data() is None
    finally:
        page.win.close()


def test_phase_image_worker_shows_images_without_landmarks_and_clears_on_return(viewer, monkeypatch):
    from qtpy.QtWidgets import QApplication
    from tre_viewer import phase_panel
    from tre_viewer.phases import PhaseResult
    import time
    panel = viewer._phase_panel
    panel.runs = [viewer.run]
    panel.patient.addItem(viewer.run.run_root.name)
    panel.phase.setCurrentText('T70')
    record = PhaseResult(viewer.run, 'synth', 'T70', 'unavailable', 'No annotations')
    panel.results[(str(viewer.run.run_root), 'T70')] = record
    truth = viewer.vol_dst
    monkeypatch.setattr(phase_panel, 'synth_image_check', lambda *a: (truth, truth.data + 5, np.full_like(truth.data, 5), 5.))
    panel.image_check()
    deadline = time.monotonic() + 20
    while panel.image_worker is not None and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(.01)
    assert panel.image_worker is None
    assert len(viewer._phase_images) == 2
    assert '5.0 HU' in panel.detail.text()
    assert not len(viewer.layer_truth.data)
    layers = list(viewer._phase_images)
    assert viewer._switch_to_run(viewer.run)
    assert all(layer not in viewer.viewer.layers for layer in layers)
