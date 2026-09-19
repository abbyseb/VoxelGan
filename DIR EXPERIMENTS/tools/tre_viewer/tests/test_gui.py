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
