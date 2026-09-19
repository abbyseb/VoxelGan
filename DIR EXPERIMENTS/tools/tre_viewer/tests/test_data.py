from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tre_viewer import data
from tre_viewer.__main__ import main


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "runs" / "DIR_C01"
    root.mkdir(parents=True)
    return data.RunRef("test", 1, "DIR_C01", root, "native", ("identity",))


@pytest.fixture
def points(monkeypatch):
    src = np.tile([30., 40., 50.], (75, 1))
    dst = src + [2., 0., 0.]
    monkeypatch.setattr(data, "_load_pair", lambda *args: (src, dst))
    return src, dst


def test_cache_invalidates_when_landmarks_change(run, points):
    first = data.per_landmark(run, "identity")
    points[1][:, 0] += 1
    second = data.per_landmark(run, "identity")
    assert second.registered_stats["mean"] > first.registered_stats["mean"]
    np.testing.assert_allclose(second.tre_mm, 3 * .97)


def test_cache_invalidates_when_dvf_changes(run, points):
    path = data.voxelmap_cache_path(run)
    path.parent.mkdir()
    dvf = np.zeros((128, 128, 128, 3), dtype=np.float32)
    np.save(path, dvf)
    first = data.per_landmark(run, "voxelmap")
    dvf[..., 0] = 1
    np.save(path, dvf)
    second = data.per_landmark(run, "voxelmap")
    np.testing.assert_allclose(second.pred_pack[:, 0], points[0][:, 0] - 2)
    assert second.registered_stats["mean"] > first.registered_stats["mean"]


def test_cache_write_failure_does_not_prevent_computation(run, points, monkeypatch):
    def cannot_save(*args, **kwargs):
        raise PermissionError("read-only folder")
    monkeypatch.setattr(data, "_save_per_landmark_npz", cannot_save)
    with pytest.warns(UserWarning, match="could not save cache"):
        result = data.per_landmark(run, "identity")
    assert len(result.tre_mm) == 75


def test_corrupt_cache_is_rebuilt(run, points):
    path = data.per_landmark_cache_path(run, "identity", "75", "T00_T50")
    path.parent.mkdir()
    path.write_bytes(b"broken npz")
    with pytest.warns(UserWarning, match="Rebuilding"):
        result = data.per_landmark(run, "identity")
    np.testing.assert_allclose(result.tre_mm, 2 * .97)


def test_custom_field_cache_stays_in_tre_directory(run):
    path = data.per_landmark_cache_path(run, "/tmp/fields/displacement.npy", "75", "T00_T50")
    assert path.parent == run.run_root / "tre"
    assert path.name.startswith("per_landmark_75_T00_T50_custom_")


def test_unknown_frame_is_not_assumed_native(run, points):
    with pytest.raises(ValueError, match="Unknown DVF frame"):
        data.per_landmark(replace(run, frame="unknown"), "elastix_mha")


@pytest.mark.parametrize("shape", [(2, 4, 4, 4), (128, 3, 128), (4, 4)])
def test_malformed_volume_is_rejected_without_squeeze_loop(tmp_path, shape):
    path = tmp_path / "bad.npy"
    np.save(path, np.zeros(shape))
    with pytest.raises(ValueError, match="128³"):
        data._load_hwd_volume(path)


def test_dvf_requires_all_three_spatial_dimensions(tmp_path):
    path = tmp_path / "bad.npy"
    np.save(path, np.zeros((128, 2, 128, 3)))
    with pytest.raises(ValueError, match="128"):
        data._load_zyx_npy(path)
    with pytest.raises(ValueError, match="128"):
        data.require_evaluator().load_dvf_zyx3(path)


def test_discovery_skips_invalid_cases_and_handles_bad_json(run):
    (run.run_root.parent / "DIR_C99").mkdir()
    path = run.run_root / "tre" / "tre_summary.json"
    path.parent.mkdir()
    path.write_text("[]")
    with pytest.warns(UserWarning, match="JSON object"):
        found = data.discover_runs_in_folder(run.run_root.parent)
    assert [r.case for r in found] == [1]
    assert found[0].tre_summary_path is None
    with pytest.warns(UserWarning):
        assert not data.discover_runs_in_folder(run.run_root.parent, require_summary=True)


@pytest.mark.parametrize("pair", ["T00_T50", "T50_T00"])
def test_verification_matches_correct_a0_landmark_set(run, monkeypatch, tmp_path, pair):
    a0 = tmp_path / "A0_identity" / "results"
    a0.mkdir(parents=True)
    (a0 / "summary.json").write_text(json.dumps({"cases": [
        {"case": 1, "set": "300", "src": "T00", "dst": "T50", "mean_mm": 3.89},
        {"case": 1, "set": "75", "src": "T00", "dst": "T50", "mean_mm": 3.91},
    ]}))
    monkeypatch.setattr(data, "ARMS_ROOT", tmp_path)
    def compute(*args, **kwargs):
        assert kwargs["use_cache"] is False
        return SimpleNamespace(identity_stats={"mean": 3.91})
    monkeypatch.setattr(data, "per_landmark", compute)
    checks = data.verify_against_summary(run, field="identity", which="75", pair=pair)
    assert checks[0]["ok"] is True
    assert checks[0]["expected"] == 3.91


def test_verification_does_not_pass_without_references(run, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(data, "ARMS_ROOT", tmp_path)
    monkeypatch.setattr(data, "per_landmark", lambda *a, **k: SimpleNamespace(identity_stats={"mean": 3.91}))
    code = main(["verify", "--runs-dir", str(run.run_root.parent), "--field", "identity"])
    assert code == 1
    assert "Verification incomplete" in capsys.readouterr().out


def test_300_verification_does_not_use_legacy_75_summary(run, monkeypatch, tmp_path):
    monkeypatch.setattr(data, "ARMS_ROOT", tmp_path)
    monkeypatch.setattr(data, "per_landmark", lambda *a, **k: SimpleNamespace(identity_stats={"mean": 1.}))
    monkeypatch.setattr(data, "load_tre_summary", lambda r: {"arms": {"elastix_mha": {
        "T00_T50": {"registered": {"mean": 1.}, "identity": {"mean": 1.}}
    }}})
    checks = data.verify_against_summary(run, field="elastix_mha", which="300")
    assert checks[0]["ok"] is False
    assert checks[0]["note"] == "pair missing in summary"


def test_cli_reports_missing_run_without_traceback(tmp_path, capsys):
    assert main(["view", "--runs-dir", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert "No run" in err and "Traceback" not in err


def test_doctor_reports_missing_system_library(monkeypatch, capsys):
    import importlib
    original = importlib.import_module
    def import_with_missing_glib(name, *args, **kwargs):
        if name == "PyQt6.QtWidgets":
            raise ImportError("libglib-2.0.so.0: cannot open shared object file")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(importlib, "import_module", import_with_missing_glib)
    assert main(["doctor"]) == 1
    assert "[MISSING] PyQt6.QtWidgets: libglib-2.0.so.0" in capsys.readouterr().out


@pytest.mark.parametrize("args", [["view", "--case", "0"], ["verify", "--atol", "nan"]])
def test_cli_rejects_invalid_values(args):
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2


def test_reverse_image_warp_is_explicitly_unavailable(run):
    with pytest.raises(ValueError, match="inverse DVF"):
        data.field_warp_bundle(run, "elastix_mha", "T50_T00")


def test_coordinate_round_trips():
    evaluator = data.require_evaluator()
    pts = np.array([[1., 2., 3.], [20.5, 40.25, 60.75]])
    pack = evaluator.official_to_pack(pts, 94)
    np.testing.assert_allclose(evaluator.pack_to_official(pack, 94), pts)
    np.testing.assert_allclose(evaluator.r3_to_pack(evaluator.pack_to_r3(pack, 256, 94), 256, 94), pack)


def test_clamping_is_reported_on_sub_grid():
    src_off = np.array([[255., 255., 0.]])
    _, _, oob = data._push_forward(np.zeros((128, 128, 128, 3)), 1, src_off, r3=False, sign=-1)
    assert oob.tolist() == [True]


def test_missing_packs_fail_identity_check(tmp_path, monkeypatch):
    import dirlab_tre
    monkeypatch.setattr(dirlab_tre, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        dirlab_tre.report_identity(1)
