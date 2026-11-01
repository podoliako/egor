"""Standalone viewer regressions for saved real-pick runs, without forward inputs."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server


_SHAPE = (3, 2, 4)
_SUBDIVISION = 8
_INITIAL = 4000. + np.arange(np.prod(_SHAPE), dtype=float).reshape(_SHAPE)
_RUN_NAMES = {
    "hard": "run_geysers-real-hard-pick01_v1.5-sparse-real_20260920_120000",
    "soft": "run_geysers-real-soft-pick01_v1.5-sparse-real_20260920_120001",
}
_EVENT_IDS = ["event-Z", "event-A"]
_STATION_IDS = ["STA.Z", "STA.A", "STA.M"]
_SPARSE_RESIDUALS = np.array([[999., 3., np.nan],
                              [100., 999., 4.],
                              [200., 300., 999.]])
_MISSING = object()


@pytest.fixture
def real_viewer(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    experiments = tmp_path / "no-forward-inputs"
    queue_manifest = tmp_path / "unmounted-host" / "queue_manifest.json"
    original_open = Path.open

    def local_open(path, *args, **kwargs):
        if path == queue_manifest or path.is_relative_to(experiments):
            raise AssertionError(f"Real-run viewer must not read source inputs: {path}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", local_open)
    monkeypatch.setattr(server, "RUNS_DIR", runs)
    monkeypatch.setattr(server, "EXPERIMENTS_ROOT", experiments)
    monkeypatch.setitem(server.app.config, "TESTING", True)

    def add_run(mode="hard"):
        rd = runs / _RUN_NAMES[mode]
        rd.mkdir()
        meta = {
            "viewer_completion_protocol": 1,
            "run_params": {"n_events": 2, "n_cycles": 1, "subdivision": _SUBDIVISION,
                           "viewer_quality_expected": False},
            "grid_info": {"coarse_shape": list(_SHAPE), "coarse_cell_size": 1000.,
                          "coarse_side_m": [3000., 2000., 4000.],
                          "fine_shape": [n * _SUBDIVISION for n in _SHAPE],
                          "fine_cell_size": 125.},
            "station_locs": [[0., 0., 0.], [1000., 500., 0.], [3000., 2000., 4000.]],
            "event_locs": [],
            "source_experiment": {
                "kind": "real_picks", "id": "geysers_real_v1_train90",
                "arrays_sha256": hashlib.sha256(b"tiny real-pick fixture").hexdigest(),
                "event_ids": list(_EVENT_IDS), "station_ids": list(_STATION_IDS),
                "queue_manifest": str(queue_manifest),
            },
        }
        (rd / "meta.json").write_text(json.dumps(meta))
        np.save(rd / "initial_model.npy", _INITIAL)
        cycle = rd / "iter_0"
        cycle.mkdir()
        for name, values in (("model", _INITIAL + 200.),
                             ("delta_s", np.full(_SHAPE, 1.e-6)),
                             ("ray_count", np.ones(_SHAPE)),
                             ("sensitivity_diagonal", np.full(_SHAPE, 2.)),
                             ("coverage_confidence", np.full(_SHAPE, .5))):
            np.save(cycle / f"{name}.npy", values)
        for event in range(2):
            event_dir = cycle / f"event_{event}"
            event_dir.mkdir()
            np.savez(event_dir / "weights.npz", weight_shape=meta["grid_info"]["fine_shape"],
                     weight_indices=[[0, 0, 0]], positions=[[.25, .5, .75]],
                     weight_values=[1.])
            residuals = _SPARSE_RESIDUALS if event == 0 else np.full((3, 3), np.nan)
            np.save(event_dir / "residuals.npy", residuals)
        (rd / "timing.jsonl").write_text(json.dumps({"iter": 0, "elapsed_s": 1.}) + "\n")
        # Publication is explicit; absence of truth/quality must not hide this cycle.
        (cycle / "complete.json").write_text(json.dumps(
            {"iter": 0, "viewer_completion_protocol": 1}))
        return rd

    return server.app.test_client(), add_run, runs


def _json(client, url, **kwargs):
    response = client.get(url, **kwargs)
    assert response.status_code == 200, response.get_data(as_text=True)
    return response.get_json()


def _assert_excluded(client, rd):
    assert _json(client, "/api/runs") == []
    assert _json(client, "/api/runs/summary") == []
    assert _json(client, "/api/runs/latest") == {"run_id": None, "max_iter": None}
    for endpoint in ("meta", "info", "quality", "hypo_metrics", "slice?model_type=initial"):
        assert client.get(f"/api/runs/{rd.name}/{endpoint}").status_code == 404


def test_hard_and_soft_real_runs_are_visible_in_summary_and_latest(real_viewer):
    client, add, _ = real_viewer
    hard = add("hard")
    soft = add("soft")
    assert _json(client, "/api/runs") == [soft.name, hard.name]
    assert _json(client, "/api/runs/summary") == [
        {"id": rd.name, "run_name": rd.name, "iterations": [0],
         "completed_iterations": 1, "planned_iterations": 1}
        for rd in (soft, hard)
    ]
    assert _json(client, "/api/runs/latest") == {"run_id": soft.name, "max_iter": 0}
    for rd in (hard, soft):
        base = f"/api/runs/{rd.name}/"
        info = _json(client, base + "info")
        assert info["iterations"] == [0]
        assert info["completed_iterations"] == info["planned_iterations"] == 1
        assert info["n_events"] == 2
        assert info["n_stations"] == 3
        assert info["has_true_model"] is False
        assert _json(client, base + "iters_list") == ["iter_0"]
        assert _json(client, base + "timing") == [{"iter": 0, "elapsed_s": 1.}]
        assert not (rd / "quality.jsonl").exists()
        assert _json(client, base + "quality") == []


@pytest.mark.parametrize("mode", ["hard", "soft"])
def test_real_source_ids_are_not_reference_events_and_need_no_host_manifest(real_viewer, mode):
    client, add, _ = real_viewer
    rd = add(mode)
    original = (rd / "meta.json").read_bytes()
    saved = json.loads(original)
    assert not Path(saved["source_experiment"]["queue_manifest"]).exists()
    assert not server.EXPERIMENTS_ROOT.exists()
    assert not (rd / "true_model.npy").exists()
    meta = _json(client, f"/api/runs/{rd.name}/meta")
    assert meta["source_experiment"] == saved["source_experiment"]
    assert meta["event_ids"] == _EVENT_IDS
    assert meta["station_ids"] == _STATION_IDS
    assert meta["station_locs"] == saved["station_locs"]
    assert meta["has_reference_events"] is False
    assert meta["event_locs"] == []
    assert meta["reference_event_ids"] == []
    assert meta["reference_event_coordinates_m"] == []
    assert meta["coarse_ny"] == 2
    assert meta["fine_ny"] == 16
    assert server._source_model(rd, saved) is None
    assert (rd / "meta.json").read_bytes() == original


@pytest.mark.parametrize("mode", ["hard", "soft"])
@pytest.mark.parametrize("plane", ["xz", "xy"])
@pytest.mark.parametrize("phase", ["initial", "iter", "final"])
def test_real_model_slices_initial_iteration_and_final(real_viewer, mode, plane, phase):
    client, add, _ = real_viewer
    rd = add(mode)
    base = f"/api/runs/{rd.name}/"
    # The existing viewer represents final by the last published iter, not model_type=final.
    iteration = _json(client, base + "info")["iterations"][-1] if phase == "final" else 0
    query = {"type": "model", "model_type": "initial" if phase == "initial" else "iter",
             "iter": iteration, "plane": plane}
    query["y_km" if plane == "xz" else "z_km"] = .5 if plane == "xz" else 1.5
    values = _INITIAL if phase == "initial" else _INITIAL + 200.
    coarse_plane = values[:, 0, :] if plane == "xz" else values[:, :, 1]
    expected = coarse_plane.repeat(_SUBDIVISION, axis=0).repeat(_SUBDIVISION, axis=1)
    data = _json(client, base + "slice", query_string=query)
    np.testing.assert_array_equal(data["slice"], expected)
    assert data["shape"] == list(expected.shape)
    assert data["full_shape"] == [24, 16, 32]
    assert data["slice_plane"] == plane
    assert data["cell_size"] == 125.
    assert data["x_edges_km"][0] == 0.
    assert data["x_edges_km"][-1] == 3.
    axis = "z" if plane == "xz" else "y"
    assert data[f"{axis}_edges_km"][-1] == (4. if plane == "xz" else 2.)
    assert data["vmin"] == expected.min()
    assert data["vmax"] == expected.max()
    # Explicit coarse resolution uses the same API as the existing viewer.
    query.update(nx=3, ny=2, nz=4)
    coarse = _json(client, base + "slice", query_string=query)
    np.testing.assert_array_equal(coarse["slice"], coarse_plane)
    assert coarse["full_shape"] == list(_SHAPE)


@pytest.mark.parametrize("plane", ["xz", "xy"])
def test_real_runs_have_no_truth_slice_or_fabricated_hypocentre_distances(real_viewer, plane):
    client, add, _ = real_viewer
    rd = add()
    base = f"/api/runs/{rd.name}/"
    assert _json(client, base + "info")["has_true_model"] is False
    # A saved initial model and nonempty hypotheses are not a reference model/events.
    assert client.get(base + "slice", query_string={
        "type": "model", "model_type": "true", "plane": plane}).status_code == 404
    metrics = _json(client, base + "hypo_metrics?iter=0")
    assert metrics["distance_iter"] == []
    assert metrics["distance_summary"] == []
    assert metrics["residual_iter"]


@pytest.mark.parametrize("mode", ["hard", "soft"])
@pytest.mark.parametrize("all_missing", [False, True])
def test_sparse_real_residual_metrics_are_finite_or_absent(real_viewer, mode, all_missing):
    client, add, _ = real_viewer
    rd = add(mode)
    if all_missing:
        np.save(rd / "iter_0" / "event_0" / "residuals.npy", np.full((3, 3), np.nan))
    metrics = _json(client, f"/api/runs/{rd.name}/hypo_metrics?iter=0")
    assert metrics["distance_iter"] == metrics["distance_summary"] == []
    if all_missing:
        assert metrics["residual_iter"] == metrics["residual_summary"] == []
    else:
        rms = np.sqrt((3. ** 2 + 4. ** 2) / 2.)
        assert metrics["residual_iter"] == [{"event": 0, "rms": pytest.approx(rms)}]
        assert metrics["residual_summary"] == [
            {"iter": 0, "n_events": 1, "mean": pytest.approx(rms),
             "median": pytest.approx(rms), "p10": pytest.approx(rms), "p90": pytest.approx(rms)}]
        assert np.isfinite(metrics["residual_iter"][0]["rms"])


@pytest.mark.parametrize("residuals, expected", [
    pytest.param(_SPARSE_RESIDUALS, np.sqrt(12.5), id="finite-upper-triangle-only"),
    pytest.param([[0., 3., np.inf], [100., 0., 4.], [200., 300., 0.]],
                 np.sqrt(12.5), id="ignore-infinity"),
    pytest.param(np.zeros((3, 3)), 0., id="observed-zero-is-valid"),
    pytest.param(np.full((3, 3), np.nan), None, id="all-nan"),
    pytest.param([[0., np.nan, np.inf], [3., 0., -np.inf], [4., 5., 0.]],
                 None, id="finite-diagonal-and-lower-are-not-picks"),
    pytest.param(np.empty((0, 0)), None, id="no-stations"),
    pytest.param([[0.]], None, id="no-station-pairs"),
])
def test_residual_rms_filters_missing_pairs(tmp_path, residuals, expected):
    path = tmp_path / "residuals.npy"
    np.save(path, residuals)
    rms = server._rms_from_residuals(path)
    if expected is None:
        assert rms is None
    else:
        assert rms == pytest.approx(expected)
        assert np.isfinite(rms)


def test_missing_residual_file_is_not_zero_rms(tmp_path):
    assert server._rms_from_residuals(tmp_path / "missing.npy") is None


@pytest.mark.parametrize("path, value", [
    pytest.param(("source_experiment",), _MISSING, id="missing-source"),
    pytest.param(("source_experiment",), None, id="null-source"),
    pytest.param(("source_experiment",), [], id="source-not-object"),
    pytest.param(("source_experiment", "kind"), _MISSING, id="missing-explicit-kind"),
    pytest.param(("source_experiment", "kind"), "synthetic", id="wrong-kind"),
    pytest.param(("source_experiment", "id"), _MISSING, id="missing-source-id"),
    pytest.param(("source_experiment", "id"), "", id="empty-source-id"),
    pytest.param(("source_experiment", "id"), "../escape", id="escaping-source-id"),
    pytest.param(("source_experiment", "id"), 123, id="source-id-not-string"),
    pytest.param(("source_experiment", "arrays_sha256"), _MISSING, id="missing-arrays-hash"),
    pytest.param(("source_experiment", "arrays_sha256"), "a" * 63, id="short-hash"),
    pytest.param(("source_experiment", "arrays_sha256"), "a" * 65, id="long-hash"),
    pytest.param(("source_experiment", "arrays_sha256"), "g" * 64, id="nonhex-hash"),
    pytest.param(("source_experiment", "arrays_sha256"), 123, id="hash-not-string"),
    pytest.param(("source_experiment", "event_ids"), _MISSING, id="missing-event-ids"),
    pytest.param(("source_experiment", "event_ids"), [], id="empty-event-ids"),
    pytest.param(("source_experiment", "event_ids"), ["e"], id="event-count-mismatch"),
    pytest.param(("source_experiment", "event_ids"), ["e", "e"], id="duplicate-event-ids"),
    pytest.param(("source_experiment", "event_ids"), ["e", 2], id="event-id-not-string"),
    pytest.param(("source_experiment", "station_ids"), _MISSING, id="missing-station-ids"),
    pytest.param(("source_experiment", "station_ids"), "ABC", id="station-ids-not-list"),
    pytest.param(("source_experiment", "station_ids"), ["a", "b"], id="station-count-mismatch"),
    pytest.param(("source_experiment", "station_ids"), ["a", "b", "a"], id="duplicate-station-ids"),
    pytest.param(("source_experiment", "station_ids"), ["a", "b", 3], id="station-id-not-string"),
    pytest.param(("run_params", "n_events"), _MISSING, id="missing-event-count"),
    pytest.param(("run_params", "n_events"), 1, id="wrong-event-count"),
    pytest.param(("run_params", "n_events"), True, id="boolean-event-count"),
    pytest.param(("run_params", "n_events"), 2.5, id="fractional-event-count"),
    pytest.param(("station_locs",), _MISSING, id="missing-station-locations"),
    pytest.param(("station_locs",), [[0, 0, 0]] * 2, id="station-location-count"),
    pytest.param(("station_locs",), [[0, 0]] * 3, id="station-location-not-3d"),
    pytest.param(("station_locs",), [[0, 0, 0], [1, 2], [1, 2, 3]], id="ragged-stations"),
    pytest.param(("station_locs",), [[np.nan, 0, 0]] * 3, id="nan-station"),
    pytest.param(("station_locs",), [[0, np.inf, 0]] * 3, id="infinite-station"),
    pytest.param(("station_locs",), [[-1, 0, 0]] * 3, id="negative-station"),
    pytest.param(("station_locs",), [[3001, 0, 0]] * 3, id="station-outside-x"),
    pytest.param(("station_locs",), [[0, 2001, 0]] * 3, id="station-outside-y"),
    pytest.param(("station_locs",), [[0, 0, 4001]] * 3, id="station-outside-z"),
    pytest.param(("grid_info",), _MISSING, id="missing-grid"),
    pytest.param(("grid_info",), None, id="null-grid"),
    pytest.param(("grid_info", "coarse_shape"), [0, 2, 4], id="zero-coarse-dimension"),
    pytest.param(("grid_info", "coarse_shape"), [-3, 2, 4], id="negative-coarse-dimension"),
    pytest.param(("grid_info", "coarse_shape"), [3, 2], id="coarse-shape-not-3d"),
    pytest.param(("grid_info", "coarse_shape"), [3.5, 2, 4], id="fractional-coarse-dimension"),
    pytest.param(("grid_info", "coarse_cell_size"), 0, id="zero-coarse-cell"),
    pytest.param(("grid_info", "coarse_cell_size"), np.nan, id="nan-coarse-cell"),
    pytest.param(("grid_info", "coarse_side_m"), [3000, 2000, 3999], id="coarse-extent-mismatch"),
    pytest.param(("grid_info", "fine_shape"), [24, 16, 31], id="fine-shape-mismatch"),
    pytest.param(("grid_info", "fine_shape"), [24, 16], id="fine-shape-not-3d"),
    pytest.param(("grid_info", "fine_cell_size"), 250., id="fine-cell-mismatch"),
    pytest.param(("grid_info", "fine_cell_size"), np.inf, id="infinite-fine-cell"),
    pytest.param(("run_params", "subdivision"), 4, id="subdivision-mismatch"),
    pytest.param(("run_params", "subdivision"), 0, id="zero-subdivision"),
])
def test_invalid_real_metadata_is_excluded_from_every_discovery_endpoint(real_viewer, path, value):
    client, add, _ = real_viewer
    rd = add()
    meta = json.loads((rd / "meta.json").read_text())
    target = meta
    for key in path[:-1]:
        target = target[key]
    if value is _MISSING:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    (rd / "meta.json").write_text(json.dumps(meta))
    _assert_excluded(client, rd)


@pytest.mark.parametrize("contents", ["{", "[]", "null", '"not metadata"'])
def test_unreadable_or_nonobject_metadata_is_excluded(real_viewer, contents):
    client, add, _ = real_viewer
    rd = add()
    (rd / "meta.json").write_text(contents)
    _assert_excluded(client, rd)


@pytest.mark.parametrize("problem", [
    "missing", "truncated", "wrong-shape", "flat", "zero", "negative", "nan", "infinite",
    "object-array", "outside-symlink",
])
def test_invalid_or_uncontained_initial_model_is_excluded(real_viewer, tmp_path, problem):
    client, add, _ = real_viewer
    rd = add()
    path = rd / "initial_model.npy"
    if problem == "missing":
        path.unlink()
    elif problem == "truncated":
        path.write_bytes(path.read_bytes()[:32])
    elif problem == "wrong-shape":
        np.save(path, _INITIAL.reshape(2, 3, 4))
    elif problem == "flat":
        np.save(path, _INITIAL.ravel())
    elif problem == "object-array":
        np.save(path, _INITIAL.astype(object))
    elif problem == "outside-symlink":
        outside = tmp_path / "outside_initial.npy"
        np.save(outside, _INITIAL)
        path.unlink()
        path.symlink_to(outside)
    else:
        values = _INITIAL.copy()
        values[0, 0, 0] = {"zero": 0., "negative": -1., "nan": np.nan, "infinite": np.inf}[problem]
        np.save(path, values)
    _assert_excluded(client, rd)
