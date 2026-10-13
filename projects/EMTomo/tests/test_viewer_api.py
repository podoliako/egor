"""Viewer contract for saved forward-input truth and physical slice axes."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server
from config import InversionConfig
from instruments.likelihood import PickNoise
from tomography.tomography_em import run_em
from velocity_model import VelocityModel


def _save_viewer_cycle(rd, iteration, *, complete=True):
    """Standalone fixtures include the outputs needed to infer old-format completion."""
    meta = json.loads((rd / "meta.json").read_text())
    shape = tuple(meta["grid_info"]["coarse_shape"])
    directory = rd / f"iter_{iteration}"
    directory.mkdir(exist_ok=True)
    for name, value in (("model", 5200.), ("delta_s", 1.), ("ray_count", 1.),
                        ("sensitivity_diagonal", 2.), ("coverage_confidence", 0.5)):
        np.save(directory / f"{name}.npy", np.full(shape, value))
    for event in range(meta["run_params"]["n_events"]):
        event_dir = directory / f"event_{event}"
        event_dir.mkdir(exist_ok=True)
        np.savez(event_dir / "weights.npz", weight_shape=shape,
                 weight_indices=np.empty((0, 3), dtype=int), weight_values=[])
        np.save(event_dir / "residuals.npy", np.empty((0, 0)))
    with (rd / "timing.jsonl").open("a") as stream:
        stream.write(json.dumps({"iter": iteration, "elapsed_s": 1.}) + "\n")
    with (rd / "quality.jsonl").open("a") as stream:
        stream.write(json.dumps({"iter": iteration, "avg_abs_pct_dev": 1., "rms_m_s": 2.}) + "\n")
    if complete and meta.get("viewer_completion_protocol") == 1:
        (directory / "complete.json").write_text(json.dumps(
            {"iter": iteration, "viewer_completion_protocol": 1}))
    return directory


@pytest.fixture
def viewer(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    experiments = tmp_path / "experiments" / "input" / "sample"
    experiments.mkdir(parents=True)
    truth = np.arange(16, dtype=float).reshape(4, 2, 2) + 4000
    model = experiments / "model.npz"
    np.savez(model, velocity=truth, cell_size_m=np.array(60000.), origin_m=np.zeros(3))
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    (experiments / "stations.csv").write_text("station_id,x_m,y_m,z_m\nsB,0,0,0\nsA,10000,0,0\n")
    (experiments / "events.csv").write_text(
        "event_id,x_m,y_m,z_m\neZ,5000,5000,5000\neA,15000,5000,5000\neM,25000,5000,5000\n")
    output = tmp_path / "experiments" / "output" / "sample"
    output.mkdir(parents=True)
    (output / "metadata.json").write_text(json.dumps({
        "time_reference": "earliest_station_arrival_per_event",
        "units": {"coordinates": "m", "cell_size": "m", "velocity": "m/s",
                  "arrival_time": "s", "elapsed_time": "s"},
        "input_sha256": {name: hashlib.sha256((experiments / name).read_bytes()).hexdigest()
                         for name in ("model.npz", "stations.csv", "events.csv")},
    }))
    (output / "arrivals.csv").write_text(
        "station_id,event_id,arrival_time_s\n" +
        "".join(f"sA,{event},1\nsB,{event},0\n" for event in ("eM", "eA", "eZ")))
    grid = {"coarse_shape": [24, 12, 12], "coarse_cell_size": 10000.,
            "coarse_side_m": [240000., 120000., 120000.], "fine_cell_size": 10000.}

    def add_run(name, *, source=True, hash_value=digest, iteration=True, protocol=1):
        rd = runs / name
        rd.mkdir()
        meta = {"run_params": {"subdivision": 1, "n_events": 3}, "grid_info": grid,
                "station_locs": [[0, 0, 0], [10000, 0, 0]], "event_locs": [],
                "source_experiment": {"id": "sample", "model_sha256": hash_value} if source else None}
        if protocol is not None:
            meta["viewer_completion_protocol"] = protocol
        (rd / "meta.json").write_text(json.dumps(meta))
        np.save(rd / "initial_model.npy", np.full((24, 12, 12), 5100.))
        if iteration:
            _save_viewer_cycle(rd, 0)
        return rd

    monkeypatch.setattr(server, "RUNS_DIR", runs)
    monkeypatch.setattr(server, "EXPERIMENTS_ROOT", tmp_path / "experiments")
    server.app.config["TESTING"] = True
    return server.app.test_client(), add_run, truth, model, digest, runs


def test_reference_events_are_viewer_only_and_keep_input_order(viewer):
    client, add, _, _, _, _ = viewer
    rd = add("current")
    original = (rd / "meta.json").read_bytes()
    meta = client.get("/api/runs/current/meta").json
    assert meta["has_reference_events"] is True
    assert meta["reference_event_ids"] == ["eZ", "eA", "eM"]
    assert meta["reference_event_coordinates_m"] == [
        [5000, 5000, 5000], [15000, 5000, 5000], [25000, 5000, 5000]]
    assert meta["event_locs"] == meta["reference_event_coordinates_m"]
    assert (rd / "meta.json").read_bytes() == original
    assert json.loads(original)["event_locs"] == []


def test_viewer_standalone_without_forward_stack_or_arrivals(viewer, tmp_path):
    _, add, _, model, _, runs = viewer
    add("current")
    (model.parents[2] / "output" / "sample" / "arrivals.csv").unlink()
    standalone = tmp_path / "server.py"
    standalone.write_bytes(Path(server.__file__).read_bytes())
    script = """
import builtins
import importlib.util
from pathlib import Path
import sys
original_import = builtins.__import__
def lightweight_import(name, *args, **kwargs):
    if name.split('.')[0] in {'experiment_data', 'projects', 'tomography', 'scipy', 'pykonal'}:
        raise AssertionError('Scientific dependency imported: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = lightweight_import
spec = importlib.util.spec_from_file_location('viewer_server', sys.argv[1])
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)
server.RUNS_DIR = Path(sys.argv[2])
server.EXPERIMENTS_ROOT = Path(sys.argv[3])
server.app.config['TESTING'] = True
client = server.app.test_client()
meta = client.get('/api/runs/current/meta')
assert meta.status_code == 200
assert meta.json['reference_event_ids'] == ['eZ', 'eA', 'eM']
assert client.get('/api/runs/current/hypo_metrics').status_code == 200
"""
    result = subprocess.run([sys.executable, "-I", "-c", script, str(standalone),
                             str(runs), str(model.parents[2])],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_reference_cache_reuses_parsing_and_revalidates_metadata(viewer, monkeypatch):
    client, add, _, model, _, _ = viewer
    add("current")
    calls = []
    original = server._reference_points

    def tracked(*args):
        calls.append(args[0].name)
        return original(*args)

    monkeypatch.setattr(server, "_reference_points", tracked)
    url = "/api/runs/current/meta"
    assert client.get(url).json["has_reference_events"]
    misses = server._file_sha256.cache_info().misses
    assert client.get(url).json["has_reference_events"]
    client.get("/api/runs/current/hypo_metrics")
    assert calls == ["stations.csv", "events.csv"]
    assert server._file_sha256.cache_info().misses == misses
    metadata_path = model.parents[2] / "output" / "sample" / "metadata.json"
    original_metadata = metadata_path.read_text()
    metadata = json.loads(original_metadata)
    metadata["event_ids"] = ["eA", "eZ", "eM"]
    metadata_path.write_text(json.dumps(metadata))
    assert not client.get(url).json["has_reference_events"]
    metadata_path.write_text(original_metadata)
    assert client.get(url).json["has_reference_events"]
    # A valid source update must change the returned coordinates too.
    events = model.parent / "events.csv"
    events.write_text(events.read_text().replace("eZ,5000", "eZ,6000"))
    metadata = json.loads(original_metadata)
    metadata["input_sha256"]["events.csv"] = hashlib.sha256(events.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata))
    assert client.get(url).json["event_locs"][0] == [6000, 5000, 5000]


@pytest.mark.parametrize("field,value", [("event_ids", ["eA", "eZ", "eM"]),
                                          ("station_ids", ["sA", "sB"]),
                                          ("n_events", 2), ("n_stations", 3)])
def test_forward_metadata_ids_and_counts_if_present(viewer, field, value):
    client, add, _, model, _, _ = viewer
    add("current")
    path = model.parents[2] / "output" / "sample" / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata.update(event_ids=["eZ", "eA", "eM"], station_ids=["sB", "sA"], n_events=3, n_stations=2)
    path.write_text(json.dumps(metadata))
    assert client.get("/api/runs/current/meta").json["has_reference_events"]
    metadata[field] = value
    path.write_text(json.dumps(metadata))
    assert not client.get("/api/runs/current/meta").json["has_reference_events"]


@pytest.mark.parametrize("change", ["duplicate", "blank_id", "nan", "outside", "header", "empty", "short_row"])
def test_reference_csv_validation_even_with_matching_hash(viewer, change):
    client, add, _, model, _, _ = viewer
    add("current")
    path = model.parent / "events.csv"
    text = path.read_text()
    replacements = {"duplicate": ("eA", "eZ"), "blank_id": ("eA", "  "),
                    "nan": ("15000", "nan"), "outside": ("15000", "9999999"),
                    "header": ("x_m", "x_km"), "short_row": ("eA,15000", "eA")}
    if change == "empty":
        text = text.splitlines()[0] + "\n"
    else:
        text = text.replace(*replacements[change])
    path.write_text(text)
    metadata_path = model.parents[2] / "output" / "sample" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["input_sha256"]["events.csv"] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata))
    assert not client.get("/api/runs/current/meta").json["has_reference_events"]


def test_optional_generation_hash_is_validated_and_cache_invalidated(viewer):
    client, add, _, model, _, _ = viewer
    add("current")
    path = model.parent / "generation.json"
    path.write_text('{}')
    url = "/api/runs/current/meta"
    assert not client.get(url).json["has_reference_events"]
    metadata_path = model.parents[2] / "output" / "sample" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["input_sha256"]["generation.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata))
    assert client.get(url).json["has_reference_events"]
    path.write_text('{"changed": true}')
    assert not client.get(url).json["has_reference_events"]


def test_hypo_distances_all_iterations_and_cache_updates(viewer):
    client, add, _, model, _, _ = viewer
    rd = add("current")
    _save_viewer_cycle(rd, 1)
    for iteration in (0, 1):
        for event in (0, 1):
            directory = rd / f"iter_{iteration}" / f"event_{event}"
            directory.mkdir(parents=True, exist_ok=True)
            np.savez(directory / "weights.npz",
                     positions=[[20, 0, 0], [event + iteration + 0.25, 0, 0]],
                     weight_values=[0.1, 0.9])
            np.save(directory / "residuals.npy", [[0., 2.], [-2., 0.]])
    # Compact logs without refined positions use fine-grid cell centres.
    directory = rd / "iter_0" / "event_2"
    directory.mkdir(exist_ok=True)
    np.savez(directory / "weights.npz", weight_shape=[3, 1, 1],
             weight_indices=[[2, 0, 0]], weight_values=[1.])
    url = "/api/runs/current/hypo_metrics?iter=0"
    data = client.get(url).json
    assert data["distance_iter"] == [{"event": 0, "dist_m": 2500.},
                                     {"event": 1, "dist_m": 2500.},
                                     {"event": 2, "dist_m": 0.}]
    assert [row["iter"] for row in data["distance_summary"]] == [0, 1]
    assert data["distance_summary"][1]["mean"] == 12500.
    assert data["distance_summary"][1]["n_events"] == 2
    assert data["residual_iter"][0]["rms"] == 2.
    np.savez(rd / "iter_0" / "event_0" / "weights.npz",
             positions=[[0, 0, 0]], weight_values=[1.])
    assert client.get(url).json["distance_iter"][0]["dist_m"] == 0.
    # Cached results must not survive a provenance failure.
    events = model.parent / "events.csv"
    events.write_text(events.read_text() + "\n")
    data = client.get(url).json
    assert data["distance_summary"] == []
    assert data["distance_iter"] == []
    assert data["residual_iter"]


@pytest.mark.parametrize("payload", [None, {"positions": [[float("nan"), 0, 0]], "weight_values": [1.]},
                                    {"positions": [[0, 0, 0]], "weight_values": [float("nan")]},
                                    {"positions": [[0, 0, 0]], "weight_values": [0.]},
                                    {"weight_shape": [3, 1, 1], "weight_indices": np.empty((0, 3), dtype=int)}])
def test_missing_or_invalid_hypotheses_do_not_fabricate_distances(viewer, payload):
    client, add, _, _, _, _ = viewer
    rd = add("current")
    event = rd / "iter_0" / "event_0"
    event.mkdir(exist_ok=True)
    (event / "weights.npz").unlink()
    if payload is not None:
        np.savez(event / "weights.npz", **payload)
    data = client.get("/api/runs/current/hypo_metrics?iter=0").json
    assert data["distance_iter"] == []
    assert data["distance_summary"] == []


@pytest.mark.parametrize("failure", ["missing", "hash", "metadata", "units", "symlink", "count", "stations"])
def test_invalid_reference_provenance_is_not_exposed(viewer, tmp_path, failure):
    client, add, _, model, _, _ = viewer
    rd = add("current")
    events = model.parent / "events.csv"
    output_meta = model.parents[2] / "output" / "sample" / "metadata.json"
    if failure == "missing":
        events.unlink()
    elif failure == "hash":
        events.write_text(events.read_text().replace("5000", "6000"))
    elif failure == "metadata":
        output_meta.write_text("{")
    elif failure == "units":
        metadata = json.loads(output_meta.read_text())
        metadata["units"]["coordinates"] = "km"
        output_meta.write_text(json.dumps(metadata))
    elif failure == "symlink":
        outside = tmp_path / "events.csv"
        outside.write_bytes(events.read_bytes())
        events.unlink()
        events.symlink_to(outside)
    else:
        meta = json.loads((rd / "meta.json").read_text())
        if failure == "count":
            meta["run_params"]["n_events"] = 2
        else:
            meta["station_locs"].reverse()
        (rd / "meta.json").write_text(json.dumps(meta))
    response = client.get("/api/runs/current/meta")
    assert response.status_code == 200
    assert response.json["has_reference_events"] is False
    assert response.json["reference_event_ids"] == []
    assert response.json["event_locs"] == []
    assert client.get("/api/runs/current/hypo_metrics").json["distance_summary"] == []


def test_native_truth_coordinates_and_boundary(viewer):
    client, add, truth, _, _, _ = viewer
    add("current")
    url = "/api/runs/current/slice?type=model&model_type=true&y_km="
    for y_km, block in ((0, 0), (59.999, 0), (60, 1), (120, 1)):
        response = client.get(url + str(y_km))
        assert response.status_code == 200
        data = response.json
        np.testing.assert_array_equal(data["slice"], truth[:, block, :])
        assert data["shape"] == [4, 2]
        assert data["full_shape"] == [4, 2, 2]
        assert data["cell_size"] == 60000
        assert data["origin_m"] == [0, 0, 0]
        assert data["grid_step"] == [1, 1]
        assert data["y_km"] == y_km
        assert data["x_km"] == [30, 90, 150, 210]
        assert data["z_km"] == [30, 90]
        assert data["x_edges_km"] == [0, 60, 120, 180, 240]
        assert data["z_edges_km"] == [0, 60, 120]
        assert data["vmin"] == float(truth[:, block, :].min())
        assert data["vmax"] == float(truth[:, block, :].max())
    assert client.get(url + "60&nx=24&ny=12&nz=12").status_code == 400
    assert client.get("/api/runs/current/slice?type=true_model").status_code == 400
    assert client.get("/api/runs/current/slice?type=model&model_type=true_fine").status_code == 400


def test_inversion_resampling_and_diagnostics_have_physical_axes(viewer):
    client, add, _, _, _, _ = viewer
    add("current")
    base = "/api/runs/current/slice?y_km=60"
    for model_type, value in (("initial", 5100), ("iter", 5200)):
        data = client.get(base + "&type=model&model_type=" + model_type).json
        assert data["full_shape"] == [24, 12, 12]
        assert data["slice"][0][0] == value
        assert data["x_km"][0] == 5
        assert data["z_km"][-1] == 115
        assert data["x_edges_km"][-1] == 240
        assert data["z_edges_km"][-1] == 120
    sampled = client.get(base + "&type=model&nx=256&ny=12&nz=192").json
    assert sampled["shape"] == [256, 192]
    assert sampled["x_edges_km"][-1] == 240
    assert sampled["z_edges_km"][-1] == 120
    for dtype in ("delta_s", "ray_count", "sensitivity_diagonal", "coverage_confidence"):
        data = client.get(base + "&type=" + dtype).json
        assert data["y_km"] == 60
        assert data["x_km"][0] == 5
        assert data["z_edges_km"][-1] == 120
        assert data["cell_size"] == 10000
    assert client.get("/api/runs/current/info").json["n_events"] == 3


@pytest.mark.parametrize("query", ["y_km=-1", "y_km=120.01", "y_km=nan", "y_km=inf",
                                    "y_km=garbage", "nx=0&ny=12&nz=12", "nx=513&ny=12&nz=12",
                                    "nx=512&ny=12&nz=512", "nx=24", "nx=24&ny=nan&nz=12"])
def test_invalid_requests(viewer, query):
    client, add, _, _, _, _ = viewer
    add("current")
    assert client.get("/api/runs/current/slice?type=model&model_type=iter&" + query).status_code == 400
    if query.startswith("y_km="):
        assert client.get("/api/runs/current/slice?type=model&model_type=true&" + query).status_code == 400


def test_run_provenance_latest_and_security(viewer, tmp_path):
    client, add, _, model, digest, runs = viewer
    add("old", source=False)
    add("bad", hash_value="0" * 64)
    add("good")
    add("incomplete", iteration=False)
    assert set(client.get("/api/runs").json) == {"good", "incomplete"}
    assert client.get("/api/runs/latest").json == {"run_id": "incomplete", "max_iter": None}
    for name in ("old", "bad"):
        for suffix in ("meta", "info", "slice?type=model&model_type=true"):
            assert client.get(f"/api/runs/{name}/{suffix}").status_code == 404
    assert client.get("/api/runs/%2e%2e/meta").status_code == 404
    outside = tmp_path / "outside"
    outside.mkdir()
    (runs / "escape").symlink_to(outside, target_is_directory=True)
    assert "escape" not in client.get("/api/runs").json
    assert client.get("/api/runs/escape/meta").status_code == 404
    model.write_bytes(model.read_bytes() + b"changed")
    assert "good" not in client.get("/api/runs").json
    assert client.get("/api/runs/good/slice?type=model&model_type=true").status_code == 404
    assert client.get("/api/runs/latest").json == {"run_id": None, "max_iter": None}


def test_saved_em_metadata_records_provenance_without_resampled_truth(tmp_path):
    model = VelocityModel(np.ones((2, 2, 2)), 1000.)
    source = {"id": "sample", "model_sha256": "a" * 64}
    config = InversionConfig(n_cycles=1, subdivision=1, n_workers=1, runs_dir=str(tmp_path))
    logger = run_em(config, model, np.zeros((3, 2)), [(0, 0, 0), (1000, 0, 0)],
                    PickNoise(), reference_model=model, source_experiment=source)
    meta = json.loads((logger.run_dir / "meta.json").read_text())
    assert meta["source_experiment"] == source
    assert meta["run_params"]["n_events"] == 3
    assert not (logger.run_dir / "true_model.npy").exists()


def test_experiment_symlink_and_id_escape(viewer, tmp_path):
    client, add, _, _, digest, _ = viewer
    add("good")
    rd = add("escaped")
    meta = json.loads((rd / "meta.json").read_text())
    meta["source_experiment"] = {"id": "..", "model_sha256": digest}
    (rd / "meta.json").write_text(json.dumps(meta))
    assert client.get("/api/runs/escaped/meta").status_code == 404
    model = tmp_path / "experiments" / "input" / "sample" / "model.npz"
    copy = tmp_path / "copied.npz"
    copy.write_bytes(model.read_bytes())
    model.unlink()
    model.symlink_to(copy)
    assert client.get("/api/runs/good/meta").status_code == 404


def test_only_marked_cycles_are_published_by_every_endpoint(viewer):
    client, add, _, _, _, _ = viewer
    rd = add("current")
    _save_viewer_cycle(rd, 3)
    partial = _save_viewer_cycle(rd, 1, complete=False)
    for iteration in (0, 1, 3):
        event = rd / f"iter_{iteration}" / "event_0"
        np.savez(event / "weights.npz", weight_shape=[24, 12, 12],
                 weight_indices=[[0, 0, 0]], positions=[[0, 0, 0]], weight_values=[1.])
        np.save(event / "residuals.npy", [[0., 2.], [-2., 0.]])
        (event / "weight_0").mkdir()
    (rd / "quality.jsonl").write_text("".join(json.dumps(
        {"iter": i, "avg_abs_pct_dev": 1., "rms_m_s": 2.}) + "\n" for i in (0, 1, 3)))
    meta = json.loads((rd / "meta.json").read_text())
    meta["run_params"]["n_cycles"] = 8
    (rd / "meta.json").write_text(json.dumps(meta))
    base = "/api/runs/current/"
    info = client.get(base + "info").json
    assert info["iterations"] == [0, 3]
    assert info["completed_iterations"] == 2  # not max + 1
    assert info["planned_iterations"] == 8
    assert client.get("/api/runs/summary").json == [
        {"id": "current", "run_name": "current", "iterations": [0, 3], "completed_iterations": 2, "planned_iterations": 8}]
    assert client.get("/api/runs").json == ["current"]
    assert client.get("/api/runs/latest").json == {"run_id": "current", "max_iter": 3}
    assert client.get(base + "iters_list").json == ["iter_0", "iter_3"]
    for endpoint in ("timing", "quality"):
        assert [row["iter"] for row in client.get(base + endpoint).json] == [0, 3]
    for endpoint in ("events_list", "weights_list"):
        assert client.get(base + endpoint + "?iter=1").json == []
        assert client.get(base + endpoint + "?iter=0").json
    for dtype in ("model", "weights", "G", "ray_count", "delta_s", "sensitivity_diagonal", "coverage_confidence"):
        assert client.get(base + f"slice?type={dtype}&iter=1").status_code == 404
    metrics = client.get(base + "hypo_metrics?iter=1").json
    assert metrics["distance_iter"] == metrics["residual_iter"] == []
    assert [row["iter"] for row in metrics["residual_summary"]] == [0, 3]
    # Publication alone must invalidate the metrics cache, without changing event files.
    (partial / "complete.json").write_text(json.dumps({"iter": 1, "viewer_completion_protocol": 1}))
    metrics = client.get(base + "hypo_metrics?iter=1").json
    assert metrics["residual_iter"]
    assert [row["iter"] for row in metrics["distance_summary"]] == [0, 1, 3]


def test_no_complete_cycles_defaults_to_initial_and_latest_null(viewer):
    client, add, _, _, _, _ = viewer
    rd = add("current", iteration=False)
    _save_viewer_cycle(rd, 0, complete=False)
    assert client.get("/api/runs/current/info").json["iterations"] == []
    assert client.get("/api/runs/summary").json == [
        {"id": "current", "run_name": "current", "iterations": [], "completed_iterations": 0, "planned_iterations": None}]
    assert client.get("/api/runs/latest").json == {"run_id": "current", "max_iter": None}
    base = "/api/runs/current/slice?type=model"
    assert client.get(base).json["slice"][0][0] == 5100.
    for model_type in ("initial", "true"):
        assert client.get(base + "&model_type=" + model_type).status_code == 200
    assert client.get(base + "&model_type=iter").status_code == 404
    assert client.get("/api/runs/current/timing").json == []


@pytest.mark.parametrize("marker", ["{", "[]", '{"iter": 9, "viewer_completion_protocol": 1}',
                                    '{"iter": 0, "viewer_completion_protocol": 2}'])
def test_invalid_completion_marker_never_falls_back_to_timing(viewer, marker):
    client, add, _, _, _, _ = viewer
    rd = add("current")
    (rd / "iter_0" / "complete.json").write_text(marker)
    assert client.get("/api/runs/current/info").json["iterations"] == []


def test_legacy_completion_requires_unique_valid_timing_not_model_directories(viewer):
    client, add, _, _, _, _ = viewer
    rd = add("legacy", protocol=None)
    _save_viewer_cycle(rd, 4)
    model_only = rd / "iter_7"
    model_only.mkdir()
    np.save(model_only / "model.npy", np.ones((2, 2, 2)))
    assert client.get("/api/runs/legacy/info").json["iterations"] == [0, 4]
    with (rd / "timing.jsonl").open("a") as stream:
        stream.write('{"iter": 4, "elapsed_s": 2}\n')
        stream.write('null\n{"iter": 7, "elapsed_s": NaN}\n{"iter": true, "elapsed_s": 1}\n{')
    assert client.get("/api/runs/legacy/info").json["iterations"] == [0]
    assert [row["iter"] for row in client.get("/api/runs/legacy/timing").json] == [0]
    assert client.get("/api/runs/legacy/slice?iter=4&model_type=iter").status_code == 404
    (rd / "timing.jsonl").unlink()
    assert client.get("/api/runs/legacy/info").json["iterations"] == []


@pytest.mark.parametrize("missing", ["model.npy", "delta_s.npy", "sensitivity_diagonal.npy",
                                      "coverage_confidence.npy", "event_0/weights.npz",
                                      "event_2/residuals.npy", "station_fields.npy", "event_0/misfit.npy",
                                      "event_0/weight_0/G_stations_sparse.npz", "event_0/weight_0/ray_count.npy",
                                      "quality.jsonl"])
def test_legacy_requires_configured_artifacts(viewer, missing):
    client, add, _, _, _, _ = viewer
    rd = add("legacy", protocol=None)
    meta = json.loads((rd / "meta.json").read_text())
    # An underspecified older layout retains detailed cold-read compatibility checks.
    meta["run_params"].update(log_g_per_weight=True, save_timefields=True,
                              save_misfit=True, viewer_quality_expected=True)
    (rd / "meta.json").write_text(json.dumps(meta))
    directory = rd / "iter_0"
    np.save(directory / "station_fields.npy", np.ones((2, 2, 2, 2)))
    for event in directory.glob("event_*"):
        np.savez(event / "weights.npz", weight_shape=[2, 2, 2],
                 weight_indices=[[0, 0, 0]], weight_values=[1.])
        np.save(event / "misfit.npy", np.ones((2, 2, 2)))
        weight = event / "weight_0"
        weight.mkdir()
        np.save(weight / "ray_count.npy", np.ones((2, 2, 2)))
        np.savez(weight / "G_stations_sparse.npz", shape=[2, 2, 2], offsets=[0, 0],
                 coords=np.empty((0, 3), dtype=int), values=[])
    (rd / "quality.jsonl").write_text('{"iter": 0, "avg_abs_pct_dev": 1}\n')
    # Construct a damaged old snapshot before its first read. Published event
    # outputs are immutable; warm invalidation watches end saves, not event history.
    (rd / missing if missing == "quality.jsonl" else directory / missing).unlink()
    assert client.get("/api/runs/legacy/info").json["iterations"] == []


def test_legacy_pre_diagnostics_format_and_truncated_array(viewer):
    client, add, _, _, _, _ = viewer
    rd = add("legacy", protocol=None)
    for name in ("sensitivity_diagonal", "coverage_confidence"):
        (rd / "iter_0" / f"{name}.npy").unlink()
    assert client.get("/api/runs/legacy/info").json["iterations"] == [0]
    (rd / "iter_0" / "delta_s.npy").write_bytes(b"partial array")
    assert client.get("/api/runs/legacy/info").json["iterations"] == []


def test_sparse_slices_allocate_only_plane_and_keep_refined_hypocenters(viewer, monkeypatch):
    client, add, _, _, _, _ = viewer
    rd = add("current")
    event = rd / "iter_0" / "event_0"
    shape = [8, 10000000, 6]
    indices = np.array([[2, 5000000, 3], [1, 4999999, 4], [7, 9999999, 5]])
    positions = [[2.25, 5000000.1, 3.4], [1.1, 4999999.2, 4.3], [7., 9999999., 5.]]
    np.savez(event / "weights.npz", weight_shape=shape, weight_indices=indices,
             weight_values=[0.75, 0.2, 0.05], positions=positions)
    weight = event / "weight_0"
    weight.mkdir()
    np.savez(weight / "G_stations_sparse.npz", shape=shape, offsets=[0, 2, 3],
             coords=indices, values=[1.25, 2., 3.])
    zeros = np.zeros
    allocations = []

    def plane_only(size, *args, **kwargs):
        if isinstance(size, (tuple, list)):
            assert len(size) <= 2, "Sparse slice allocated a full volume"
            allocations.append(tuple(size))
        return zeros(size, *args, **kwargs)

    monkeypatch.setattr(server.np, "zeros", plane_only)
    base = "/api/runs/current/slice?iter=0&event=0&y_km=60&type="
    for dtype, value in (("weights", 0.75), ("G", 1.25)):
        data = client.get(base + dtype).json
        assert data["shape"] == [8, 6]
        assert data["full_shape"] == shape
        assert data["slice"][2][3] == value
        assert np.count_nonzero(data["slice"]) == 1
        assert data["hypocenters"][0] == {"coord": positions[0], "weight": 0.75}
        assert len(data["hypocenters"]) == (3 if dtype == "weights" else 1)
        assert data["x_edges_km"][-1] == 240
        assert data["z_edges_km"][-1] == 120
        assert data["y_km"] == 60
    assert allocations == [(8, 6), (8, 6)]
    assert np.count_nonzero(client.get(base + "G&station=1").json["slice"]) == 0
    assert client.get(base + "G&station=2").json["slice"] is None
    boundary = client.get(base.replace("y_km=60", "y_km=120") + "weights").json
    assert boundary["slice"][7][5] == 0.05


def test_unavailable_truth_has_false_flag_and_no_none_load(viewer, monkeypatch):
    client, add, _, _, _, _ = viewer
    add("current", source=False)
    # Isolate truth availability from the existing provenance-based run eligibility rule.
    monkeypatch.setattr(server, "_eligible", lambda rd: rd.is_dir())
    assert client.get("/api/runs/current/info").json["has_true_model"] is False
    assert client.get("/api/runs/current/slice?model_type=true").status_code == 404


@pytest.mark.parametrize("protocol", [2, 0, True, "1"])
def test_unknown_completion_protocol_is_not_inferred_as_legacy(viewer, protocol):
    client, add, _, _, _, _ = viewer
    add("current", protocol=protocol)
    assert client.get("/api/runs/current/info").json["iterations"] == []


def test_legacy_source_requires_quality_unless_explicitly_disabled(viewer):
    client, add, _, _, _, _ = viewer
    rd = add("legacy", protocol=None)
    (rd / "quality.jsonl").unlink()
    assert client.get("/api/runs/legacy/info").json["iterations"] == []
    meta = json.loads((rd / "meta.json").read_text())
    meta["run_params"]["viewer_quality_expected"] = False
    (rd / "meta.json").write_text(json.dumps(meta))
    assert client.get("/api/runs/legacy/info").json["iterations"] == [0]


def test_noncanonical_marker_directory_does_not_inflate_completion_count(viewer):
    client, add, _, _, _, _ = viewer
    rd = add("current")
    alias = rd / "iter_00"
    alias.mkdir()
    (alias / "complete.json").write_bytes((rd / "iter_0/complete.json").read_bytes())
    assert client.get("/api/runs/current/info").json["completed_iterations"] == 1


def test_legacy_requires_weight_outputs_even_if_directory_was_never_written(viewer):
    client, add, _, _, _, _ = viewer
    rd = add("legacy", protocol=None)
    event = rd / "iter_0/event_0"
    np.savez(event / "weights.npz", weight_shape=[24, 12, 12],
             weight_indices=[[0, 0, 0]], weight_values=[1.])
    assert client.get("/api/runs/legacy/info").json["iterations"] == []
    weight = event / "weight_0"
    weight.mkdir()
    np.save(weight / "ray_count.npy", np.ones((24, 12, 12)))
    assert client.get("/api/runs/legacy/info").json["iterations"] == [0]


def test_current_legacy_summary_is_bounded_and_caches_layout_and_readability(viewer, monkeypatch):
    client, add, _, _, _, _ = viewer
    rd = add("legacy", protocol=None)
    _save_viewer_cycle(rd, 4)
    meta = json.loads((rd / "meta.json").read_text())
    # Declaring many events must not make summary work proportional to event count.
    meta["run_params"].update(coverage_damping_power=1, n_events=100000, n_cycles=7,
                              run_name="Immediate dropdown name")
    (rd / "meta.json").write_text(json.dumps(meta))
    loads, layouts = [], []
    original_load, original_layout, original_glob = np.load, server._legacy_features, Path.glob

    def load(path, *args, **kwargs):
        assert Path(path).suffix == ".npy", "Summary reopened an event NPZ"
        loads.append(Path(path).name)
        return original_load(path, *args, **kwargs)

    def layout(*args):
        layouts.append(args[0])
        return original_layout(*args)

    def no_history_glob(path, pattern):
        assert not pattern.startswith("iter_*/"), "Summary traversed event history"
        return original_glob(path, pattern)

    monkeypatch.setattr(server.np, "load", load)
    monkeypatch.setattr(server, "_legacy_features", layout)
    monkeypatch.setattr(Path, "glob", no_history_glob)
    cold = client.get("/api/runs/summary").json
    assert cold == [{"id": "legacy", "run_name": "Immediate dropdown name", "iterations": [0, 4],
                     "completed_iterations": 2, "planned_iterations": 7}]
    assert len(loads) == 8  # model, delta, and two diagnostics, for each timed cycle
    assert client.get("/api/runs/summary").json == cold
    assert len(loads) == 8 and len(layouts) == 1
    np.save(rd / "iter_4/delta_s.npy", np.full((24, 12, 12), 2.))
    assert client.get("/api/runs/summary").json == cold
    assert len(loads) == 9 and len(layouts) == 1
    (rd / "quality.jsonl").write_text('{"iter": 0, "avg_abs_pct_dev": 1}\n')
    assert client.get("/api/runs/summary").json[0]["iterations"] == [0]
    assert len(loads) == 9 and len(layouts) == 1


@pytest.mark.parametrize("artifact", ["model.npy", "delta_s.npy", "sensitivity_diagonal.npy",
                                       "coverage_confidence.npy", "station_fields.npy", "quality.jsonl"])
def test_cached_current_legacy_completion_tracks_late_and_rewritten_end_saves(viewer, artifact):
    client, add, _, _, _, _ = viewer
    rd = add("legacy", protocol=None)
    meta = json.loads((rd / "meta.json").read_text())
    meta["run_params"].update(coverage_damping_power=1, save_timefields=True, viewer_quality_expected=True)
    (rd / "meta.json").write_text(json.dumps(meta))
    np.save(rd / "iter_0/station_fields.npy", np.ones((2, 2, 2, 2)))
    path = rd / artifact if artifact == "quality.jsonl" else rd / "iter_0" / artifact
    saved = path.read_bytes()
    path.unlink()
    url = "/api/runs/summary"
    assert client.get(url).json[0]["iterations"] == []
    assert client.get(url).json[0]["iterations"] == []
    path.write_bytes(saved)
    assert client.get(url).json[0]["iterations"] == [0]
    path.write_bytes(b"partial rewrite")
    assert client.get(url).json[0]["iterations"] == []
    path.write_bytes(saved)
    assert client.get(url).json[0]["iterations"] == [0]


def test_cached_current_legacy_completion_detects_new_cycles_and_meta_changes(viewer):
    client, add, _, _, _, _ = viewer
    rd = add("legacy", protocol=None)
    meta = json.loads((rd / "meta.json").read_text())
    meta["run_params"]["coverage_damping_power"] = 1
    (rd / "meta.json").write_text(json.dumps(meta))
    url = "/api/runs/summary"
    assert client.get(url).json[0]["iterations"] == [0]
    _save_viewer_cycle(rd, 3)
    assert client.get(url).json[0]["iterations"] == [0, 3]
    meta.update(run_name="Renamed run")
    meta["run_params"]["n_cycles"] = 9
    (rd / "meta.json").write_text(json.dumps(meta))
    assert client.get(url).json[0]["run_name"] == "Renamed run"
    assert client.get(url).json[0]["planned_iterations"] == 9
    (rd / "iter_3/coverage_confidence.npy").unlink()
    assert client.get(url).json[0]["iterations"] == [0]
    assert client.get("/api/runs/legacy/slice?type=delta_s&iter=3").status_code == 404
    np.save(rd / "iter_3/coverage_confidence.npy", np.ones((24, 12, 12)))
    assert client.get(url).json[0]["iterations"] == [0, 3]


@pytest.mark.parametrize("layout", ["npy", "npz_c", "npz_fortran", "sparse_npz"])
def test_legacy_standalone_G_slices_read_only_plane(viewer, monkeypatch, layout):
    client, add, _, _, _, _ = viewer
    rd = add("current")
    event = rd / "iter_0/event_0"
    weight = event / "weight_0"
    weight.mkdir()
    dense = np.arange(8 * 3 * 6, dtype=np.float32).reshape(8, 3, 6)
    stem = weight / "G_station_0"
    if layout == "npy":
        np.save(stem.with_suffix(".npy"), dense)
    elif layout == "sparse_npz":
        coords = np.column_stack(np.nonzero(dense))
        np.savez_compressed(stem.with_suffix(".npz"), shape=dense.shape, coords=coords,
                            values=dense[tuple(coords.T)])
    else:
        values = np.asfortranarray(dense) if layout == "npz_fortran" else dense
        np.savez_compressed(stem.with_suffix(".npz"), G=values)
    np.savez(event / "weights.npz", weight_shape=dense.shape, weight_indices=[[2, 1, 3]],
             positions=[[2.25, 1.1, 3.4]], weight_values=[1.])
    empty, zeros, load = np.empty, np.zeros, np.load

    def plane_allocation(allocator):
        def allocate(shape, *args, **kwargs):
            if isinstance(shape, (list, tuple)):
                assert len(shape) <= 2, "Standalone G slice allocated a dense volume"
            return allocator(shape, *args, **kwargs)
        return allocate

    def no_dense_npz(path, *args, **kwargs):
        if layout in ("npz_c", "npz_fortran"):
            assert Path(path).name != "G_station_0.npz", "Dense NPZ must be streamed, not expanded"
        return load(path, *args, **kwargs)

    monkeypatch.setattr(server.np, "empty", plane_allocation(empty))
    monkeypatch.setattr(server.np, "zeros", plane_allocation(zeros))
    monkeypatch.setattr(server.np, "load", no_dense_npz)
    for y_km, y in ((0, 0), (60, 1), (120, 2)):
        data = client.get(f"/api/runs/current/slice?type=G&event=0&weight=0&station=0&y_km={y_km}").json
        np.testing.assert_array_equal(data["slice"], dense[:, y, :])
        assert data["full_shape"] == [8, 3, 6]
        assert data["shape"] == [8, 6]
        assert data["x_edges_km"][-1] == 240 and data["z_edges_km"][-1] == 120
        assert data["hypocenters"] == [{"coord": [2.25, 1.1, 3.4], "weight": 1.}]
    assert client.get("/api/runs/current/slice?type=G&station=1").json["slice"] is None
    np.testing.assert_array_equal(server._load_G_station(stem, 1), dense[:, 1, :])


def test_old_legacy_completion_accepts_standalone_G_and_late_alternate_extension(viewer):
    client, add, _, _, _, _ = viewer
    rd = add("legacy", protocol=None)
    meta = json.loads((rd / "meta.json").read_text())
    meta["run_params"]["log_G_per_weight"] = True
    (rd / "meta.json").write_text(json.dumps(meta))
    event = rd / "iter_0/event_0"
    np.savez(event / "weights.npz", weight_shape=[2, 2, 2], weight_indices=[[0, 0, 0]], weight_values=[1.])
    weight = event / "weight_0"
    weight.mkdir()
    np.save(weight / "ray_count.npy", np.ones((2, 2, 2)))
    np.save(weight / "G_station_0.npy", np.ones((2, 2, 2)))
    url = "/api/runs/summary"
    assert client.get(url).json[0]["iterations"] == []  # station 1 is still missing
    np.savez_compressed(weight / "G_station_1.npz", G=np.ones((2, 2, 2)))
    assert client.get(url).json[0]["iterations"] == [0]
    assert not (weight / "G_stations_sparse.npz").exists()
    assert client.get("/api/runs/legacy/slice?type=G&station=1").json["slice"] == [[1., 1.], [1., 1.]]


@pytest.fixture
def asymmetric_viewer(viewer):
    client, add, _, model_path, _, _ = viewer
    rd = add("xy")
    coarse = (4000. + 1000. * np.arange(3)[:, None, None]
              + 100. * np.arange(4)[None, :, None]
              + 10. * np.arange(2)[None, None, :])
    truth = (3000. + 1000. * np.arange(5)[:, None, None]
             + 100. * np.arange(3)[None, :, None]
             + 10. * np.arange(4)[None, None, :])
    np.savez(model_path, velocity=truth, cell_size_m=np.array(2000.), origin_m=np.zeros(3))
    meta = json.loads((rd / "meta.json").read_text())
    meta["source_experiment"]["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    meta["grid_info"] = {"coarse_shape": [3, 4, 2], "coarse_cell_size": 2000.,
                         "coarse_side_m": [6000., 12000., 10000.], "fine_cell_size": 2000.}
    (rd / "meta.json").write_text(json.dumps(meta))
    np.save(rd / "initial_model.npy", coarse)
    for name, values in (("model", coarse + 500), ("delta_s", coarse - 5000),
                         ("sensitivity_diagonal", coarse * 2),
                         ("coverage_confidence", coarse / 10000)):
        np.save(rd / "iter_0" / f"{name}.npy", values)
    return client, rd, coarse, truth


@pytest.mark.parametrize("model_type", ["initial", "iter"])
def test_xy_model_native_layers_axes_and_boundaries(asymmetric_viewer, model_type):
    client, _, coarse, _ = asymmetric_viewer
    values = coarse if model_type == "initial" else coarse + 500
    base = f"/api/runs/xy/slice?type=model&model_type={model_type}&plane=xy"
    assert client.get(base).json["z_km"] == 2.5
    for depth, layer in ((0, 0), (4.999, 0), (5, 1), (10, 1)):
        response = client.get(base + f"&z_km={depth}")
        assert response.status_code == 200
        data = response.json
        np.testing.assert_array_equal(data["slice"], values[:, :, layer])
        assert data["shape"] == [3, 4]
        assert data["full_shape"] == [3, 4, 2]
        assert data["slice_plane"] == "xy"
        assert data["z_km"] == depth
        assert data["x_km"] == [1, 3, 5]
        assert data["y_km"] == [1.5, 4.5, 7.5, 10.5]
        assert data["x_edges_km"] == [0, 2, 4, 6]
        assert data["y_edges_km"] == [0, 3, 6, 9, 12]
        assert "z_edges_km" not in data
        assert data["origin_m"] == [0, 0, 0]
        assert data["cell_size"] == 2000
        assert data["grid_step"] == [1, 1]
        assert data["vmin"] == values[:, :, layer].min()
        assert data["vmax"] == values[:, :, layer].max()
    # Explicit XZ remains identical to the legacy default, including its axes.
    assert client.get(base.replace("&plane=xy", "") + "&y_km=6").json == client.get(
        base.replace("plane=xy", "plane=xz") + "&y_km=6").json


@pytest.mark.parametrize("dtype,scale,offset", [
    ("delta_s", 1, -5000), ("sensitivity_diagonal", 2, 0), ("coverage_confidence", 0.0001, 0),
])
def test_xy_diagnostics_keep_native_values_and_geometry(asymmetric_viewer, dtype, scale, offset):
    client, _, coarse, _ = asymmetric_viewer
    base = f"/api/runs/xy/slice?type={dtype}&plane=xy"
    for depth, layer in ((0, 0), (4.999, 0), (5, 1), (10, 1)):
        response = client.get(base + f"&z_km={depth}")
        assert response.status_code == 200
        data = response.json
        np.testing.assert_allclose(data["slice"], coarse[:, :, layer] * scale + offset)
        assert data["shape"] == [3, 4]
        assert data["full_shape"] == [3, 4, 2]
        assert data["z_km"] == depth
        assert data["slice_plane"] == "xy"
        assert data["x_edges_km"] == [0, 2, 4, 6]
        assert data["y_edges_km"] == [0, 3, 6, 9, 12]


def test_xy_truth_stays_on_independent_native_grid(asymmetric_viewer):
    client, _, _, truth = asymmetric_viewer
    base = "/api/runs/xy/slice?type=model&model_type=true&plane=xy"
    assert client.get(base).json["z_km"] == 1
    for depth, layer in ((0, 0), (1.999, 0), (2, 1), (4, 2), (6, 3), (8, 3)):
        response = client.get(base + f"&z_km={depth}")
        assert response.status_code == 200
        data = response.json
        np.testing.assert_array_equal(data["slice"], truth[:, :, layer])
        assert data["shape"] == [5, 3]
        assert data["full_shape"] == [5, 3, 4]
        assert data["z_km"] == depth
        assert data["slice_plane"] == "xy"
        assert data["x_km"] == [1, 3, 5, 7, 9]
        assert data["y_km"] == [1, 3, 5]
        assert data["x_edges_km"] == [0, 2, 4, 6, 8, 10]
        assert data["y_edges_km"] == [0, 2, 4, 6]
        assert data["cell_size"] == 2000
        assert data["grid_step"] == [1, 1]
    for query in ("nx=5", "nx=5&ny=3&nz=4"):
        assert client.get(base + "&" + query).status_code == 400
    assert client.get(base + "&z_km=8.001").status_code == 400


@pytest.mark.parametrize("subdivision", [False, True])
def test_xy_nearest_resampling_and_grid_steps(asymmetric_viewer, subdivision):
    client, rd, coarse, _ = asymmetric_viewer
    base = "/api/runs/xy/slice?type=model&model_type=initial&plane=xy"
    if subdivision:
        meta = json.loads((rd / "meta.json").read_text())
        meta["run_params"]["subdivision"] = 2
        (rd / "meta.json").write_text(json.dumps(meta))
        target = (6, 8, 4)
    else:
        target = (6, 12, 8)
        base += "&nx=6&ny=12&nz=8"
    ix = ((np.arange(target[0]) + 0.5) * coarse.shape[0] / target[0]).astype(int)
    iy = ((np.arange(target[1]) + 0.5) * coarse.shape[1] / target[1]).astype(int)
    for depth in (0, 4.999, 5, 10):
        response = client.get(base + f"&z_km={depth}")
        assert response.status_code == 200
        data = response.json
        layer = min(int(depth / 10 * target[2]), target[2] - 1)
        iz = int((layer + 0.5) * coarse.shape[2] / target[2])
        np.testing.assert_array_equal(data["slice"], coarse[ix[:, None], iy[None, :], iz])
        assert data["shape"] == list(target[:2])
        assert data["full_shape"] == list(target)
        assert data["grid_step"] == [2, 2 if subdivision else 3]
        assert data["z_km"] == depth
        assert data["x_edges_km"][-1] == 6
        assert data["y_edges_km"][-1] == 12


@pytest.mark.parametrize("positive", [False, True])
def test_xy_trilinear_matches_cell_centred_slowness_formula(asymmetric_viewer, monkeypatch, positive):
    client, rd, coarse, _ = asymmetric_viewer
    values = coarse if positive else coarse - 5000
    np.save(rd / "initial_model.npy", values)
    meta = json.loads((rd / "meta.json").read_text())
    meta["run_params"]["slowness_interpolation"] = "trilinear"
    (rd / "meta.json").write_text(json.dumps(meta))
    target = (7, 9, 5)
    coords = [np.clip((np.arange(n) + 0.5) * size / n - 0.5, 0, size - 1)
              for size, n in zip(values.shape, target)]
    zeros = server.np.zeros

    def plane_only(shape, *args, **kwargs):
        assert not isinstance(shape, tuple) or len(shape) <= 2, "Resampling allocated a volume"
        return zeros(shape, *args, **kwargs)

    monkeypatch.setattr(server.np, "zeros", plane_only)
    base = "/api/runs/xy/slice?type=model&model_type=initial&plane=xy&nx=7&ny=9&nz=5"
    for depth in (0, 4.5, 10):
        response = client.get(base + f"&z_km={depth}")
        assert response.status_code == 200
        data = response.json
        layer = min(int(depth / 10 * target[2]), target[2] - 1)
        expected = np.empty(target[:2])
        for x in range(target[0]):
            for y in range(target[1]):
                position = (coords[0][x], coords[1][y], coords[2][layer])
                lower = np.floor(position).astype(int)
                fraction = np.asarray(position) - lower
                result = 0.
                for corner in np.ndindex(2, 2, 2):
                    index = tuple(min(lower[a] + corner[a], values.shape[a] - 1) for a in range(3))
                    weight = np.prod([fraction[a] if corner[a] else 1 - fraction[a] for a in range(3)])
                    result += weight * (1 / values[index] if positive else values[index])
                expected[x, y] = 1 / result if positive else result
        np.testing.assert_allclose(data["slice"], expected.astype(np.float32), rtol=1e-7)
        # The original XZ interpolation on a transpose view must be equivalent.
        equivalent, _, _ = server._model_slice(values.transpose(0, 2, 1), layer, meta,
                                               (target[0], target[2], target[1]))
        np.testing.assert_array_equal(data["slice"], equivalent)
        assert data["full_shape"] == list(target)
        assert data["shape"] == [7, 9]
        assert data["grid_step"] == [1, 1]
        assert data["z_km"] == depth
        np.testing.assert_allclose(data["x_km"], (np.arange(7) + 0.5) * 6 / 7)
        np.testing.assert_allclose(data["y_km"], (np.arange(9) + 0.5) * 12 / 9)


@pytest.mark.parametrize("query", [
    "plane=yz", "plane=", "plane=XY", "plane=xy&y_km=0", "plane=xz&z_km=0", "z_km=0",
    "plane=xy&z_km=-0.001", "plane=xy&z_km=10.001", "plane=xy&z_km=nan",
    "plane=xy&z_km=inf", "plane=xy&z_km=-inf", "plane=xy&z_km=garbage", "plane=xy&z_km=",
    "plane=xy&nx=0&ny=4&nz=2", "plane=xy&nx=3&ny=513&nz=2",
    "plane=xy&nx=3&ny=4&nz=513", "plane=xy&nx=3", "plane=xy&nx=3&ny=nan&nz=2",
])
def test_xy_invalid_plane_coordinates_and_target_shapes(asymmetric_viewer, query):
    client, _, _, _ = asymmetric_viewer
    assert client.get("/api/runs/xy/slice?type=model&model_type=initial&" + query).status_code == 400
    if not any(name + "=" in query for name in ("nx", "ny", "nz")):
        for dtype in ("delta_s", "sensitivity_diagonal", "coverage_confidence"):
            assert client.get(f"/api/runs/xy/slice?type={dtype}&" + query).status_code == 400
        assert client.get("/api/runs/xy/slice?type=model&model_type=true&" + query).status_code == 400


@pytest.mark.parametrize("dtype", ["weights", "G", "ray_count"])
def test_xy_sparse_and_ray_types_remain_xz_only(asymmetric_viewer, dtype):
    client, _, _, _ = asymmetric_viewer
    assert client.get(f"/api/runs/xy/slice?type={dtype}&plane=xy&z_km=0").status_code == 400


def test_xy_pixel_limit_uses_shown_axes_for_explicit_and_default_grids(asymmetric_viewer):
    client, rd, _, _ = asymmetric_viewer
    base = "/api/runs/xy/slice?type=model&model_type=initial"
    for plane, shape, status in (("xy", (256, 2, 512), 200), ("xz", (256, 2, 512), 400),
                                 ("xy", (256, 512, 2), 400), ("xz", (256, 512, 2), 200),
                                 ("xy", (256, 256, 1), 200), ("xy", (256, 257, 1), 400)):
        query = f"&plane={plane}&nx={shape[0]}&ny={shape[1]}&nz={shape[2]}"
        assert client.get(base + query).status_code == status
    meta = json.loads((rd / "meta.json").read_text())
    meta["run_params"]["subdivision"] = 74
    (rd / "meta.json").write_text(json.dumps(meta))
    assert client.get(base + "&plane=xy").status_code == 400
    assert client.get(base + "&plane=xz").status_code == 200
    assert client.get(base + "&plane=xy&nx=3&ny=4&nz=2").status_code == 200


def test_xy_publication_gating_and_initial_before_completion(asymmetric_viewer):
    client, rd, _, _ = asymmetric_viewer
    (rd / "iter_0/complete.json").unlink()
    base = "/api/runs/xy/slice?plane=xy&z_km=0"
    for model_type in ("initial", "true"):
        assert client.get(base + f"&type=model&model_type={model_type}").status_code == 200
    np.testing.assert_array_equal(client.get(base + "&type=model").json["slice"],
                                  np.load(rd / "initial_model.npy")[:, :, 0])
    for query in ("type=model&model_type=iter", "type=delta_s", "type=sensitivity_diagonal",
                  "type=coverage_confidence"):
        assert client.get(base + "&" + query).status_code == 404
    (rd / "iter_0/complete.json").write_text(json.dumps({"iter": 0, "viewer_completion_protocol": 1}))
    for query in ("type=model&model_type=iter", "type=delta_s", "type=sensitivity_diagonal",
                  "type=coverage_confidence"):
        assert client.get(base + "&" + query).status_code == 200
        assert client.get(base + "&" + query + "&iter=1").status_code == 404


def _ray_status_fixture(failed=False):
    return {
        "schema_version": 1, "expected": 6, "reached": 5 if failed else 6,
        "failed": int(failed), "used": 4 if failed else 6,
        "n_events": 3, "n_hypotheses": 3, "events_with_failed_rays": int(failed),
        "hypotheses_with_insufficient_stations": int(failed),
        "failures": [{"event": 2, "hypothesis": 0, "stations": [1]}] if failed else [],
    }


@pytest.mark.parametrize("failed", [False, True])
def test_ray_status_returns_primary_trace_totals_only_after_publication(viewer, failed):
    client, add, *_ = viewer
    rd = add("current")
    status = _ray_status_fixture(failed)
    (rd / "iter_0/ray_status.json").write_text(json.dumps(status))
    response = client.get("/api/runs/current/ray_status?iter=0")
    assert response.status_code == 200
    assert response.json == {**status, "available": True, "iter": 0}
    (rd / "iter_0/complete.json").unlink()
    assert client.get("/api/runs/current/ray_status?iter=0").json == {
        "available": False, "reason": "not_completed", "iter": 0,
    }


def test_old_logs_do_not_invent_ray_success_from_coverage(viewer):
    client, add, *_ = viewer
    rd = add("old", protocol=None)
    np.save(rd / "iter_0/ray_count.npy", np.full((24, 12, 12), 99))
    assert client.get("/api/runs/old/ray_status?iter=0").json == {
        "available": False, "reason": "not_recorded", "iter": 0,
    }


@pytest.mark.parametrize("changes", [
    {"expected": 7}, {"reached": -1}, {"used": 7}, {"expected": True},
    {"schema_version": 2}, {"failures": []},
    {"failures": [{"event": 2, "hypothesis": 0, "stations": [1, 1]}]},
    {"failures": [{"event": 2, "hypothesis": 0, "stations": [-1]}]},
    {"events_with_failed_rays": 4}, {"hypotheses_with_insufficient_stations": 4},
])
def test_invalid_ray_accounting_is_explicitly_unavailable(viewer, changes):
    client, add, *_ = viewer
    rd = add("current")
    status = {**_ray_status_fixture(True), **changes}
    (rd / "iter_0/ray_status.json").write_text(json.dumps(status))
    assert client.get("/api/runs/current/ray_status?iter=0").json == {
        "available": False, "reason": "invalid_artifact", "iter": 0,
    }


def test_ray_status_rejects_partial_json_invalid_iteration_and_symlinks(viewer, tmp_path):
    client, add, *_ = viewer
    rd = add("current")
    path = rd / "iter_0/ray_status.json"
    path.write_text('{"expected":')
    assert client.get("/api/runs/current/ray_status").json["reason"] == "invalid_artifact"
    assert client.get("/api/runs/current/ray_status?iter=-1").status_code == 400
    path.unlink()
    target = tmp_path / "outside.json"
    target.write_text(json.dumps(_ray_status_fixture()))
    path.symlink_to(target)
    assert client.get("/api/runs/current/ray_status").status_code == 404
