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

    def add_run(name, *, source=True, hash_value=digest, iteration=True):
        rd = runs / name
        rd.mkdir()
        meta = {"run_params": {"subdivision": 1, "n_events": 3}, "grid_info": grid,
                "station_locs": [[0, 0, 0], [10000, 0, 0]], "event_locs": [],
                "source_experiment": {"id": "sample", "model_sha256": hash_value} if source else None}
        (rd / "meta.json").write_text(json.dumps(meta))
        np.save(rd / "initial_model.npy", np.full((24, 12, 12), 5100.))
        if iteration:
            it = rd / "iter_0"
            it.mkdir()
            np.save(it / "model.npy", np.full((24, 12, 12), 5200.))
            np.save(it / "delta_s.npy", np.ones((24, 12, 12)))
            np.save(it / "ray_count.npy", np.ones((24, 12, 12)))
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
    for iteration in (0, 1):
        for event in (0, 1):
            directory = rd / f"iter_{iteration}" / f"event_{event}"
            directory.mkdir(parents=True)
            np.savez(directory / "weights.npz",
                     positions=[[20, 0, 0], [event + iteration + 0.25, 0, 0]],
                     weight_values=[0.1, 0.9])
            np.save(directory / "residuals.npy", [[0., 2.], [-2., 0.]])
    # Compact logs without refined positions use fine-grid cell centres.
    directory = rd / "iter_0" / "event_2"
    directory.mkdir()
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
    event.mkdir()
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
    for dtype in ("delta_s", "ray_count"):
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
    assert client.get("/api/runs/latest").json == {"run_id": "good", "max_iter": 0}
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
    assert client.get("/api/runs/latest").json == {"run_id": None, "max_iter": 0}


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
