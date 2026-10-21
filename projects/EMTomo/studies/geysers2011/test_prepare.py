"""Small, arrival-free fixtures plus read-only validation of the real sources."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from projects.EMTomo.studies.geysers2011 import prepare
from projects.forward_modeling.experiments import load_inputs, save_inputs
from projects.forward_modeling.model import PointSet, VelocityGrid


GEOMETRY = {
    "top_face_center_latitude": 38.80275,
    "top_face_center_longitude": -122.7707,
    "azimuth_degrees": 135.0,
    "width_x_m": 20000.0, "width_y_m": 7000.0, "depth_m": 7000.0,
    "station_z_m": 0.0,
}


def _coordinates(latitude, longitude):
    north = (latitude - GEOMETRY["top_face_center_latitude"]) * 110574.0
    east = (longitude - GEOMETRY["top_face_center_longitude"]) * (
        111320.0 * math.cos(math.radians(GEOMETRY["top_face_center_latitude"]))
    )
    angle = math.radians(135)
    return (10000 + north * math.cos(angle) + east * math.sin(angle),
            3500 - north * math.sin(angle) + east * math.cos(angle))


def _event(identifier, time, depth=2.0):
    lat, lon = 38.81, -122.78
    x, y = _coordinates(lat, lon)
    return {"event_id": identifier, "origin_time": time, "latitude": lat,
            "longitude": lon, "depth_km": depth, "x_m": x, "y_m": y, "z_m": depth * 1000}


def _station(identifier, start="2011-03-22T20:32:00", end="3000-01-01T00:00:00"):
    lat, lon = 38.80, -122.77
    x, y = _coordinates(lat, lon)
    return {"network": "BG", "station": identifier, "latitude": lat, "longitude": lon,
            "elevation_m": 803.0, "site_name": "Actual fixture elevation",
            "start_time": start, "end_time": end, "x_m": x, "y_m": y, "z_m": 0.0}


def _csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def sources(tmp_path, monkeypatch):
    root, data = tmp_path / "experiments", tmp_path / "data"
    data.mkdir()
    stations = [
        _station("S02", "2011-03-06T01:30:00", prepare.END_UTC_EXCLUSIVE),
        _station("late", "2011-04-01T00:00:00.000001"),
        _station("S01"),
        _station("short", end="2011-04-20T23:59:59.999999"),
    ]
    events = [
        _event("end", prepare.END_UTC_EXCLUSIVE),
        _event("e02", "2011/04/20 23:59:59.999999"),
        _event("start", prepare.START_UTC),
        _event("before", "2011/03/31 23:59:59.999999"),
        _event("e01", "2011-04-21T00:59:59.999999+01:00"),
    ]
    _csv(data / "stations_metric.csv", stations)
    _csv(data / "events_metric_all.csv", events)
    (data / "geometry_metadata.json").write_text(json.dumps({
        "geometry": GEOMETRY, "source": {"catalog": "fixture"},
        "sampling": {"method": "not used"},
    }), encoding="utf-8")
    model = VelocityGrid(4000 + np.arange(20 * 7 * 7).reshape(20, 7, 7) / 8, 1000.0)
    point = PointSet(("base",), np.asarray([[10000, 3500, 1000]]))
    save_inputs(root, prepare.TRUE_MODEL_SOURCE, model, point, point, {"fixture": True})
    monkeypatch.setattr(prepare, "EXPECTED_STATION_COUNT", 2)
    monkeypatch.setattr(prepare, "EXPECTED_EVENT_COUNT", 3)
    return root, data, stations, events, model


def _prepare(sources, **kwargs):
    return prepare.prepare_experiment(sources[0], sources[1], **kwargs)


def test_interval_all_events_canonical_order_and_station_epochs(sources):
    stations, events, provenance = prepare._select_geometry(sources[1])
    assert stations.ids == ("S01", "S02")
    assert events.ids == ("start", "e01", "e02")
    assert provenance["event_origin_times_utc"]["e01"] == "2011-04-20T23:59:59.999999Z"
    assert [row["elevation_m"] for row in provenance["station_metadata"]] == [803.0, 803.0]
    assert np.all(stations.coordinates_m[:, 2] == 0.0)
    assert "sampling" not in provenance


@pytest.mark.parametrize("start,end,selected", [
    (prepare.START_UTC, prepare.END_UTC_EXCLUSIVE, True),
    ("2011-03-01T00:00:00Z", "", True),
    ("2011-04-01T00:00:00.000001Z", "", False),
    ("2011-03-01T00:00:00Z", "2011-04-20T23:59:59Z", False),
])
def test_epoch_coverage_filter(sources, start, end, selected):
    rows = sources[2] + [_station("epoch", start, end)]
    _csv(sources[1] / "stations_metric.csv", rows)
    stations, _, _ = prepare._select_geometry(sources[1])
    assert ("epoch" in stations.ids) is selected


def test_invalid_epoch_rejected(sources):
    rows = sources[2] + [_station("bad", "2011-04-05", "2011-04-04")]
    _csv(sources[1] / "stations_metric.csv", rows)
    with pytest.raises(ValueError, match="Invalid station epoch"):
        _prepare(sources)


@pytest.mark.parametrize("kind", ["event", "station"])
def test_duplicate_ids_rejected_even_outside_interval(sources, kind):
    if kind == "event":
        rows = sources[3] + [copy.deepcopy(sources[3][0])]
        name = "events_metric_all.csv"
    else:
        rows = sources[2] + [copy.deepcopy(sources[2][0])]
        name = "stations_metric.csv"
    _csv(sources[1] / name, rows)
    with pytest.raises(ValueError, match="duplicate"):
        _prepare(sources)
    assert not (sources[0] / "input" / prepare.EXPERIMENT_ID).exists()


@pytest.mark.parametrize("change,match", [
    ({"depth_km": 7.001, "z_m": 7001}, "outside"),
    ({"depth_km": -0.001, "z_m": -1}, "outside"),
    ({"x_m": 10001}, "projection"),
    ({"z_m": 1999}, "projection"),
    ({"longitude": float("nan")}, "Nonfinite"),
])
def test_event_bounds_projection_and_finite_validation(sources, change, match):
    rows = copy.deepcopy(sources[3])
    rows[2].update(change)
    _csv(sources[1] / "events_metric_all.csv", rows)
    with pytest.raises(ValueError, match=match):
        _prepare(sources)


def test_stations_not_silently_flattened_or_elevations_discarded(sources):
    rows = copy.deepcopy(sources[2])
    rows[0]["z_m"] = -803
    _csv(sources[1] / "stations_metric.csv", rows)
    with pytest.raises(ValueError, match="projection|flat"):
        _prepare(sources)


def test_counts_are_not_satisfied_by_thinning_or_generating(sources, monkeypatch):
    monkeypatch.setattr(prepare, "EXPECTED_EVENT_COUNT", 4)
    with pytest.raises(ValueError, match="Expected 2 stations/4 events; got 2 stations/3 events"):
        _prepare(sources)


def test_model_bitwise_identity_native_inputs_and_provenance(sources):
    result = _prepare(sources)
    assert result["experiment_id"] == prepare.EXPERIMENT_ID
    assert result["status"] == "created"
    assert result["event_count"] == 3
    assert result["station_count"] == 2
    assert result["pair_count"] == 6
    model, stations, events = load_inputs(sources[0], prepare.EXPERIMENT_ID)
    assert prepare._same_model(model, sources[4])
    assert model.velocity.tobytes() == sources[4].velocity.tobytes()
    assert model.velocity.shape == sources[4].velocity.shape
    assert stations.ids == ("S01", "S02")
    assert events.ids == ("start", "e01", "e02")
    generation = json.loads((Path(result["input_dir"]) / "generation.json").read_text())
    assert generation == result["generation"]
    archive = sources[0] / "input" / prepare.TRUE_MODEL_SOURCE / "model.npz"
    assert result["true_model_source_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert "no elevation" in generation["depth_datum_approximation"]
    assert "no local real picks" in generation["arrival_kind"]
    assert not (sources[0] / "output").exists()
    json.dumps(result, allow_nan=False)


def test_validate_only_without_arrivals_or_writes(sources):
    before = {str(path): path.read_bytes() for path in sources[0].rglob("*") if path.is_file()}
    result = _prepare(sources, validate_only=True)
    assert result["status"] == "validated"
    assert not Path(result["input_dir"]).exists()
    after = {str(path): path.read_bytes() for path in sources[0].rglob("*") if path.is_file()}
    assert before == after


def test_reuse_is_idempotent_without_writes(sources):
    first = _prepare(sources)
    directory = Path(first["input_dir"])
    before = {path.name: (path.stat().st_mtime_ns, path.read_bytes()) for path in directory.iterdir()}
    second = _prepare(sources)
    assert second["status"] == "reused"
    assert first["generation_sha256"] == second["generation_sha256"]
    assert _prepare(sources, validate_only=True)["status"] == "validated"
    after = {path.name: (path.stat().st_mtime_ns, path.read_bytes()) for path in directory.iterdir()}
    assert before == after


@pytest.mark.parametrize("kind", ["model", "stations", "events", "generation", "source", "archive"])
def test_different_existing_inputs_or_sources_never_overwritten(sources, kind):
    result = _prepare(sources)
    directory = Path(result["input_dir"])
    if kind == "model":
        model = sources[4]
        values = model.velocity.copy()
        values[0, 0, 0] = np.nextafter(values[0, 0, 0], np.inf)
        np.savez(directory / "model.npz", velocity=values, cell_size_m=model.cell_size_m,
                 origin_m=model.origin_m)
    elif kind in ("stations", "events"):
        file = directory / f"{kind}.csv"
        with file.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["x_m"] = str(float(rows[0]["x_m"]) + 1)
        _csv(file, rows)
    elif kind == "generation":
        file = directory / "generation.json"
        generation = json.loads(file.read_text())
        generation["source_hashes"]["events_metric_all.csv"] = "0" * 64
        file.write_text(json.dumps(generation))
    elif kind == "source":
        # Whitespace changes have identical parsed geometry but a different source hash.
        file = sources[1] / "geometry_metadata.json"
        file.write_text(file.read_text() + "\n")
    else:
        # NPZ accepts a trailing comment; semantic model unchanged, archive SHA changed.
        file = sources[0] / "input" / prepare.TRUE_MODEL_SOURCE / "model.npz"
        with file.open("ab") as stream:
            stream.write(b"archive provenance changed")
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    with pytest.raises(ValueError, match="refusing overwrite"):
        _prepare(sources)
    with pytest.raises(ValueError, match="refusing overwrite"):
        _prepare(sources, validate_only=True)
    assert before == {path.name: path.read_bytes() for path in directory.iterdir()}


def test_source_extent_not_regridded_to_fit(sources):
    model = sources[4]
    file = sources[0] / "input" / prepare.TRUE_MODEL_SOURCE / "model.npz"
    np.savez(file, velocity=model.velocity, cell_size_m=999.0, origin_m=model.origin_m)
    with pytest.raises(ValueError, match="Source model must cover"):
        _prepare(sources)


def test_invalid_id_cannot_escape_input_directory(sources):
    with pytest.raises(ValueError, match="Invalid experiment_id"):
        _prepare(sources, experiment_id="../escape")


def test_cli_json_and_prepare_default(sources, capsys):
    args = ["--experiments-root", str(sources[0]), "--data-dir", str(sources[1])]
    assert prepare.main(args + ["--prepare", "--validate-only"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "validated"
    assert prepare.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "created"
    assert prepare.main(args + ["--prepare"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "reused"


def test_direct_cli_help_works_outside_repository(tmp_path):
    result = subprocess.run([sys.executable, str(Path(prepare.__file__)), "--help"],
                            cwd=tmp_path, text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "--validate-only" in result.stdout


def test_real_sources_read_only_validation():
    archive = prepare.DEFAULT_EXPERIMENTS_ROOT / "input" / prepare.TRUE_MODEL_SOURCE / "model.npz"
    if not archive.is_file() or not prepare.DEFAULT_DATA_DIR.is_dir():
        pytest.skip("Real input sources not installed")
    result = prepare.prepare_experiment(validate_only=True)
    assert result["event_count"] == 3780
    assert result["station_count"] == 26
    assert result["pair_count"] == 98280
    generation = result["generation"]
    assert generation["previous_pair_count"] == 99840
    assert generation["pair_count_change_percent"] == -1.5625
    assert generation["true_grid"]["shape"] == [160, 56, 56]
    assert len(generation["event_origin_times_utc"]) == 3780
    assert len(generation["station_metadata"]) == 26
