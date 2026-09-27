"""Contract tests for adapting actual saved forward-modeling experiments."""

import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

# Allow the same test to run from the repository root or from EMTomo.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from projects.EMTomo.experiment_data import DEFAULT_EXPERIMENTS_ROOT, load_tomography_experiment
from projects.forward_modeling.experiments import load_arrivals, run_experiment, save_inputs
from projects.forward_modeling.model import ForwardConfig, PointSet, VelocityGrid


@pytest.fixture
def saved(tmp_path):
    model = VelocityGrid(np.full((4, 2, 2), 3000.0), 1000.0)
    stations = PointSet(("S3", "S1", "S2"), [[100., 100., 0.],
                                                [2500., 300., 0.], [3700., 1700., 0.]])
    events = PointSet(("E2", "E1", "E3"), [[400., 200., 800.],
                                            [1800., 1500., 900.], [3300., 600., 1200.]])
    save_inputs(tmp_path, "tiny", model, stations, events)
    run_experiment(tmp_path, "tiny", config=ForwardConfig(refinement=1))
    return tmp_path, model, stations, events


def _rows(root):
    path = root / "output" / "tiny" / "arrivals.csv"
    with path.open(newline="") as stream:
        return list(csv.reader(stream))


def _write_rows(root, rows):
    path = root / "output" / "tiny" / "arrivals.csv"
    with path.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)


def _metadata_path(root):
    return root / "output" / "tiny" / "metadata.json"


def test_real_run_reordered_csv(saved):
    root, model, stations, events = saved
    original = {(a.event_id, a.station_id): a.arrival_time_s for a in load_arrivals(root, "tiny")}
    rows = _rows(root)
    _write_rows(root, [rows[0], *reversed(rows[1:])])
    result = load_tomography_experiment("tiny", root)
    assert result.source_model.velocity.shape == (4, 2, 2)
    np.testing.assert_array_equal(result.source_model.velocity, model.velocity)
    assert result.source_model.cell_size_m == 1000.0
    assert result.station_ids == stations.ids
    assert result.event_ids == events.ids
    np.testing.assert_array_equal(result.station_coordinates_m, stations.coordinates_m)
    np.testing.assert_array_equal(result.reference_event_coordinates_m, events.coordinates_m)
    assert result.arrival_times_s.shape == (3, 3)
    assert result.arrival_times_s.dtype == np.float64
    np.testing.assert_array_equal(result.arrival_times_s, [
        [original[(event, station)] for station in stations.ids] for event in events.ids
    ])
    np.testing.assert_array_equal(result.arrival_times_s.min(axis=1), 0.)


def test_default_root():
    assert DEFAULT_EXPERIMENTS_ROOT == Path(__file__).resolve().parents[2] / "forward_modeling" / "experiments"


@pytest.mark.parametrize("change, error", [
    (lambda rows: rows[:-1], "Incomplete arrivals"),
    (lambda rows: [*rows, ["UNKNOWN", "E1", "0"]], "Unknown arrival ID"),
    (lambda rows: [*rows, ["S1", "UNKNOWN", "0"]], "Unknown arrival ID"),
    (lambda rows: [*rows, rows[1]], "Duplicate arrival pair"),
    (lambda rows: [*rows[:1], *[[s, e, str(float(t) + 0.01)]
                                  for s, e, t in rows[1:]]], "zero earliest-station"),
    (lambda rows: [*rows[:1], *[[s, e, "nan" if i == 0 else t]
                                  for i, (s, e, t) in enumerate(rows[1:])]], "finite nonnegative"),
])
def test_reject_bad_arrivals(saved, change, error):
    root = saved[0]
    _write_rows(root, change(_rows(root)))
    with pytest.raises(ValueError, match=error):
        load_tomography_experiment("tiny", root)


@pytest.mark.parametrize("field,value,error", [
    ("time_reference", "event_origin_time", "time_reference"),
    ("units", {"coordinates": "km"}, "units"),
    ("input_sha256", {}, "input_sha256"),
])
def test_reject_bad_metadata(saved, field, value, error):
    root = saved[0]
    path = _metadata_path(root)
    metadata = json.loads(path.read_text())
    metadata[field] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=error):
        load_tomography_experiment("tiny", root)


@pytest.mark.parametrize("content,error", [("{", ValueError), ("[]", ValueError), (None, FileNotFoundError)])
def test_reject_malformed_or_missing_metadata(saved, content, error):
    root = saved[0]
    path = _metadata_path(root)
    if content is None:
        path.unlink()
    else:
        path.write_text(content)
    with pytest.raises(error, match="metadata"):
        load_tomography_experiment("tiny", root)


def test_reject_changed_input(saved):
    root = saved[0]
    path = root / "input" / "tiny" / "stations.csv"
    path.write_text(path.read_text().replace("100.0", "101.0", 1))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_tomography_experiment("tiny", root)


def test_reject_nonzero_origin(tmp_path):
    model = VelocityGrid(np.full((4, 2, 2), 3000.0), 1000.0, (10., 0., 0.))
    stations = PointSet(("A", "B"), [[100., 100., 0.], [2500., 300., 0.]])
    events = PointSet(("E1", "E2", "E3"), [[400., 200., 800.],
                                            [1800., 1500., 900.], [3300., 600., 1200.]])
    save_inputs(tmp_path, "tiny", model, stations, events)
    run_experiment(tmp_path, "tiny", config=ForwardConfig(refinement=1))
    with pytest.raises(ValueError, match="zero-origin"):
        load_tomography_experiment("tiny", tmp_path)


def test_reject_one_station(tmp_path):
    model = VelocityGrid(np.full((4, 2, 2), 3000.0), 1000.0)
    stations = PointSet(("A",), [[100., 100., 0.]])
    events = PointSet(("E1", "E2", "E3"), [[400., 200., 800.],
                                            [1800., 1500., 900.], [3300., 600., 1200.]])
    save_inputs(tmp_path, "tiny", model, stations, events)
    run_experiment(tmp_path, "tiny", config=ForwardConfig(refinement=1))
    with pytest.raises(ValueError, match="at least two stations"):
        load_tomography_experiment("tiny", tmp_path)
