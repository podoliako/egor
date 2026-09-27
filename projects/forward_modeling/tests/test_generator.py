"""Checkerboard geometry, sampling, and generated-input provenance."""

from dataclasses import asdict, replace
import hashlib
import json

import numpy as np
import pytest

from projects.forward_modeling import experiments
from projects.forward_modeling.generator import (
    CheckerboardConfig, generate_checkerboard, main, save_checkerboard,
)
from projects.forward_modeling.model import Arrival, ForwardConfig


def test_default_checkerboard_shape_and_parity():
    config = CheckerboardConfig()
    assert asdict(config) == {
        "lengths_km": (240.0, 120.0, 120.0),
        "blocks": (4, 2, 2),
        "velocities_m_s": (4750.0, 5250.0),
        "surface_stations": (14, 7),
        "n_events": 1000,
        "min_depth_km": 10.0,
        "bottom_bias": 0.15,
        "seed": 42,
    }
    model, _, _ = generate_checkerboard(config)
    assert model.velocity.shape == (4, 2, 2)
    assert model.cell_size_m == 60_000
    assert model.origin_m == (0, 0, 0)
    expected = np.where(np.indices(config.blocks).sum(axis=0) % 2 == 0,
                        4750.0, 5250.0)
    np.testing.assert_array_equal(model.velocity, expected)
    np.testing.assert_array_equal(np.asarray(model.velocity.shape) * model.cell_size_m,
                                  [240_000, 120_000, 120_000])


def test_default_surface_stations_are_evenly_spaced_and_unique():
    _, stations, _ = generate_checkerboard()
    coords = stations.coordinates_m
    assert coords.shape == (14 * 7, 3)
    assert len(set(stations.ids)) == len(stations.ids) == 98
    assert len(np.unique(coords, axis=0)) == 98
    np.testing.assert_array_equal(coords[:, 2], np.zeros(98))
    x = np.unique(coords[:, 0])
    y = np.unique(coords[:, 1])
    np.testing.assert_allclose(x, (np.arange(14) + 0.5) * 240_000 / 14)
    np.testing.assert_allclose(y, (np.arange(7) + 0.5) * 120_000 / 7)
    assert set(map(tuple, coords[:, :2])) == {(a, b) for a in x for b in y}


def test_default_events_cover_domain_and_slightly_favor_bottom():
    model, _, events = generate_checkerboard()
    coords = events.coordinates_m
    assert coords.shape == (1000, 3)
    assert len(set(events.ids)) == 1000
    model.validate_points(events, "events")
    assert np.all((coords[:, 0] >= 0) & (coords[:, 0] <= 240_000))
    assert np.all((coords[:, 1] >= 0) & (coords[:, 1] <= 120_000))
    assert np.all((coords[:, 2] >= 10_000) & (coords[:, 2] <= 120_000))
    # Coverage is about the whole footprint, not just membership in its bounds.
    for axis, length in ((0, 240_000), (1, 120_000)):
        assert coords[:, axis].min() < 0.05 * length
        assert coords[:, axis].max() > 0.95 * length
    midpoint = (10_000 + 120_000) / 2
    lower = np.count_nonzero(coords[:, 2] > midpoint)
    assert 510 < lower < 600  # mild bias, rather than uniform or bottom clustering


def test_seed_reproduces_events_without_changing_model_or_stations():
    config = CheckerboardConfig(n_events=32, seed=7)
    first = generate_checkerboard(config)
    again = generate_checkerboard(config)
    other = generate_checkerboard(replace(config, seed=8))
    for a, b in zip(first, again):
        if hasattr(a, "velocity"):
            np.testing.assert_array_equal(a.velocity, b.velocity)
        else:
            assert a.ids == b.ids
            np.testing.assert_array_equal(a.coordinates_m, b.coordinates_m)
    np.testing.assert_array_equal(first[0].velocity, other[0].velocity)
    np.testing.assert_array_equal(first[1].coordinates_m, other[1].coordinates_m)
    assert not np.array_equal(first[2].coordinates_m, other[2].coordinates_m)


@pytest.mark.parametrize("changes, message", [
    ({"lengths_km": (240, 120)}, "lengths_km"),
    ({"lengths_km": (240, 120, 0)}, "lengths_km"),
    ({"lengths_km": (240, 120, float("nan"))}, "lengths_km"),
    ({"lengths_km": (240, 120, 100)}, "cubic"),
    ({"blocks": (4, 2)}, "blocks"),
    ({"blocks": (4, 2, 0)}, "blocks"),
    ({"blocks": (4, True, 2)}, "blocks"),
    ({"blocks": (3, 2, 2)}, "cubic"),
    ({"velocities_m_s": (4750,)}, "velocities_m_s"),
    ({"velocities_m_s": (4750, float("inf"))}, "velocities_m_s"),
    ({"velocities_m_s": (4750, -1)}, "velocities_m_s"),
    ({"surface_stations": (14,)}, "surface_stations"),
    ({"surface_stations": (0, 7)}, "surface_stations"),
    ({"surface_stations": (True, 7)}, "surface_stations"),
    ({"n_events": 0}, "n_events"),
    ({"n_events": 1.5}, "n_events"),
    ({"min_depth_km": 0}, "min_depth_km"),
    ({"min_depth_km": 120}, "min_depth_km"),
    ({"min_depth_km": float("nan")}, "min_depth_km"),
    ({"bottom_bias": -0.01}, "bottom_bias"),
    ({"bottom_bias": 1}, "bottom_bias"),
    ({"bottom_bias": float("inf")}, "bottom_bias"),
    ({"seed": -1}, "seed"),
    ({"seed": True}, "seed"),
])
def test_invalid_config(changes, message):
    with pytest.raises(ValueError, match=message):
        CheckerboardConfig(**changes)


def test_generated_inputs_and_run_metadata_include_provenance_hashes(tmp_path, monkeypatch):
    config = CheckerboardConfig(lengths_km=(2, 1, 1), blocks=(2, 1, 1),
                                surface_stations=(2, 1), n_events=2,
                                min_depth_km=0.1, seed=9)
    source = save_checkerboard(tmp_path, "small", config)
    assert source == tmp_path / "input" / "small"
    names = {"model.npz", "stations.csv", "events.csv", "generation.json"}
    assert {path.name for path in source.iterdir()} == names
    generation = json.loads((source / "generation.json").read_text())
    assert generation == {"generator": "checkerboard",
                          "parameters": json.loads(json.dumps(asdict(config)))}
    expected = generate_checkerboard(config)
    loaded = experiments.load_inputs(tmp_path, "small")
    np.testing.assert_array_equal(loaded[0].velocity, expected[0].velocity)
    for actual, original in zip(loaded[1:], expected[1:]):
        assert actual.ids == original.ids
        np.testing.assert_allclose(actual.coordinates_m, original.coordinates_m)

    def compute_arrivals(*, model, stations, events, config):
        assert model.velocity.shape == (2, 1, 1)
        assert len(stations.ids) == len(events.ids) == 2
        assert config == ForwardConfig(refinement=1)
        return [Arrival(stations.ids[0], event_id, 0.0) for event_id in events.ids]

    monkeypatch.setattr(experiments.solver, "compute_arrivals", compute_arrivals)
    destination = experiments.run_experiment(tmp_path, "small", ForwardConfig(refinement=1))
    assert destination == tmp_path / "output" / "small"
    metadata = json.loads((destination / "metadata.json").read_text())
    assert metadata["input_sha256"] == {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in names
    }
    assert len(experiments.load_arrivals(tmp_path, "small")) == 2


def test_cli_generates_inputs_only_and_roundtrips_options(tmp_path, capsys):
    args = ["cli-case", "--root", str(tmp_path), "--lengths-km", "4", "2", "2",
            "--blocks", "2", "1", "1", "--velocities-m-s", "3000", "4000",
            "--surface-stations", "3", "2", "--events", "8",
            "--min-depth-km", "0.25", "--bottom-bias", "0.2", "--seed", "11"]
    assert main(args) == 0
    source = tmp_path / "input" / "cli-case"
    assert capsys.readouterr().out.strip() == str(source)
    assert {path.name for path in source.iterdir()} == {
        "model.npz", "stations.csv", "events.csv", "generation.json"
    }
    assert not (tmp_path / "output").exists()
    config = CheckerboardConfig(lengths_km=(4, 2, 2), blocks=(2, 1, 1),
                                velocities_m_s=(3000, 4000), surface_stations=(3, 2),
                                n_events=8, min_depth_km=0.25, bottom_bias=0.2, seed=11)
    assert json.loads((source / "generation.json").read_text())["parameters"] == json.loads(
        json.dumps(asdict(config)))
    model, stations, events = experiments.load_inputs(tmp_path, "cli-case")
    assert model.velocity.shape == (2, 1, 1)
    assert model.cell_size_m == 2000
    assert len(stations.ids) == 6
    assert len(events.ids) == 8
    assert np.all(events.coordinates_m[:, 2] >= 250)
    for actual, expected in zip((model, stations, events), generate_checkerboard(config)):
        if hasattr(actual, "velocity"):
            np.testing.assert_array_equal(actual.velocity, expected.velocity)
        else:
            assert actual.ids == expected.ids
            np.testing.assert_allclose(actual.coordinates_m, expected.coordinates_m)
