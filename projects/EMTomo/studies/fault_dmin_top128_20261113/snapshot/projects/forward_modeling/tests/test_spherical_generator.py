"""Depth-gradient spherical anomalies and preservation of saved geometry."""
from dataclasses import asdict
import hashlib
import json

import numpy as np
import pytest

from projects.forward_modeling import PointSet, VelocityGrid, load_inputs, save_inputs
from projects.forward_modeling.spherical_generator import (
    DEFAULT_ANOMALIES, SphereAnomaly, SphericalConfig,
    generate_spherical_model, main, save_spherical_experiment,
)


def test_default_spheres_have_smooth_compact_support_over_depth_gradient():
    config = SphericalConfig()
    model = generate_spherical_model(config)
    assert model.velocity.shape == (96, 48, 48)
    assert model.cell_size_m == 2500.
    assert len(config.anomalies) == 6
    assert len({sphere.radius_km for sphere in config.anomalies}) == 6
    assert any(s.peak_delta_m_s > 0 for s in config.anomalies)
    assert any(s.peak_delta_m_s < 0 for s in config.anomalies)

    x, y, z = [(np.arange(n) + 0.5) * config.cell_size_km for n in model.velocity.shape]
    background = config.surface_velocity_m_s + (
        config.bottom_velocity_m_s - config.surface_velocity_m_s
    ) * z / config.lengths_km[2]
    np.testing.assert_allclose(model.velocity[0, 0], background)
    for sphere in DEFAULT_ANOMALIES:
        centre = np.asarray(sphere.center_km)
        sample = tuple(np.argmin(abs(axis - value)) for axis, value in zip((x, y, z), centre))
        distance = np.linalg.norm(np.array([axis[index] for axis, index in zip((x, y, z), sample)]) - centre)
        expected = background[sample[2]] + sphere.peak_delta_m_s * (
            1 + np.cos(np.pi * distance / sphere.radius_km)
        ) / 2
        assert model.velocity[sample] == pytest.approx(expected)
    assert np.isfinite(model.velocity).all() and model.velocity.min() > 0


@pytest.mark.parametrize("sphere, message", [
    (((1., 2.), 1., 50.), "center_km"),
    (((1., 2., float("nan")), 1., 50.), "center_km"),
    (((1., 2., 3.), 0., 50.), "radius_km"),
    (((1., 2., 3.), 1., 0.), "peak_delta_m_s"),
])
def test_invalid_sphere(sphere, message):
    with pytest.raises(ValueError, match=message):
        SphereAnomaly(*sphere)


@pytest.mark.parametrize("options, message", [
    ({"lengths_km": (240, 120)}, "lengths_km"),
    ({"cell_size_km": 7.}, "cell_size_km"),
    ({"surface_velocity_m_s": -1.}, "background velocities"),
    ({"anomalies": ()}, "anomalies"),
    ({"anomalies": (SphereAnomaly((10., 20., 30.), 15., 100.),)}, "within the domain"),
])
def test_invalid_config(options, message):
    with pytest.raises(ValueError, match=message):
        SphericalConfig(**options)


@pytest.fixture
def geometry(tmp_path):
    stations = PointSet(("S2", "S1"), [[1740., 960., 0.], [360., 340., 0.]])
    events = PointSet(("E2", "E1"), [[1340., 580., 670.], [350., 320., 450.]])
    save_inputs(tmp_path, "reference", VelocityGrid(np.full((2, 1, 1), 5000.), 1000.),
                stations, events)
    return tmp_path, stations, events


def test_saved_spherical_input_reuses_exact_geometry_and_records_provenance(geometry):
    root, stations, events = geometry
    config = SphericalConfig(lengths_km=(2., 1., 1.), cell_size_km=0.25,
                             anomalies=(SphereAnomaly((1., 0.5, 0.5), 0.25, 100.),))
    directory = save_spherical_experiment(root, "spheres", "reference", config)
    reference = root / "input" / "reference"
    for filename in ("stations.csv", "events.csv"):
        assert (directory / filename).read_bytes() == (reference / filename).read_bytes()
    generation = json.loads((directory / "generation.json").read_text())
    assert generation == {
        "generator": "spherical_gradient", "parameters": json.loads(json.dumps(asdict(config))),
        "geometry_source": {
            "id": "reference",
            "sha256": {name: hashlib.sha256((reference / name).read_bytes()).hexdigest()
                       for name in ("stations.csv", "events.csv")},
        },
    }
    model, loaded_stations, loaded_events = load_inputs(root, "spheres")
    assert model.velocity.shape == (8, 4, 4)
    assert loaded_stations.ids == stations.ids
    assert loaded_events.ids == events.ids
    assert not (root / "output").exists()


def test_cli_uses_existing_geometry_without_solving(tmp_path, capsys):
    save_inputs(tmp_path, "reference", VelocityGrid(np.full((4, 2, 2), 5000.), 60_000.),
                PointSet(("S1",), [[6000., 7000., 0.]]),
                PointSet(("E1",), [[12000., 20000., 35000.]]))
    assert main(["spheres", "--root", str(tmp_path),
                 "--geometry-experiment", "reference"]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "input" / "spheres")
    model, stations, events = load_inputs(tmp_path, "spheres")
    assert model.velocity.shape == (96, 48, 48)
    assert stations.ids == ("S1",) and events.ids == ("E1",)
    assert not (tmp_path / "output").exists()
