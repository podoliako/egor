"""Horizontal layers over a dipping interface, flat-topped spheres, uniform geometry."""
from dataclasses import replace
import json

import numpy as np
import pytest

from projects.forward_modeling import load_inputs
from projects.forward_modeling.dipping_layers_generator import (
    DippingLayersConfig, dipping_boundary_depth_km, generate_dipping_layers,
    generate_dipping_layers_model, main,
)
from projects.forward_modeling.spherical_generator import SphereAnomaly


def _no_anomalies(config):
    # A negligible anomaly keeps the configuration valid while isolating the layers.
    return replace(config, anomalies=(SphereAnomaly((120., 60., 60.), 1., 1e-9),))


def test_layers_are_horizontal_above_and_dipping_below():
    config = _no_anomalies(DippingLayersConfig())
    velocity = generate_dipping_layers_model(config).velocity
    z = (np.arange(48) + 0.5) * 2.5
    profile = velocity[0, 0]
    np.testing.assert_allclose(profile[z < 20], 4600.)
    np.testing.assert_allclose(profile[(z > 20) & (z < 45)], 4900.)
    np.testing.assert_allclose(profile[(z > 45) & (z < 65)], 5200.)
    np.testing.assert_allclose(profile[z > 66], 5600.)
    for i in (0, 47, 95):
        x = (i + 0.5) * 2.5
        column = velocity[i, 7]
        depth = dipping_boundary_depth_km(config, x)
        np.testing.assert_allclose(column[z >= depth], 5600.)
        np.testing.assert_allclose(column[(z > 45) & (z < depth)], 5200.)
        np.testing.assert_array_equal(column, velocity[i, 40])  # no variation along y


def test_flat_topped_anomalies_reach_full_amplitude_and_vanish_outside():
    anomaly = SphereAnomaly((50., 50., 50.), 20., -700., 0.5)
    np.testing.assert_allclose(anomaly.profile(np.array([0., 9.9, 20., 30.])), [-700., -700., 0., 0.])
    assert -700. < anomaly.profile(np.array([15.]))[0] < 0.
    legacy = SphereAnomaly((50., 50., 50.), 20., 300.)
    distance = np.linspace(0., 25., 11)
    np.testing.assert_allclose(legacy.profile(distance),
                               np.where(distance < 20., 300. * (1 + np.cos(np.pi * distance / 20.)) / 2, 0.))


def test_geometry_is_uniform_and_reproducible():
    config = DippingLayersConfig(n_events=256)
    _, stations, events = generate_dipping_layers(config)
    assert len(stations.ids) == 72 and np.all(stations.coordinates_m[:, 2] == 0.)
    np.testing.assert_allclose(np.unique(stations.coordinates_m[:, 0]), (np.arange(12) + 0.5) * 20_000.)
    z = events.coordinates_m[:, 2] / 1000
    assert z.min() >= 5. and z.max() <= 115.
    counts = np.histogram(z, bins=4, range=(5., 115.))[0]
    assert counts.min() >= 0.9 * 256 / 4
    _, _, again = generate_dipping_layers(config)
    np.testing.assert_array_equal(events.coordinates_m, again.coordinates_m)


@pytest.mark.parametrize("change", [
    dict(horizontal_boundaries_km=(45., 20.)),
    dict(dipping_boundary_km=(40., 95.)),
    dict(layer_velocities_m_s=(4600., 4900., 5200.)),
    dict(anomalies=(SphereAnomaly((10., 60., 60.), 15., 500.),)),
    dict(event_depth_km=(0., 115.)),
    dict(surface_stations=(0, 6)),
])
def test_invalid_configurations_rejected(change):
    with pytest.raises(ValueError):
        replace(DippingLayersConfig(), **change)


def test_cli_saves_inputs_with_generation_record(tmp_path):
    assert main(["dip", "--root", str(tmp_path), "--surface-stations", "4", "2",
                 "--events", "16", "--seed", "3"]) == 0
    model, stations, events = load_inputs(tmp_path, "dip")
    assert model.velocity.shape == (96, 48, 48)
    assert len(stations.ids) == 8 and len(events.ids) == 16
    generation = json.loads((tmp_path / "input" / "dip" / "generation.json").read_text())
    assert generation["generator"] == "dipping_layers"
    assert generation["parameters"]["seed"] == 3
    assert len(generation["parameters"]["anomalies"]) == 6
