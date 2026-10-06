"""Wavy layers, compact bubbles, and immutable forward-modeling inputs."""
from dataclasses import asdict
import hashlib
import json

import numpy as np
import pytest

from projects.forward_modeling import PointSet, VelocityGrid, load_inputs, save_inputs
from projects.forward_modeling.layered_bubbles_generator import (
    DEFAULT_BUBBLES, LayeredBubblesConfig, generate_layered_bubbles_model,
    layer_boundaries_km, main, save_layered_bubbles_experiment,
)
from projects.forward_modeling.spherical_generator import DEFAULT_ANOMALIES, SphereAnomaly


@pytest.fixture(scope="module")
def sampled_model():
    config = LayeredBubblesConfig()
    return config, generate_layered_bubbles_model(config)


def test_default_parameters_and_all_sampled_interfaces(sampled_model):
    config, model = sampled_model
    assert config.lengths_km == (240., 120., 120.)
    assert model.velocity.shape == (96, 48, 48)
    assert model.cell_size_m == 2500.
    assert config.layer_velocities_m_s == (4600., 4850., 5150., 5400.)
    assert [b.center_km for b in DEFAULT_BUBBLES] == [
        (34., 29., 36.), (76., 82., 67.), (134., 37., 92.),
        (185., 80., 41.), (212., 32., 84.), (120., 95., 30.),
    ]
    for bubble, original in zip(DEFAULT_BUBBLES, DEFAULT_ANOMALIES):
        assert bubble.radius_km == original.radius_km * 1.5
        assert bubble.peak_delta_m_s == original.peak_delta_m_s
        assert all(bubble.radius_km < p < length - bubble.radius_km
                   for p, length in zip(bubble.center_km, config.lengths_km))
    assert [b.radius_km for b in DEFAULT_BUBBLES] == [21., 31.5, 27., 39., 18., 24.]

    x, y, z = [(np.arange(n) + 0.5) * 2.5 for n in model.velocity.shape]
    b1, b2, b3 = layer_boundaries_km(x[:, None], y[None, :])
    assert np.all((0 < b1) & (b1 < b2) & (b2 < b3) & (b3 < 120))
    # Remove the analytically expected bubble contribution, then check every
    # voxel's background, including voxels on both sides of each interface.
    expected = np.full(model.velocity.shape, 4600.)
    for boundary, speed in zip((b1, b2, b3), (4850., 5150., 5400.)):
        expected = np.where(z[None, None, :] >= boundary[:, :, None], speed, expected)
    for bubble in DEFAULT_BUBBLES:
        cx, cy, cz = bubble.center_km
        distance = np.sqrt((x[:, None, None] - cx) ** 2 +
                           (y[None, :, None] - cy) ** 2 +
                           (z[None, None, :] - cz) ** 2)
        expected += np.where(distance < bubble.radius_km,
                             bubble.peak_delta_m_s * (1 + np.cos(
                                 np.pi * distance / bubble.radius_km)) / 2, 0.)
    np.testing.assert_allclose(model.velocity, expected, rtol=0, atol=1e-12)
    assert model.velocity.min() > 0 and np.isfinite(model.velocity).all()
    # Layer depths vary horizontally; they are not four flat slices.
    assert np.ptp(b1) > 15 and np.ptp(b2) > 20 and np.ptp(b3) > 15


def test_layer_boundary_formulas():
    b1, b2, b3 = layer_boundaries_km(0., 0.)
    assert b1 == pytest.approx(34.)
    assert b2 == pytest.approx(60 + 8 * np.sin(0.8) + 5 * np.sin(0.3))
    assert b3 == pytest.approx(90 + 6 * np.cos(0.4))
    b1, b2, b3 = layer_boundaries_km(120., 60.)
    assert b1 == pytest.approx(26.)
    assert b2 == pytest.approx(60 - 8 * np.sin(0.8) - 5 * np.sin(0.3))
    assert b3 == pytest.approx(90 - 6 * np.cos(0.4))


def test_cosine_bubble_is_compact_and_c1_at_its_boundary():
    # The bubble and sampled horizontal line lie wholly in the first layer.
    config = LayeredBubblesConfig(anomalies=(SphereAnomaly((50., 50., 15.), 10., 360.),))
    model = generate_layered_bubbles_model(config)
    x = (np.arange(model.velocity.shape[0]) + 0.5) * 2.5
    y = (np.arange(model.velocity.shape[1]) + 0.5) * 2.5
    z = (np.arange(model.velocity.shape[2]) + 0.5) * 2.5
    iz = int(np.argmin(abs(z - 15.)))
    iy = int(np.argmin(abs(y - 50.)))
    for ix in range(13, 27):
        r = np.linalg.norm((x[ix] - 50., y[iy] - 50., z[iz] - 15.))
        expected = 4600. + (360. * (1 + np.cos(np.pi * r / 10.)) / 2 if r < 10. else 0.)
        assert model.velocity[ix, iy, iz] == pytest.approx(expected)
    # The interior slope tends to zero at r=R, matching the zero exterior slope.
    epsilon = 1e-7
    interior_slope = -360. * np.pi / 20. * np.sin(np.pi * (10. - epsilon) / 10.)
    assert abs(interior_slope) < 1e-5


@pytest.mark.parametrize("options, message", [
    ({"lengths_km": (240., 120.)}, "lengths_km"),
    ({"cell_size_km": 7.}, "cell_size_km"),
    ({"layer_velocities_m_s": (4600., 4850.)}, "layer_velocities_m_s"),
    ({"layer_velocities_m_s": (4600., 4850., -1., 5400.)}, "layer_velocities_m_s"),
    ({"anomalies": ()}, "anomalies"),
    ({"anomalies": (SphereAnomaly((10., 20., 30.), 15., 100.),)}, "within the domain"),
    ({"lengths_km": (240., 120., 80.), "anomalies":
      (SphereAnomaly((50., 50., 30.), 10., 100.),)}, "layer boundaries"),
])
def test_invalid_config(options, message):
    with pytest.raises(ValueError, match=message):
        LayeredBubblesConfig(**options)


@pytest.fixture
def geometry(tmp_path):
    stations = PointSet(("S2", "S1"), [[1740., 960., 0.], [360., 340., 0.]])
    events = PointSet(("E2", "E1"), [[1340., 580., 670.], [350., 320., 450.]])
    save_inputs(tmp_path, "reference", VelocityGrid(np.full((2, 1, 1), 5000.), 1000.),
                stations, events)
    return tmp_path, stations, events



def test_cli_and_geometry_provenance(tmp_path, capsys):
    save_inputs(tmp_path, "reference", VelocityGrid(np.full((4, 2, 2), 5000.), 60_000.),
                PointSet(("S2", "S1"), [[174000., 96000., 0.], [36000., 34000., 0.]]),
                PointSet(("E2", "E1"), [[134000., 58000., 67000.], [35000., 32000., 45000.]]))
    assert main(["layered", "--root", str(tmp_path),
                 "--geometry-experiment", "reference"]) == 0
    directory = tmp_path / "input" / "layered"
    assert capsys.readouterr().out.strip() == str(directory)
    reference = tmp_path / "input" / "reference"
    hashes = {}
    for name in ("stations.csv", "events.csv"):
        content = (reference / name).read_bytes()
        assert (directory / name).read_bytes() == content
        hashes[name] = hashlib.sha256(content).hexdigest()
    generation = json.loads((directory / "generation.json").read_text())
    assert generation["generator"] == "layered_bubbles"
    assert generation["parameters"] == json.loads(json.dumps(asdict(LayeredBubblesConfig())))
    assert generation["geometry_source"] == {"id": "reference", "sha256": hashes}
    assert len(generation["layer_boundaries_km"]) == 3
    model, stations, events = load_inputs(tmp_path, "layered")
    assert model.velocity.shape == (96, 48, 48)
    assert stations.ids == ("S2", "S1") and events.ids == ("E2", "E1")
    assert not (tmp_path / "output").exists()
    with pytest.raises(FileExistsError, match="already exists"):
        save_layered_bubbles_experiment(tmp_path, "layered", "reference")


def test_incompatible_geometry_is_rejected_before_writing(geometry):
    root, _, _ = geometry
    with pytest.raises(ValueError, match="same domain dimensions"):
        save_layered_bubbles_experiment(root, "layered", "reference")
    assert not (root / "input" / "layered").exists()
