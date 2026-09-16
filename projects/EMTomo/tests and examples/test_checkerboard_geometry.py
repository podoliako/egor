"""Tests for deterministic checkerboard-series geometry builders."""
from dataclasses import replace

import numpy as np

from main import (
    CONFIG,
    build_top_surface_stations,
    build_true_model,
    build_uniform_volume_events,
)
from velocity_model import VelocityModel


def test_anisotropic_checkerboard_blocks_follow_coarse_cell_shape():
    config = replace(
        CONFIG,
        cell_size=10_000.0,
        grid_shape=(24, 12, 12),
        checkerboard_block_shape=(4, 2, 2),
        checkerboard_rotation_degrees=0.0,
    )
    model = VelocityModel.from_config({
        "lon": config.lon,
        "lat": config.lat,
        "height": config.height,
        "azimuth": config.azimuth,
        "side_size": config.cell_size,
        "n_x": config.grid_shape[0],
        "n_y": config.grid_shape[1],
        "n_z": config.grid_shape[2],
    })

    build_true_model(model, config)

    low = config.background_vp * (1.0 - config.checkerboard_anomaly_fraction)
    high = config.background_vp * (1.0 + config.checkerboard_anomaly_fraction)
    assert set(np.unique(model.grid.vp)) == {low, high}
    assert np.count_nonzero(model.grid.vp == low) == model.grid.vp.size // 2
    assert np.all(model.grid.vp[0:4, 0:2, 0:2] == model.grid.vp[0, 0, 0])
    assert model.grid.vp[4, 0, 0] != model.grid.vp[0, 0, 0]
    assert model.grid.vp[0, 2, 0] != model.grid.vp[0, 0, 0]
    assert model.grid.vp[0, 0, 2] != model.grid.vp[0, 0, 0]


def test_depth_biased_event_grid_is_regular_and_denser_below():
    events = build_uniform_volume_events(
        (24, 12, 6), (24, 12, 12), 10_000.0, depth_bias=0.5
    )
    depths = np.unique(np.asarray(events)[:, 2])

    assert len(events) == 1_728
    assert len(depths) == 6
    assert np.all(np.diff(depths) > 0.0)
    assert np.all(np.diff(np.diff(depths)) < 0.0)
    assert 0.0 < depths[0] < depths[-1] < 120_000.0


def test_zero_depth_bias_preserves_uniform_spacing():
    events = build_uniform_volume_events((2, 2, 4), (2, 2, 4), 100.0)
    depths = np.unique(np.asarray(events)[:, 2])

    assert np.allclose(depths, [50.0, 150.0, 250.0, 350.0])


def test_negative_depth_bias_is_rejected():
    try:
        build_uniform_volume_events((2, 2, 2), (2, 2, 2), 100.0, depth_bias=-0.1)
    except ValueError as error:
        assert "depth_bias" in str(error)
    else:
        raise AssertionError("Negative depth_bias must be rejected")


def test_station_grid_has_equal_horizontal_spacing_for_two_to_one_model():
    stations = np.asarray(build_top_surface_stations(14, 7, 24, 12, 10_000.0))
    dx = np.diff(np.unique(stations[:, 0]))
    dy = np.diff(np.unique(stations[:, 1]))

    assert len(stations) == 98
    assert np.allclose(dx, dy[0])
    assert np.all(stations[:, 2] == 0.0)
