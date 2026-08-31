"""Regression tests for the tomography normal equations and ray geometry."""

import numpy as np

from instruments.instruments_coords import _resolve_event_locs_metric
from raytracing import (
    _trace_ray_nb,
    compute_G_all_stations_serial,
    rasterize_path_lengths,
)
from tomography.tomography_math import (
    _normal_equation_contribution,
    _solve_delta_s,
)
from wave_propagation import SKFMMSolver


def _two_station_system(g_value: float, residual: float):
    station_g = np.zeros((2, 1, 1, 1), dtype=np.float64)
    station_g[0, 0, 0, 0] = g_value
    station_r = np.array([residual, 0.0], dtype=np.float64)
    return station_g, station_r


def test_events_accumulate_as_independent_normal_equations():
    g1, r1 = _two_station_system(1.0, 1.0)
    g2, r2 = _two_station_system(2.0, 0.0)
    valid = np.ones(2, dtype=bool)

    h1, b1 = _normal_equation_contribution(g1, r1, (1, 1, 1), 1.0, valid)
    h2, b2 = _normal_equation_contribution(g2, r2, (1, 1, 1), 1.0, valid)
    delta_s = _solve_delta_s(h1 + h2, b1 + b2, (1, 1, 1), 0.0)

    assert np.isclose(delta_s.item(), 0.2)


def test_em_weight_enters_normal_equations_linearly():
    g1, r1 = _two_station_system(2.0, 3.0)
    valid = np.ones(2, dtype=bool)

    h, b = _normal_equation_contribution(g1, r1, (1, 1, 1), 0.25, valid)

    assert np.isclose(h.item(), 0.25 * 2.0**2)
    assert np.isclose(b.item(), 0.25 * 2.0 * 3.0)


def test_centered_station_formula_matches_explicit_pairs():
    rng = np.random.default_rng(7)
    station_g = rng.normal(size=(5, 2, 2, 1))
    station_r = rng.normal(size=5)
    valid = np.array([True, False, True, True, False])
    weight = 0.37

    h, b = _normal_equation_contribution(
        station_g, station_r, (2, 2, 1), weight, valid
    )

    rows = station_g[valid].reshape(3, -1)
    residuals = station_r[valid]
    pair_rows = []
    pair_residuals = []
    for i in range(3):
        for j in range(i + 1, 3):
            pair_rows.append(rows[i] - rows[j])
            pair_residuals.append(residuals[i] - residuals[j])
    pair_rows = np.asarray(pair_rows)
    pair_residuals = np.asarray(pair_residuals)

    assert np.allclose(h, weight * pair_rows.T @ pair_rows)
    assert np.allclose(b, weight * pair_rows.T @ pair_residuals)


def test_pairs_with_failed_rays_are_excluded():
    g, r = _two_station_system(2.0, 3.0)
    h, b = _normal_equation_contribution(
        g, r, (1, 1, 1), 1.0, np.array([True, False])
    )

    assert not np.any(h)
    assert not np.any(b)


def test_event_depth_offset_excludes_upper_layers():
    cell_size = 50.0
    shape = (90, 90, 90)
    z_offset = 2 * 500.0
    events, event_indices = _resolve_event_locs_metric(
        shape=shape,
        cell_size=cell_size,
        event_locs=None,
        n_events=100,
        rng=np.random.default_rng(7),
        depth_bias=0.0,
        z_offset=z_offset,
    )

    assert all(event[2] >= z_offset for event in events)
    assert all(index[2] >= 20 for index in event_indices)


def test_skfmm_source_is_at_cell_centre():
    velocity = np.full((7, 7, 7), 100.0, dtype=np.float64)
    travel_time = SKFMMSolver(order=2).solve(velocity, (3, 3, 3), 10.0)

    assert np.isclose(travel_time[3, 3, 3], 0.0)
    assert np.isclose(travel_time[4, 3, 3], 0.1)
    assert np.isclose(travel_time[5, 3, 3], 0.2)


def test_successful_ray_reaches_station_and_uses_cell_centred_boundaries():
    shape = (5, 5, 5)
    gx = np.ones(shape, dtype=np.float64)
    zeros = np.zeros(shape, dtype=np.float64)
    station = np.array([0.0, 0.0, 0.0])
    epicentre = np.array([4.0, 0.0, 0.0])

    path, reached = _trace_ray_nb(
        gx,
        zeros,
        zeros,
        station,
        epicentre,
        0.1,
        0.1**2,
        100,
        np.zeros(3),
        np.array(shape, dtype=np.float64) - 1.0,
    )
    sensitivity = rasterize_path_lengths(
        path, shape, voxel_size=(1.0, 1.0, 1.0), dtype=np.float64
    )

    assert reached
    assert np.allclose(path[-1], station)
    assert np.isclose(sensitivity.sum(), 4.0)
    assert np.allclose(sensitivity[:, 0, 0], [0.5, 1.0, 1.0, 1.0, 0.5])


def test_failed_ray_produces_no_sensitivity():
    shape = (5, 5, 5)
    gradients = np.zeros((1,) + shape, dtype=np.float64)
    stations = np.array([[0.0, 0.0, 0.0]])

    sensitivity, reached = compute_G_all_stations_serial(
        gradients,
        gradients,
        gradients,
        stations,
        np.array([4.0, 0.0, 0.0]),
        1.0,
        1.0,
        1.0,
        0.1,
        0.1,
        20,
        np.zeros(3),
        np.array(shape, dtype=np.float64) - 1.0,
    )

    assert not reached[0]
    assert not np.any(sensitivity)


if __name__ == "__main__":
    tests = [
        test_events_accumulate_as_independent_normal_equations,
        test_em_weight_enters_normal_equations_linearly,
        test_centered_station_formula_matches_explicit_pairs,
        test_pairs_with_failed_rays_are_excluded,
        test_event_depth_offset_excludes_upper_layers,
        test_skfmm_source_is_at_cell_centre,
        test_successful_ray_reaches_station_and_uses_cell_centred_boundaries,
        test_failed_ray_produces_no_sensitivity,
    ]
    for test in tests:
        test()
        print(f"✓ {test.__name__}")
    print("\n✅ All tomography logic tests passed!")
