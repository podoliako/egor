"""Regression tests for the tomography normal equations, travel times and ray geometry."""

import numpy as np
import pytest

from eikonal import station_travel_time_fields, travel_time_field
from instruments.instruments_coords import metric_to_cell_coord, sample_cell_centered_trilinear_batch
from instruments.instruments_ops import restriction_tables
from instruments.likelihood import PickNoise
from interpolation import prolongate_cell_centered_trilinear
from raytracing import _rasterize_nb, _trace_ray_nb, compute_G_all_stations_serial
from tomography.tomography_em import update_velocity
from tomography.tomography_math import (
    accumulate_normal_equations,
    refine_hypocentre_in_cell,
    select_candidate_cells,
    solve_slowness_update,
)
from velocity_model import VelocityModel


def _contribution(g, residuals, weight=1.0, valid=None, sigmas=None):
    n_vox = int(np.prod(g.shape[1:]))
    hessian, rhs = np.zeros((n_vox, n_vox)), np.zeros(n_vox)
    valid = np.ones(len(residuals), dtype=bool) if valid is None else valid
    sigmas = np.ones(len(residuals)) if sigmas is None else sigmas
    accumulate_normal_equations(hessian, rhs, g, residuals, valid, sigmas, weight)
    return hessian, rhs


def _solve(hessian, rhs, shape, lambda_reg, **kwargs):
    return solve_slowness_update(hessian, rhs, shape, lambda_reg, **kwargs)[0]


def _two_station_system(g_value, residual):
    station_g = np.zeros((2, 1, 1, 1))
    station_g[0, 0, 0, 0] = g_value
    return station_g, np.array([residual, 0.0])


def test_events_accumulate_as_independent_normal_equations():
    h1, b1 = _contribution(*_two_station_system(1.0, 1.0))
    h2, b2 = _contribution(*_two_station_system(2.0, 0.0))
    assert np.isclose(_solve(h1 + h2, b1 + b2, (1, 1, 1), 0.0).item(), 0.2)


def test_em_weight_enters_normal_equations_linearly():
    h, b = _contribution(*_two_station_system(2.0, 3.0), weight=0.25)
    assert np.isclose(h.item(), 0.25 * 2.0 ** 2 / 2)
    assert np.isclose(b.item(), 0.25 * 2.0 * 3.0 / 2)


def test_centered_station_formula_matches_explicit_pairs():
    rng = np.random.default_rng(7)
    station_g = rng.normal(size=(5, 2, 2, 1))
    station_r = rng.normal(size=5)
    valid = np.array([True, False, True, True, False])
    h, b = _contribution(station_g, station_r, 0.37, valid)

    rows, residuals = station_g[valid].reshape(3, -1), station_r[valid]
    pairs = [(i, j) for i in range(3) for j in range(i + 1, 3)]
    pair_rows = np.asarray([rows[i] - rows[j] for i, j in pairs])
    pair_residuals = np.asarray([residuals[i] - residuals[j] for i, j in pairs])
    assert np.allclose(h, 0.37 / 3 * pair_rows.T @ pair_rows)
    assert np.allclose(b, 0.37 / 3 * pair_rows.T @ pair_residuals)


def test_unequal_station_sigmas_profile_origin_and_exclude_failed_ray():
    station_g = np.zeros((4, 2, 2, 1))
    flat = station_g.reshape(4, -1)
    flat[:, 1] = [1., 2., 100., 3.]
    flat[:, 3] = [0., 4., 100., 2.]
    residuals = np.array([1., -2., 100., 3.])
    valid = np.array([True, True, False, True])
    sigmas = np.array([1., 2., np.nan, 4.])
    h, b = _contribution(station_g, residuals, 0.4, valid, sigmas)

    rows = flat[valid]
    precision = 1 / sigmas[valid] ** 2
    centered_rows = rows - np.average(rows, axis=0, weights=precision)
    centered_residuals = residuals[valid] - np.average(residuals[valid], weights=precision)
    np.testing.assert_allclose(h, 0.4 * centered_rows.T @ (precision[:, None] * centered_rows))
    np.testing.assert_allclose(b, 0.4 * centered_rows.T @ (precision * centered_residuals))
    assert not np.any(h[0]) and not np.any(h[:, 2])


def test_pairs_with_failed_rays_are_excluded():
    h, b = _contribution(*_two_station_system(2.0, 3.0), valid=np.array([True, False]))
    assert not np.any(h) and not np.any(b)


def test_coverage_damping_suppresses_poorly_resolved_cell_update():
    hessian, rhs = np.diag([1.0, 0.01]), np.array([1.0, 0.01])
    uniform = _solve(hessian, rhs, (2, 1, 1), 0.1)
    adaptive = _solve(hessian, rhs, (2, 1, 1), 0.1, coverage_damping_power=1.0, coverage_floor=0.05)
    assert adaptive[0].item() == uniform[0].item()
    assert abs(adaptive[1].item()) < abs(uniform[1].item())


@pytest.mark.parametrize("coverage_power", [0.0, 1.0])
def test_damped_solution_and_diagnostics(coverage_power):
    hessian, rhs, shape = np.diag([2.0, 0.5, 0.0]), np.array([1.0, -0.5, 0.25]), (3, 1, 1)
    sensitivity = np.diag(hessian)
    confidence = np.clip(sensitivity / np.percentile(sensitivity[sensitivity > 0], 75.0), 0, 1)
    damping = 0.2 * np.trace(hessian) / 3 / np.power(np.maximum(confidence, 0.05), coverage_power)
    expected = np.linalg.solve(hessian + np.diag(damping), rhs).reshape(shape)

    result = solve_slowness_update(hessian, rhs, shape, 0.2, coverage_damping_power=coverage_power)
    np.testing.assert_allclose(result[0], expected)
    np.testing.assert_array_equal(result[1], sensitivity.reshape(shape))
    np.testing.assert_allclose(result[2], confidence.reshape(shape))


def test_velocity_update_applies_slowness_increment_and_cellwise_trust_region():
    velocity = np.array([100.0, 200.0])
    delta_s = 1 / np.array([80.0, 240.0]) - 1 / velocity
    np.testing.assert_allclose(update_velocity(velocity, delta_s, 0.01), [99.0, 202.0])
    np.testing.assert_allclose(update_velocity(velocity, delta_s, None), [80.0, 240.0])
    with pytest.raises(ValueError, match="non-positive"):
        update_velocity(velocity, np.array([-1.0, 0.0]), None)


def test_cell_centred_coordinates_and_trilinear_sampling():
    field = np.fromfunction(lambda i, j, k: i + 10.0 * j + 100.0 * k, (5, 5, 5))
    np.testing.assert_allclose(metric_to_cell_coord((125.0, 175.0, 225.0), 50.0), (2.0, 3.0, 4.0))
    assert np.isclose(sample_cell_centered_trilinear_batch(field, (1.25, 2.5, 3.75)),
                      1.25 + 10.0 * 2.5 + 100.0 * 3.75)


def _rasterize(path, shape, subdivision=1, interpolation="identity"):
    out_shape = tuple(n // subdivision for n in shape) if interpolation != "identity" else shape
    G = np.zeros(out_shape)
    _rasterize_nb(np.asarray(path, float), G, 1.0, *restriction_tables(out_shape, subdivision, interpolation))
    return G


def _random_path(rng, shape, n=40):
    return np.cumsum(rng.normal(scale=0.7, size=(n, 3)), axis=0) % (np.asarray(shape) - 1)


def test_nearest_restriction_sums_subcell_lengths():
    rng = np.random.default_rng(3)
    shape = (6, 9, 12)
    path = _random_path(rng, shape)
    fine = _rasterize(path, shape)
    coarse = _rasterize(path, shape, 3, "nearest")
    np.testing.assert_allclose(coarse, fine.reshape(2, 3, 3, 3, 4, 3).sum(axis=(1, 3, 5)))
    assert np.isclose(coarse.sum(), np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def test_trilinear_restriction_is_adjoint_of_slowness_prolongation():
    rng = np.random.default_rng(7)
    shape = (6, 8, 4)
    coarse_slowness = rng.uniform(0.005, 0.02, size=(3, 4, 2))
    fine_slowness = prolongate_cell_centered_trilinear(coarse_slowness, 2)
    path = _random_path(rng, shape)
    fine = _rasterize(path, shape)
    coarse = _rasterize(path, shape, 2, "trilinear")
    assert np.isclose(np.sum(fine_slowness * fine), np.sum(coarse_slowness * coarse))


def test_top_cells_are_separated_including_diagonals():
    cost = np.full((5, 5, 5), 100.0)
    cost[1, 1, 1], cost[2, 2, 2], cost[3, 1, 1] = 0.0, 0.1, 0.2
    np.testing.assert_array_equal(select_candidate_cells(cost, 2, 2), [[1, 1, 1], [3, 1, 1]])


def test_candidate_shortlist_matches_full_sort():
    rng = np.random.default_rng(2)
    cost = rng.normal(size=(20, 15, 12))
    for n, distance in ((1, 1), (5, 1), (4, 3), (8, 6)):
        selected = []
        for flat in np.argsort(cost, axis=None, kind="stable"):
            index = np.array(np.unravel_index(flat, cost.shape))
            if all(np.max(np.abs(index - other)) >= distance for other in selected):
                selected.append(index)
            if len(selected) == n:
                break
        np.testing.assert_array_equal(select_candidate_cells(cost, n, distance), selected)


def test_single_hypothesis_refinement_recovers_subcell_position():
    shape = (5, 5, 5)
    x, y, z = np.indices(shape, dtype=np.float64)
    fields = np.stack([x, y, z, np.zeros(shape)], axis=0)
    target = np.array([2.2, 2.3, 2.4])
    arrivals = np.append(target, 0.0) + 5.0
    refined, value = refine_hypocentre_in_cell(fields, arrivals, (2, 2, 2), PickNoise(0., 0.1, 0.))
    assert np.allclose(refined, target, atol=2e-3)
    assert value < 1e-6


def test_exact_station_travel_times_are_accurate_and_parallel_matches_serial():
    h, shape, velocity = 500.0, (40, 30, 24), 5000.0
    model = VelocityModel(np.full(shape, velocity), h)
    stations = np.array([[10_130.0, 7_270.0, 0.0], [0.0, 0.0, 0.0], [19_900.0, 15_000.0, 3_100.0]])
    serial = station_travel_time_fields(model, stations, n_workers=1)
    parallel = station_travel_time_fields(model, stations, n_workers=3)
    np.testing.assert_array_equal(serial, parallel)
    axes = [(np.arange(n) + 0.5) * h for n in shape]
    centres = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    for field, station in zip(serial, stations):
        exact = np.linalg.norm(centres - station, axis=-1) / velocity
        error = field - exact
        assert np.abs(error).max() < 0.03          # 0.06 h / v
        assert np.std(error) < 0.01


def test_travel_time_field_rejects_sources_outside_domain():
    with pytest.raises(ValueError, match="outside"):
        travel_time_field(np.full((4, 4, 4), 1000.0), 100.0, (-1.0, 0.0, 0.0))


def test_surface_ray_finishes_within_half_a_cell_of_the_station():
    shape = (4, 4, 2)
    gx, gy, gz = np.zeros(shape), np.zeros(shape), np.ones(shape)
    station = np.array([2.0, 2.0, -0.5])
    path, reached = _trace_ray_nb(gx, gy, gz, station, np.array([1.7, 2.0, 1.0]),
                                  0.1, 0.01, 100, np.zeros(3), np.asarray(shape, float) - 1.0)
    assert reached
    np.testing.assert_array_equal(path[-1], station)


def test_successful_ray_reaches_station_and_uses_cell_centred_boundaries():
    shape = (5, 5, 5)
    gx, zeros = np.ones(shape), np.zeros(shape)
    station = np.array([0.0, 0.0, 0.0])
    path, reached = _trace_ray_nb(gx, zeros, zeros, station, np.array([4.0, 0.0, 0.0]),
                                  0.1, 0.01, 100, np.zeros(3), np.asarray(shape, float) - 1.0)
    sensitivity = _rasterize(path, shape)
    assert reached
    assert np.allclose(path[-1], station)
    assert np.isclose(sensitivity.sum(), 4.0)
    assert np.allclose(sensitivity[:, 0, 0], [0.5, 1.0, 1.0, 1.0, 0.5])


def test_failed_ray_produces_no_sensitivity():
    shape = (5, 5, 5)
    gradients = np.zeros((1,) + shape, dtype=np.float32)
    sensitivity, reached = compute_G_all_stations_serial(
        gradients, gradients, gradients, np.zeros((1, 3)), np.array([4.0, 0.0, 0.0]),
        1.0, 0.1, 0.1, 20, np.zeros(3), np.asarray(shape, float) - 1.0,
        *restriction_tables(shape, 1, "identity"), shape,
    )
    assert not reached[0]
    assert not np.any(sensitivity)
