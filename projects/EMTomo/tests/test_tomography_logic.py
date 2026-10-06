"""Regression tests for the tomography normal equations and ray geometry."""

import numpy as np
import pytest

from archive.legacy_synthetics.instruments_synthetic import generate_synthetic_arrivals_table
from archive.legacy_synthetics.locations import _resolve_event_locs_metric
from instruments.instruments_coords import (
    metric_to_cell_coord,
    metric_to_cell_index,
    sample_cell_centered_trilinear,
    snap_metric_points_to_cell_centers,
)
from instruments.instruments_ops import coarsen_G, coarsen_G_all
from instruments.instruments_travel import compute_station_travel_time_fields
from instruments.instruments_weights import candidate_posterior_weights, candidate_station_sigmas
from interpolation import prolongate_cell_centered_trilinear
from raytracing import (
    _trace_ray_nb,
    compute_G_all_stations_serial,
    rasterize_path_lengths,
)
from tomography.tomography_em import _limit_velocity_update
from tomography.tomography_math import (
    _normal_equation_contribution,
    _refine_epicenter_in_cell,
    _select_top_n_cells_by_misfit,
    _solve_delta_s,
)
from velocity_model import VelocityModel
from wave_propagation import SKFMMSolver


def test_candidate_station_sigmas_match_posterior_uncertainty_model():
    times = np.array([[1., 2.], [3., 4.]])
    sigmas = candidate_station_sigmas(times, 0.1, 0.2, 0.3)
    np.testing.assert_allclose(sigmas ** 2, (0.1 * times) ** 2 + 0.2 ** 2 + 0.3 ** 2)
    assert sigmas.shape == times.shape
    observed = np.array([0., 1.])
    weights = candidate_posterior_weights(
        times, observed, relative_sigma=0.1, absolute_sigma_s=0.2, model_sigma_s=0.3,
    )
    precision = 1 / sigmas ** 2
    residuals = observed - times
    means = np.sum(precision * residuals, axis=1) / precision.sum(axis=1)
    chi2 = np.sum(precision * (residuals - means[:, None]) ** 2, axis=1)
    likelihood = np.exp(-chi2 / 2) / (np.prod(sigmas, axis=1) * np.sqrt(precision.sum(axis=1)))
    np.testing.assert_allclose(weights, likelihood / likelihood.sum())
    with pytest.raises(ValueError, match="positive sigma"):
        candidate_station_sigmas(times, 0., 0., 0.)
    with pytest.raises(ValueError, match="relative_sigma"):
        candidate_station_sigmas(times, -0.1, 0.2, 0.3)


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

    assert np.isclose(h.item(), 0.25 * 2.0**2 / 2)
    assert np.isclose(b.item(), 0.25 * 2.0 * 3.0 / 2)


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

    assert np.allclose(h, weight / len(rows) * pair_rows.T @ pair_rows)
    assert np.allclose(b, weight / len(rows) * pair_rows.T @ pair_residuals)


def test_unequal_station_sigmas_profile_origin_and_exclude_failed_ray():
    station_g = np.array([1., 3., 100., 5.]).reshape(4, 1, 1, 1)
    station_r = np.array([2., -1., 100., 4.])
    sigmas = np.array([1., 2., np.nan, 4.])
    valid = np.array([True, True, False, True])
    weight = 0.6

    h, b = _normal_equation_contribution(
        station_g, station_r, (1, 1, 1), weight, valid,
        station_sigmas=sigmas,
    )
    rows = station_g[valid].reshape(-1, 1)
    residuals = station_r[valid]
    precision = 1 / sigmas[valid] ** 2
    pair_h = 0.0
    pair_b = 0.0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            pair_weight = weight * precision[i] * precision[j] / precision.sum()
            pair_h += pair_weight * (rows[i, 0] - rows[j, 0]) ** 2
            pair_b += pair_weight * (rows[i, 0] - rows[j, 0]) * (residuals[i] - residuals[j])
    assert h.item() == pytest.approx(pair_h)
    assert b.item() == pytest.approx(pair_b)


def test_pairs_with_failed_rays_are_excluded():
    g, r = _two_station_system(2.0, 3.0)
    h, b = _normal_equation_contribution(
        g, r, (1, 1, 1), 1.0, np.array([True, False])
    )

    assert not np.any(h)
    assert not np.any(b)


def test_coverage_damping_suppresses_poorly_resolved_cell_update():
    hessian = np.diag([1.0, 0.01])
    rhs = np.array([1.0, 0.01])

    uniform = _solve_delta_s(hessian, rhs, (2, 1, 1), lambda_reg=0.1)
    adaptive = _solve_delta_s(
        hessian,
        rhs,
        (2, 1, 1),
        lambda_reg=0.1,
        coverage_damping_power=1.0,
        coverage_floor=0.05,
    )

    assert adaptive[0].item() == uniform[0].item()
    assert abs(adaptive[1].item()) < abs(uniform[1].item())


@pytest.mark.parametrize("coverage_power", [0.0, 1.0])
def test_zero_smoothness_matches_previous_solver_exactly(coverage_power):
    hessian = np.diag([2.0, 0.5, 0.0])
    rhs = np.array([1.0, -0.5, 0.25])
    shape = (3, 1, 1)
    scale = np.trace(hessian) / 3
    sensitivity = np.maximum(np.diag(hessian), 0.0)
    positive = sensitivity[sensitivity > 0.0]
    reference = float(np.percentile(positive, 75.0))
    confidence = np.clip(sensitivity / max(reference, np.finfo(float).tiny), 0.0, 1.0)
    damping = 0.2 * scale / np.power(np.maximum(confidence, 0.05), coverage_power)
    expected = np.linalg.solve(hessian + np.diag(damping), rhs).reshape(shape)

    default = _solve_delta_s(
        hessian, rhs, shape, 0.2, coverage_damping_power=coverage_power,
        return_diagnostics=True,
    )
    explicit_zero = _solve_delta_s(
        hessian, rhs, shape, 0.2, coverage_damping_power=coverage_power,
        return_diagnostics=True, smoothness_reg=0.0,
    )
    assert all(np.array_equal(a, b) for a, b in zip(default, explicit_zero))
    assert np.array_equal(default[0], expected)
    assert np.array_equal(default[1], sensitivity.reshape(shape))
    assert np.array_equal(default[2], confidence.reshape(shape))


def test_smoothness_does_not_penalize_uniform_updates():
    shape = (2, 2, 2)
    hessian = np.eye(8)
    rhs = np.full(8, 2.0)

    delta_s = _solve_delta_s(hessian, rhs, shape, lambda_reg=0.0, smoothness_reg=5.0)

    assert np.allclose(delta_s, 2.0)
    assert np.allclose(hessian @ delta_s.ravel(), rhs)


def test_smoothness_interpolates_underdetermined_updates_without_absolute_prior():
    hessian = np.diag([1.0, 0.0, 0.0])
    rhs = np.array([2.0, 0.0, 0.0])
    shape = (3, 1, 1)

    without = _solve_delta_s(hessian, rhs, shape, lambda_reg=0.0)
    with_smoothness = _solve_delta_s(
        hessian, rhs, shape, lambda_reg=0.0, smoothness_reg=2.0,
    )

    assert np.array_equal(without.ravel(), [2.0, 0.0, 0.0])
    assert np.allclose(with_smoothness, 2.0)
    assert np.sum(np.diff(with_smoothness[:, 0, 0]) ** 2) < np.sum(
        np.diff(without[:, 0, 0]) ** 2
    )


@pytest.mark.parametrize("shape", [(2, 2, 2), (2, 3, 1), (1, 2, 3)])
def test_smoothness_uses_only_face_neighbours_and_six_degree_normalization(shape):
    n_vox = int(np.prod(shape))
    hessian = np.eye(n_vox)
    rhs = np.arange(1.0, n_vox + 1.0)
    laplacian = np.zeros((n_vox, n_vox))
    for cell in np.ndindex(shape):
        i = np.ravel_multi_index(cell, shape)
        for axis in range(3):
            neighbour = list(cell)
            neighbour[axis] += 1
            if neighbour[axis] >= shape[axis]:
                continue
            j = np.ravel_multi_index(tuple(neighbour), shape)
            laplacian[i, i] += 1.0
            laplacian[j, j] += 1.0
            laplacian[i, j] -= 1.0
            laplacian[j, i] -= 1.0

    result = _solve_delta_s(hessian, rhs, shape, lambda_reg=0.0, smoothness_reg=3.0)

    assert np.allclose(result.ravel(), np.linalg.solve(hessian + laplacian / 2.0, rhs))
    assert np.array_equal(laplacian @ np.ones(n_vox), np.zeros(n_vox))


def test_smoothness_scales_with_data_sensitivity_only():
    shape = (2, 1, 2)
    hessian = np.diag([1.0, 0.0, 2.0, 0.0])
    rhs = np.array([2.0, 0.0, -1.0, 0.0])
    baseline = _solve_delta_s(hessian, rhs, shape, lambda_reg=0.0, smoothness_reg=1.0)

    for factor in (1e-4, 1e4):
        scaled = _solve_delta_s(
            factor * hessian, factor * rhs, shape, lambda_reg=0.0, smoothness_reg=1.0,
        )
        assert np.allclose(scaled, baseline)
    assert np.array_equal(
        _solve_delta_s(np.zeros((4, 4)), np.zeros(4), shape, 0.0, smoothness_reg=1.0),
        np.zeros(shape),
    )


@pytest.mark.parametrize("invalid", [-1.0, np.nan, np.inf, -np.inf])
def test_smoothness_requires_nonnegative_finite_strength(invalid):
    with pytest.raises(ValueError, match="smoothness_reg must be finite and >= 0"):
        _solve_delta_s(np.eye(2), np.ones(2), (2, 1, 1), 0.0, smoothness_reg=invalid)


def test_velocity_update_trust_region_is_cellwise_and_relative():
    current = np.array([100.0, 200.0])
    proposed = np.array([80.0, 240.0])

    limited = _limit_velocity_update(current, proposed, max_step_fraction=0.01)

    assert np.allclose(limited, [99.0, 202.0])


def test_cell_centred_metric_coordinates_and_trilinear_sampling():
    field = np.fromfunction(
        lambda i, j, k: i + 10.0 * j + 100.0 * k,
        (5, 5, 5),
        dtype=np.float64,
    )

    assert metric_to_cell_coord((125.0, 175.0, 225.0), 50.0, field.shape) == (2.0, 3.0, 4.0)
    assert np.isclose(
        sample_cell_centered_trilinear(field, (1.25, 2.5, 3.75)),
        1.25 + 10.0 * 2.5 + 100.0 * 3.75,
    )


def test_trilinear_prolongation_and_G_restriction_are_adjoint():
    rng = np.random.default_rng(7)
    coarse_slowness = rng.uniform(0.005, 0.02, size=(3, 4, 2))
    fine_lengths = rng.normal(size=(6, 8, 4))

    fine_slowness = prolongate_cell_centered_trilinear(coarse_slowness, 2)
    coarse_lengths = coarsen_G(
        fine_lengths,
        subdivision=2,
        slowness_interpolation="trilinear",
    )

    assert np.allclose(
        np.sum(fine_slowness * fine_lengths),
        np.sum(coarse_slowness * coarse_lengths),
    )


def test_batched_trilinear_G_restriction_matches_stationwise_results():
    rng = np.random.default_rng(11)
    fine_lengths = rng.normal(size=(4, 6, 9, 12))

    batched = coarsen_G_all(
        fine_lengths,
        subdivision=3,
        slowness_interpolation="trilinear",
    )
    stationwise = np.stack([
        coarsen_G(
            station_lengths,
            subdivision=3,
            slowness_interpolation="trilinear",
        )
        for station_lengths in fine_lengths
    ])

    assert np.allclose(batched, stationwise)


def test_geo_grid_trilinear_slowness_interpolation():
    config = {
        "lon": 0.0,
        "lat": 0.0,
        "height": 0.0,
        "azimuth": 0.0,
        "side_size": 100.0,
        "n_x": 3,
        "n_y": 1,
        "n_z": 1,
    }
    model = VelocityModel.from_config(config)
    model.grid.vp[:, 0, 0] = [100.0, 200.0, 400.0]

    geo = model.get_geo_grid(subdivision=2, slowness_interpolation="trilinear")
    expected = 1.0 / prolongate_cell_centered_trilinear(1.0 / model.grid.vp, 2)

    assert np.allclose(geo.vp, expected)


def test_top_cells_are_separated_including_diagonals():
    misfit = np.full((5, 5, 5), 100.0)
    misfit[1, 1, 1] = 0.0
    misfit[2, 2, 2] = 0.1
    misfit[3, 1, 1] = 0.2

    selected = _select_top_n_cells_by_misfit(misfit, n=2, min_distance=2)

    assert np.array_equal(selected, [[1, 1, 1], [3, 1, 1]])


def test_single_hypothesis_refinement_recovers_subcell_position():
    shape = (5, 5, 5)
    x, y, z = np.indices(shape, dtype=np.float64)
    station_fields = np.stack([x, y, z, np.zeros(shape)], axis=0)
    target = np.array([2.2, 2.3, 2.4])
    arrivals = np.array([target[0], target[1], target[2], 0.0]) + 5.0

    refined, refined_misfit = _refine_epicenter_in_cell(
        station_fields,
        arrivals,
        cell_index=(2, 2, 2),
    )

    assert np.allclose(refined, target, atol=2e-3)
    assert refined_misfit < 1e-8


def test_surface_ray_finishes_after_entering_station_source_cell():
    shape = (4, 4, 2)
    gx = np.zeros(shape, dtype=np.float64)
    gy = np.zeros(shape, dtype=np.float64)
    gz = np.ones(shape, dtype=np.float64)
    station = np.array([2.0, 2.0, 0.0])
    epicenter = np.array([1.7, 2.0, 0.0])

    path, reached = _trace_ray_nb(
        gx,
        gy,
        gz,
        station,
        epicenter,
        0.1,
        0.1**2,
        100,
        np.zeros(3),
        np.asarray(shape, dtype=np.float64) - 1.0,
    )

    assert reached
    assert np.array_equal(path[-1], station)


def test_station_positions_snap_to_fine_cell_centers():
    centers = snap_metric_points_to_cell_centers(
        [(281.25, 281.25, 0.0), (843.75, 281.25, 0.0)],
        cell_size=50.0,
        shape=(90, 90, 90),
    )

    assert centers == [(275.0, 275.0, 25.0), (825.0, 275.0, 25.0)]
    assert [metric_to_cell_coord(point, 50.0, (90, 90, 90)) for point in centers] == [
        (5.0, 5.0, 0.0),
        (16.0, 5.0, 0.0),
    ]


def test_synthetic_arrivals_are_sampled_at_exact_event_position():
    config = {
        "lon": 0.0,
        "lat": 0.0,
        "height": 0.0,
        "azimuth": 0.0,
        "side_size": 100.0,
        "n_x": 4,
        "n_y": 4,
        "n_z": 4,
    }
    model = VelocityModel.from_config(config)
    model.fill_linear_gradient("vp", 100.0, 100.0)
    stations = [(50.0, 50.0, 0.0), (350.0, 50.0, 0.0)]
    event = (175.0, 150.0, 150.0)

    arrivals, _ = generate_synthetic_arrivals_table(
        model,
        station_locs=stations,
        event_locs=[event],
        solver="skfmm",
    )

    grid = model.get_geo_grid()
    fields = compute_station_travel_time_fields(
        grid,
        [metric_to_cell_index(station, grid.cell_size, grid.shape) for station in stations],
        "P",
        "skfmm",
    )
    event_coord = metric_to_cell_coord(event, grid.cell_size, grid.shape)
    expected_abs = np.array(
        [sample_cell_centered_trilinear(field, event_coord) for field in fields]
    )
    expected = expected_abs - np.min(expected_abs)

    assert np.allclose(arrivals[0], expected)


def test_synthetic_arrival_noise_is_reproducible_and_keeps_relative_times():
    config = {
        "lon": 0.0,
        "lat": 0.0,
        "height": 0.0,
        "azimuth": 0.0,
        "side_size": 100.0,
        "n_x": 4,
        "n_y": 4,
        "n_z": 4,
    }
    model = VelocityModel.from_config(config)
    model.fill_linear_gradient("vp", 100.0, 100.0)
    stations = [(50.0, 50.0, 0.0), (350.0, 50.0, 0.0), (50.0, 350.0, 0.0)]
    events = [(175.0, 150.0, 150.0), (250.0, 250.0, 150.0)]

    noiseless, _ = generate_synthetic_arrivals_table(
        model, station_locs=stations, event_locs=events, random_seed=7
    )
    noisy_a, _ = generate_synthetic_arrivals_table(
        model,
        station_locs=stations,
        event_locs=events,
        random_seed=7,
        arrival_noise_std=0.05,
    )
    noisy_b, _ = generate_synthetic_arrivals_table(
        model,
        station_locs=stations,
        event_locs=events,
        random_seed=7,
        arrival_noise_std=0.05,
    )

    assert np.allclose(noisy_a, noisy_b)
    assert all(np.isclose(np.min(event_arrivals), 0.0) for event_arrivals in noisy_a)
    assert not np.allclose(noisy_a, noiseless)


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
        test_coverage_damping_suppresses_poorly_resolved_cell_update,
        test_velocity_update_trust_region_is_cellwise_and_relative,
        test_cell_centred_metric_coordinates_and_trilinear_sampling,
        test_trilinear_prolongation_and_G_restriction_are_adjoint,
        test_geo_grid_trilinear_slowness_interpolation,
        test_top_cells_are_separated_including_diagonals,
        test_single_hypothesis_refinement_recovers_subcell_position,
        test_station_positions_snap_to_fine_cell_centers,
        test_synthetic_arrivals_are_sampled_at_exact_event_position,
        test_synthetic_arrival_noise_is_reproducible_and_keeps_relative_times,
        test_event_depth_offset_excludes_upper_layers,
        test_skfmm_source_is_at_cell_centre,
        test_successful_ray_reaches_station_and_uses_cell_centred_boundaries,
        test_failed_ray_produces_no_sensitivity,
    ]
    for test in tests:
        test()
        print(f"✓ {test.__name__}")
    print("\n✅ All tomography logic tests passed!")
