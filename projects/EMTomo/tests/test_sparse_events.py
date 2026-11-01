"""Missing-pick numerical tests using analytic fields only: no forward/inverse solves."""
from dataclasses import replace
import multiprocessing as mp

import numpy as np
import pytest

from instruments.hypocentre_objective import make_refinement_objective
from instruments.instruments_coords import sample_cell_centered_trilinear_batch
from instruments.likelihood import PickNoise, negative_log_likelihood
from tomography import tomography_events as events
from tomography.tomography_logging import TomographyLogger, summarize_ray_status
from tomography.tomography_math import MatrixFreeNormalEquations


def _problem(relative=0.03):
    stations = np.array([[0., 0., 0.], [3., 0., 0.], [0., 3., 0.],
                         [3., 3., 0.], [1., 0., 3.]])
    grid = np.moveaxis(np.indices((4, 4, 4), dtype=float), 0, -1)
    times = (np.linalg.norm(grid[None] - stations[:, None, None, None], axis=-1) / 5).astype(np.float32)
    noise = PickNoise(relative, 0.05, 0.1)
    fields = events.IterationFields.from_times(times, stations, 1., 2, noise)
    settings = events.EventSettings(
        subdivision=2, slowness_interpolation="nearest", n_candidates=3, weights_top_n=2,
        weights_min_distance=1, candidate_mode="soft", temperature=1., noise=noise,
        log_misfit=True, log_g_per_weight=True,
    )
    observed = sample_cell_centered_trilinear_batch(times, [1.2, 1.4, 1.7]) + 0.3
    observed += np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    return observed, fields, settings


def _subset(fields, mask, settings):
    return events.IterationFields.from_times(
        fields.times[mask], fields.stations[mask], fields.fine_cell_size,
        settings.subdivision, settings.noise,
    )


def _process(observed, fields, settings, compute_G=None):
    hessian = (MatrixFreeNormalEquations(fields.n_vox) if settings.normal_solver == "matrix_free"
               else np.zeros((fields.n_vox, fields.n_vox)))
    rhs = np.zeros(fields.n_vox)
    log = events.process_event(observed, fields, settings, hessian, rhs, compute_G)
    return hessian, rhs, log


def _dense(hessian, n):
    if isinstance(hessian, MatrixFreeNormalEquations):
        return np.column_stack([hessian.matvec(row) for row in np.eye(n)])
    return hessian


@pytest.mark.parametrize("relative", [0., 0.03])
@pytest.mark.parametrize("indices", [[0, 2, 4], [1, 2], [0, 1, 2, 3, 4]])
def test_indexed_scores_and_objective_match_explicit_subset(relative, indices):
    observed, fields, settings = _problem(relative)
    indices = np.asarray(indices)
    expected = negative_log_likelihood(fields.times[indices], observed[indices], settings.noise)
    actual = negative_log_likelihood(fields.times, observed[indices], settings.noise,
                                     station_indices=indices)
    np.testing.assert_array_equal(actual, expected)
    objective = make_refinement_objective(fields.times, observed[indices], settings.noise,
                                          station_indices=indices)
    reference = make_refinement_objective(fields.times[indices], observed[indices], settings.noise)
    for point in ([1.2, 1.4, 1.7], [0., 3., 1.], [-1., 4., 2.]):
        assert objective(point) == reference(point)
        np.testing.assert_array_equal(
            sample_cell_centered_trilinear_batch(fields.times, point, station_indices=indices),
            sample_cell_centered_trilinear_batch(fields.times[indices], point),
        )


@pytest.mark.parametrize("relative", [0., 0.03])
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("mode", ["hard", "soft"])
@pytest.mark.parametrize("solver", ["dense", "matrix_free"])
def test_sparse_event_matches_dense_subset_with_real_rays(relative, compact, fast, mode, solver):
    observed, fields, settings = _problem(relative)
    settings = replace(settings, compact_rays=compact, fast_refinement=fast, candidate_mode=mode,
                       normal_solver=solver)
    mask = np.array([True, False, True, True, False])
    missing = np.where(mask, observed, np.nan)
    actual = _process(missing, fields, settings)
    expected = _process(observed[mask], _subset(fields, mask, settings), settings)
    np.testing.assert_allclose(_dense(actual[0], fields.n_vox), _dense(expected[0], fields.n_vox),
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-12, atol=1e-12)
    assert np.any(_dense(actual[0], fields.n_vox))
    a, b = actual[2], expected[2]
    for name in ("candidate_cells", "positions", "weights", "misfit"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    np.testing.assert_array_equal(a.observed_mask, mask)
    np.testing.assert_array_equal(a.residuals[np.ix_(mask, mask)], b.residuals)
    assert np.isnan(a.residuals[~mask]).all() and np.isnan(a.residuals[:, ~mask]).all()
    for index in a.ray_reached_per_weight:
        np.testing.assert_array_equal(a.ray_reached_per_weight[index][mask], b.ray_reached_per_weight[index])
        assert not a.ray_reached_per_weight[index][~mask].any()
        np.testing.assert_array_equal(a.ray_count_per_weight[index], b.ray_count_per_weight[index])
        packed, subset = a.G_per_weight[index], b.G_per_weight[index]
        np.testing.assert_array_equal(packed["coords"], subset["coords"])
        np.testing.assert_array_equal(packed["values"], subset["values"])
        np.testing.assert_array_equal(np.diff(packed["offsets"])[mask], np.diff(subset["offsets"]))
        assert not np.diff(packed["offsets"])[~mask].any()
    summary = summarize_ray_status([(0, a)], len(mask))
    reference = summarize_ray_status([(0, b)], mask.sum())
    for name in ("expected", "reached", "failed", "used"):
        assert summary[name] == reference[name]


def _fake_rays(gx, gy, gz, stations, position, *args):
    """Deterministic station-dependent rows; no numerical ray/forward solve."""
    shape = args[-1]
    rows = np.zeros((len(stations), *shape))
    for i, station in enumerate(stations):
        rows[i].flat[int(station.sum()) % int(np.prod(shape))] = 1 + station[0] + position[0]
    return rows, np.ones(len(stations), dtype=bool)


@pytest.mark.parametrize("solver", ["dense", "matrix_free"])
@pytest.mark.parametrize("workers", [1, 2])
def test_mixed_sparse_batch_and_multiprocessing_match_individual_subsets(monkeypatch, solver, workers):
    if workers > 1 and "fork" not in mp.get_all_start_methods():
        pytest.skip("Event workers require fork")
    observed, fields, settings = _problem()
    settings = replace(settings, normal_solver=solver, log_g_per_weight=False)
    arrivals = np.stack([observed + shift for shift in (0., 0.1, -0.2, 0.05, 0.3)])
    arrivals[0, [1, 4]] = np.nan
    arrivals[2, [0, 2]] = np.nan
    arrivals[3, [0, 2, 4]] = np.nan
    monkeypatch.setattr(events, "compute_G_all_stations_compact", _fake_rays)
    batch_call = events.negative_log_likelihood_batch
    batch_sizes = []

    def checked_batch(times, observations, *args):
        assert np.isfinite(observations).all()
        batch_sizes.append(len(observations))
        return batch_call(times, observations, *args)

    monkeypatch.setattr(events, "negative_log_likelihood_batch", checked_batch)
    expected_h, expected_rhs = np.zeros((fields.n_vox, fields.n_vox)), np.zeros(fields.n_vox)
    expected_logs = []
    for row in arrivals:
        mask = np.isfinite(row)
        h, rhs, log = _process(row[mask], _subset(fields, mask, settings), settings, _fake_rays)
        expected_h += _dense(h, fields.n_vox)
        expected_rhs += rhs
        expected_logs.append(log)
    for batch_size in (1, 8):
        h, rhs, logs = events.run_events(arrivals, fields, replace(settings, likelihood_batch_size=batch_size), workers)
        np.testing.assert_allclose(_dense(h, fields.n_vox), expected_h, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(rhs, expected_rhs, rtol=1e-12, atol=1e-12)
        assert [index for index, _ in logs] == list(range(len(arrivals)))
        for (_, actual), expected in zip(logs, expected_logs):
            np.testing.assert_allclose(actual.positions, expected.positions)
            np.testing.assert_allclose(actual.weights, expected.weights)
    if workers == 1:
        assert batch_sizes == [2]  # Fully observed rows still use the tiled batch kernel.


def test_sparse_tracing_uses_views_and_logs_global_ids(tmp_path):
    observed, fields, settings = _problem()
    mask = np.array([False, True, True, False, True])
    calls = []

    def checked_rays(gx, gy, gz, stations, position, *args):
        for view, original in ((gx, fields.gx), (gy, fields.gy), (gz, fields.gz)):
            assert np.shares_memory(view, original)
        ids = [int(np.flatnonzero(np.all(fields.stations == station, axis=1))[0]) for station in stations]
        assert set(ids) <= {1, 2, 4}
        calls.extend(ids)
        rows, reached = _fake_rays(gx, gy, gz, stations, position, *args)
        reached[:] = np.array(ids) != 4
        rows[~reached] = 0
        return rows, reached

    _, _, log = _process(np.where(mask, observed, np.nan), fields, settings, checked_rays)
    active = np.count_nonzero(log.weights)
    assert len(calls) == active * mask.sum() * 2  # primary and optional fine-grid traces
    summary = summarize_ray_status([(7, log)], len(mask))
    assert summary["expected"] == active * 3
    assert summary["failed"] == active
    assert summary["used"] == summary["reached"] == active * 2
    assert all(failure["stations"] == [4] for failure in summary["failures"])
    logger = TomographyLogger(base_dir=tmp_path, save_misfit=True)
    logger.save_event_data(0, 7, log)
    directory = logger.run_dir / "iter_0/event_7"
    for name in ("weights.npz", "ray_status.npz"):
        with np.load(directory / name) as saved:
            np.testing.assert_array_equal(saved["observed_mask"], mask)
    with np.load(directory / "ray_status.npz") as saved:
        np.testing.assert_array_equal(saved["reached"][:, ~mask], False)
    with np.load(directory / "weight_0/G_stations_sparse.npz") as saved:
        assert len(saved["offsets"]) == len(mask) + 1
        assert not np.diff(saved["offsets"])[~mask].any()


def test_unobserved_fields_do_not_affect_sparse_scores_or_refinement():
    observed, fields, settings = _problem()
    mask = np.array([True, False, True, False, True])
    arrivals = np.where(mask, observed, np.nan)
    expected = _process(arrivals, fields, settings, _fake_rays)
    poisoned = fields.times.copy()
    poisoned[~mask] = np.nan
    # The full-network normalization is now NaN: sparse events must not reuse it.
    fields = events.IterationFields.from_times(poisoned, fields.stations, 1., 2, settings.noise)
    actual = _process(arrivals, fields, settings, _fake_rays)
    for a, b in zip(actual[:2], expected[:2]):
        np.testing.assert_array_equal(a, b)
    for name in ("candidate_cells", "positions", "weights", "misfit"):
        np.testing.assert_array_equal(getattr(actual[2], name), getattr(expected[2], name))


def test_sparse_likelihood_and_refinement_keep_shared_field_volumes(monkeypatch):
    from instruments import hypocentre_objective, likelihood

    observed, fields, settings = _problem()
    indices = np.array([0, 2, 4])
    original_score = likelihood._indexed_likelihood_kernel
    original_objective = hypocentre_objective._refinement_objective_kernel
    calls = []

    def checked_score(predicted, *args):
        assert np.shares_memory(predicted, fields.times)
        calls.append("score")
        return original_score(predicted, *args)

    def checked_objective(predicted, *args):
        assert predicted is fields.times
        calls.append("refinement")
        return original_objective(predicted, *args)

    monkeypatch.setattr(likelihood, "_indexed_likelihood_kernel", checked_score)
    monkeypatch.setattr(hypocentre_objective, "_refinement_objective_kernel", checked_objective)
    likelihood.negative_log_likelihood(fields.times, observed[indices], settings.noise, station_indices=indices)
    objective = hypocentre_objective.make_refinement_objective(
        fields.times, observed[indices], settings.noise, station_indices=indices,
    )
    objective([1., 2., 1.])
    assert calls == ["score", "refinement"]


def test_sparse_observation_masks_survive_disabled_optional_diagnostics():
    observed, fields, settings = _problem()
    mask = np.array([False, True, False, True, False])
    arrivals = np.where(mask, observed, np.nan)
    reference = _process(arrivals, fields, settings, _fake_rays)
    actual = _process(arrivals, fields, replace(settings, log_diagnostics=False,
                                               log_g_per_weight=False, log_misfit=False), _fake_rays)
    for a, b in zip(actual[:2], reference[:2]):
        np.testing.assert_array_equal(a, b)
    log = actual[2]
    assert log.residuals.size == 0 and not log.ray_count_per_weight
    assert log.G_per_weight is None and log.misfit is None
    np.testing.assert_array_equal(log.observed_mask, mask)
    assert summarize_ray_status([(0, log)], len(mask)) == summarize_ray_status([(0, reference[2])], len(mask))


def test_missing_picks_are_not_ray_failures():
    observed, fields, settings = _problem()
    mask = np.array([False, True, False, True, False])
    _, _, log = _process(np.where(mask, observed, np.nan), fields, settings, _fake_rays)
    summary = summarize_ray_status([(0, log)], len(mask))
    assert summary["expected"] == summary["reached"] == summary["used"] == 2 * np.count_nonzero(log.weights)
    assert summary["failed"] == 0 and summary["failures"] == []


@pytest.mark.parametrize("bad, message", [
    ([0., np.inf, 1., np.nan, 2.], "inf"),
    ([0., -np.inf, 1., 2., 3.], "inf"),
    ([np.nan] * 5, "at least two"),
    ([np.nan, 1., np.nan, np.nan, np.nan], "at least two"),
    ([0., 1.], "shape"),
])
def test_invalid_observations_fail_before_event_work(bad, message):
    _, fields, settings = _problem()
    with pytest.raises(ValueError, match=message):
        _process(np.asarray(bad), fields, settings)
    with pytest.raises(ValueError, match=message):
        events.run_events([bad], fields, settings, n_workers=2)


def test_invalid_arrivals_rejected_before_field_or_inverse_solves(monkeypatch):
    from config import InversionConfig
    from tomography import tomography_em as em
    from velocity_model import VelocityModel

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid observations reached a numerical solve")

    monkeypatch.setattr(em, "station_travel_time_fields", forbidden)
    monkeypatch.setattr(em, "solve_slowness_update", forbidden)
    model = VelocityModel(np.full((2, 2, 2), 5000.), 1000.)
    config = InversionConfig(save_runs=False)
    stations = np.zeros((3, 3))
    for entry in (em.run_em, em.make_tomography_step):
        with pytest.raises(ValueError, match="at least two"):
            if entry is em.run_em:
                entry(config, model, [[np.nan, 1., np.nan]], stations, PickNoise())
            else:
                entry(model, [[np.nan, 1., np.nan]], stations, PickNoise(), config)


def test_complete_events_preserve_batch_path_and_legacy_logs(monkeypatch):
    observed, fields, settings = _problem()
    monkeypatch.setattr(events, "compute_G_all_stations_compact", _fake_rays)
    arrivals = np.stack([observed, observed + 0.2])
    reference = events.run_events(arrivals, fields, replace(settings, likelihood_batch_size=1))
    original = events.negative_log_likelihood_batch
    calls = []

    def checked(*args):
        calls.append(args[1].shape)
        return original(*args)

    monkeypatch.setattr(events, "negative_log_likelihood_batch", checked)
    actual = events.run_events(arrivals, fields, settings)
    assert calls == [(2, len(observed))]
    for a, b in zip(actual[:2], reference[:2]):
        np.testing.assert_array_equal(a, b)
    for (_, a), (_, b) in zip(actual[2], reference[2]):
        assert a.observed_mask is None
        np.testing.assert_array_equal(a.positions, b.positions)
        np.testing.assert_array_equal(a.residuals, b.residuals)
    summary = summarize_ray_status(actual[2], len(observed))
    assert summary["expected"] == summary["reached"] == 2 * settings.weights_top_n * len(observed)
