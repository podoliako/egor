"""Contract tests for absolute, pair-ID-stable observation noise."""
from unittest.mock import Mock

import numpy as np
import pytest

from projects.forward_modeling import solver
from projects.forward_modeling.model import ForwardConfig, PointSet, VelocityGrid
from projects.forward_modeling.noise import NoiseConfig, add_arrival_noise


def test_config_and_call_defaults():
    config = NoiseConfig()
    assert config.relative_sigma == 0.01
    assert config.absolute_sigma_s == 0.05
    assert config.seed == 42
    times = np.array([[0., 10., 100.]])
    np.testing.assert_array_equal(
        add_arrival_noise(times, ["e"], ["a", "b", "c"]),
        add_arrival_noise(times, ["e"], ["a", "b", "c"], config),
    )


@pytest.mark.parametrize("field", ["relative_sigma", "absolute_sigma_s"])
@pytest.mark.parametrize("value", [-0.01, -np.inf, np.inf, np.nan])
def test_invalid_sigmas(field, value):
    with pytest.raises((ValueError, TypeError)):
        NoiseConfig(**{field: value})


@pytest.mark.parametrize("seed", [-1, 1.5, 1.0, True, False, np.bool_(True),
                                  np.nan, np.inf, "42", None])
def test_invalid_seeds(seed):
    with pytest.raises((ValueError, TypeError)):
        NoiseConfig(seed=seed)


@pytest.mark.parametrize("seed", [0, 42, np.int64(7)])
def test_nonnegative_integral_seeds(seed):
    config = NoiseConfig(relative_sigma=0, absolute_sigma_s=0, seed=seed)
    assert config.seed == seed


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64])
@pytest.mark.parametrize("config", [NoiseConfig(0, 0), NoiseConfig()])
def test_copied_output_and_no_input_mutation(dtype, config):
    times = np.array([[0, 2, 40], [1, 3, 50]], dtype=dtype)
    original = times.copy()
    times.flags.writeable = False
    event_ids = ["e-b", "e-a"]
    station_ids = ["s-c", "s-a", "s-b"]
    noisy = add_arrival_noise(times, event_ids, station_ids, config)
    assert isinstance(noisy, np.ndarray)
    assert noisy.shape == times.shape
    assert not np.shares_memory(noisy, times)
    np.testing.assert_array_equal(times, original)
    assert event_ids == ["e-b", "e-a"]
    assert station_ids == ["s-c", "s-a", "s-b"]
    if config.relative_sigma == config.absolute_sigma_s == 0:
        np.testing.assert_array_equal(noisy, original)


@pytest.mark.parametrize("bad", [-1., np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("config", [NoiseConfig(), NoiseConfig(0, 0)])
def test_invalid_times_even_when_noise_disabled(bad, config):
    times = np.array([[1., bad], [2., 3.]])
    with pytest.raises((ValueError, TypeError)):
        add_arrival_noise(times, ["e1", "e2"], ["s1", "s2"], config)


@pytest.mark.parametrize("times,event_ids,station_ids", [
    (np.array(1.), ["e"], ["s"]),
    (np.ones(2), ["e"], ["s1", "s2"]),
    (np.ones((1, 1, 1)), ["e"], ["s"]),
    (np.ones((2, 3)), ["e"], ["a", "b", "c"]),
    (np.ones((2, 3)), ["e1", "e2"], ["a", "b"]),
    (np.ones((2, 3)), ["e1", "e2", "e3"], ["a", "b"]),
    (np.ones((2, 2)), ["e", "e"], ["a", "b"]),
    (np.ones((2, 2)), ["e1", "e2"], ["s", "s"]),
])
def test_invalid_shapes_and_duplicate_ids(times, event_ids, station_ids):
    with pytest.raises((ValueError, TypeError)):
        add_arrival_noise(times, event_ids, station_ids)


def test_seed_repeatability_reordering_and_subsets():
    times = np.arange(1., 21.).reshape(4, 5)
    events = ["event-d", "event-a", "event-c", "event-b"]
    stations = ["station-4", "station-1", "station-3", "station-0", "station-2"]
    config = NoiseConfig(0.2, 0.3, 123)
    expected = add_arrival_noise(times, events, stations, config)
    np.testing.assert_array_equal(add_arrival_noise(times, events, stations, config), expected)
    different_seed = add_arrival_noise(times, events, stations, NoiseConfig(0.2, 0.3, 124))
    assert not np.array_equal(different_seed, expected)
    for rows, cols in [([2, 0, 3, 1], [4, 2, 0, 3, 1]),
                       ([3, 1], [4, 0, 2]), ([2], [3])]:
        ix = np.ix_(rows, cols)
        actual = add_arrival_noise(
            times[ix], [events[i] for i in rows], [stations[i] for i in cols], config,
        )
        np.testing.assert_array_equal(actual, expected[ix])


@pytest.mark.parametrize("relative,absolute", [(0., 0.5), (0.15, 0.), (0.15, 0.5)])
def test_residual_mean_variance_and_pairwise_independence(relative, absolute):
    # Six-standard-error bounds leave margin for sampling fluctuations without
    # accepting shared draws, linear addition of sigmas, or a normalized T.
    n = 12000
    times = np.tile([0., 1., 5., 20.], (n, 1))
    events = [f"event-{i}" for i in range(n)]
    noisy = add_arrival_noise(times, events, ["a", "b", "c", "d"],
                              NoiseConfig(relative, absolute, 2026))
    residual = noisy - times
    variance = (relative * times[0]) ** 2 + absolute ** 2
    active = variance > 0
    np.testing.assert_array_equal(residual[:, ~active], 0.)
    z = residual[:, active] / np.sqrt(variance[active])
    assert np.all(np.abs(z.mean(axis=0)) < 6 / np.sqrt(n))
    np.testing.assert_allclose(z.var(axis=0, ddof=1), 1., rtol=6 * np.sqrt(2 / (n - 1)))
    covariance = np.cov(z, rowvar=False)
    off_diagonal = covariance[np.triu_indices(z.shape[1], k=1)]
    assert np.all(np.abs(off_diagonal) < 6 / np.sqrt(n))
    # Different events at the same station must not share a random draw either.
    lag_covariance = np.mean((z[:-1] - z[:-1].mean(axis=0))
                             * (z[1:] - z[1:].mean(axis=0)), axis=0)
    assert np.all(np.abs(lag_covariance) < 6 / np.sqrt(n - 1))


def test_negative_absolute_picks_are_not_clipped_or_normalized():
    times = np.zeros((2000, 3))
    noisy = add_arrival_noise(times, [f"e-{i}" for i in range(len(times))],
                              ["a", "b", "c"], NoiseConfig(0, 1, 99))
    negative_fraction = np.mean(noisy < 0)
    assert 0.45 < negative_fraction < 0.55
    assert np.any(noisy > 0)
    assert not np.any(noisy == 0)


@pytest.fixture
def geometry():
    model = VelocityGrid(np.full((2, 2, 2), 2000.), 10.)
    stations = PointSet(["s-b", "s-a", "s-c"],
                        [[1., 1., 1.], [5., 5., 5.], [15., 15., 15.]])
    events = PointSet([f"e-{i}" for i in range(64)], np.tile([2., 3., 4.], (64, 1)))
    return model, stations, events


def _arrival_matrix(arrivals, events, stations):
    assert [(a.event_id, a.station_id) for a in arrivals] == [
        (e, s) for e in events.ids for s in stations.ids
    ]
    return np.array([a.arrival_time_s for a in arrivals]).reshape(len(events.ids), len(stations.ids))


@pytest.mark.parametrize("config", [NoiseConfig(0.2, 0., 42), NoiseConfig(0., 1., 42)])
def test_compute_arrivals_adds_noise_to_absolute_times_before_normalizing(monkeypatch, geometry, config):
    model, stations, events = geometry
    # Close first two stations allow the earliest station to change. In the
    # relative-only case their ~10 s absolute times, not their ~0 s relative
    # times, must determine their noise amplitude.
    times = np.tile([10., 10.001, 20.], (len(events.ids), 1))
    noisy = add_arrival_noise(times, events.ids, stations.ids, config)
    assert np.any(noisy.argmin(axis=1) != times.argmin(axis=1))
    expected = noisy - noisy.min(axis=1, keepdims=True)
    clean = times - times.min(axis=1, keepdims=True)
    wrong = add_arrival_noise(clean, events.ids, stations.ids, config)
    if config.relative_sigma:
        assert not np.allclose(expected, wrong - wrong.min(axis=1, keepdims=True))
    travel = Mock(side_effect=lambda *args, **kwargs: times.copy())
    monkeypatch.setattr(solver, "compute_travel_times", travel)
    forward = ForwardConfig(refinement=3)
    actual = _arrival_matrix(
        solver.compute_arrivals(model, stations, events, forward, noise=config), events, stations,
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(actual.min(axis=1), 0.)
    assert np.all(actual >= 0)
    travel.assert_called_once_with(model, stations, events, forward, workers=1)


@pytest.mark.parametrize("kwargs", [{}, {"noise": None}, {"noise": NoiseConfig(0, 0)}])
def test_compute_arrivals_noise_is_optional_and_zero_noise_is_exact(monkeypatch, geometry, kwargs):
    model, stations, events = geometry
    times = np.tile([10., 12., 11.], (len(events.ids), 1))
    monkeypatch.setattr(solver, "compute_travel_times", Mock(side_effect=lambda *a, **k: times.copy()))
    actual = _arrival_matrix(solver.compute_arrivals(model, stations, events, **kwargs), events, stations)
    np.testing.assert_array_equal(actual, times - times.min(axis=1, keepdims=True))


def test_noise_is_keyword_only(geometry):
    with pytest.raises(TypeError):
        solver.compute_arrivals(*geometry, ForwardConfig(), NoiseConfig())


def test_check_convergence_uses_clean_absolute_times(monkeypatch, geometry):
    model, stations, events = geometry
    coarse = np.tile([10., 12., 15.], (len(events.ids), 1))
    fine = coarse + [1., 2., 4.]
    travel = Mock(side_effect=[coarse.copy(), fine.copy()])
    monkeypatch.setattr(solver, "compute_travel_times", travel)
    stats = solver.check_convergence(model, stations, events, ForwardConfig(refinement=2))
    assert travel.call_count == 2
    assert travel.call_args_list[0].args == (model, stations, events, ForwardConfig(refinement=2))
    assert travel.call_args_list[1].args == (model, stations, events, ForwardConfig(refinement=4))
    assert travel.call_args_list[0].kwargs == {"workers": 1}
    assert travel.call_args_list[1].kwargs == {"workers": 1}
    assert stats["max_absolute_time_difference_s"] == pytest.approx(4.)
    assert stats["rms_absolute_time_difference_s"] == pytest.approx(np.sqrt(7.))
    assert stats["max_abs_difference_s"] == pytest.approx(3.)
    assert stats["rms_difference_s"] == pytest.approx(np.sqrt(10. / 3.))


def test_compute_travel_times_remains_clean():
    pytest.importorskip("pykonal")
    model = VelocityGrid(np.full((2, 2, 2), 2000.), 10.)
    events = PointSet(["e"], [[5., 5., 5.]])
    stations = PointSet(["same", "near"], [[5., 5., 5.], [6., 5., 5.]])
    expected = [[0., 1. / 2000.]]
    first = solver.compute_travel_times(model, stations, events)
    np.testing.assert_allclose(first, expected, rtol=0, atol=1e-12)
    solver.compute_arrivals(model, stations, events, noise=NoiseConfig(1., 1., 7))
    np.testing.assert_array_equal(solver.compute_travel_times(model, stations, events), first)
