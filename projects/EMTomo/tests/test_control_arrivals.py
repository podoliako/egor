"""Held-out splits and training-only predictive validation."""
from types import SimpleNamespace

import numpy as np
import pytest

from eikonal import station_travel_time_fields
from instruments.instruments_coords import sample_cell_centered_trilinear_batch
from instruments.likelihood import PickNoise
from tomography import tomography_validation as validation
from velocity_model import VelocityModel


def _config(**overrides):
    values = dict(subdivision=1, slowness_interpolation="nearest", n_workers=1,
                  n_candidates=2, weights_min_distance=1, fast_refinement=True,
                  temperature=1., weights_top_n=1, candidate_mode="soft")
    return SimpleNamespace(**{**values, **overrides})


def test_split_reproducible_exact_global_proportion_and_immutable():
    arrivals = np.arange(2000., dtype=float).reshape(100, 20)
    before = arrivals.copy()
    arrivals.flags.writeable = False
    kwargs = dict(fraction=0.137, seed=42, min_training_picks=5)
    training, mask = validation.split_control_arrivals(arrivals, **kwargs)
    again, same_mask = validation.split_control_arrivals(arrivals, **kwargs)
    _, other_mask = validation.split_control_arrivals(arrivals, **{**kwargs, "seed": 43})
    assert mask.dtype == np.bool_
    assert mask.sum() == round(0.137 * arrivals.size)
    assert np.all(np.isfinite(training).sum(axis=1) >= 5)
    np.testing.assert_array_equal(mask, same_mask)
    np.testing.assert_array_equal(training, again)
    assert not np.array_equal(mask, other_mask)
    assert not np.shares_memory(training, arrivals)
    np.testing.assert_array_equal(arrivals, before)
    np.testing.assert_array_equal(training[~mask], arrivals[~mask])
    assert np.isnan(training[mask]).all()


def test_split_targets_all_finite_picks_and_respects_sparse_capacity():
    arrivals = np.ones((5, 10))
    for event, count in enumerate([0, 2, 5, 6, 10]):
        arrivals[event, count:] = np.nan
    before = arrivals.copy()
    # 23 total finite picks, but only 1 + 5 removable; rounding 0.2 * 23 gives 5.
    for fraction, expected in [(0.2, 5), (0.9, 6), (1., 6)]:
        training, mask = validation.split_control_arrivals(
            arrivals, fraction=fraction, seed=5, min_training_picks=5,
        )
        assert mask.sum() == expected
        assert not mask[:3].any()
        assert not mask[np.isnan(arrivals)].any()
        assert np.all(np.isfinite(training[3:]).sum(axis=1) >= 5)
        np.testing.assert_array_equal(training[:3], arrivals[:3])
    np.testing.assert_array_equal(arrivals, before)


@pytest.mark.parametrize("shape,fraction,minimum", [
    ((3, 8), 0., 5), ((3, 5), 0.5, 5), ((3, 3), 1., 5),
    ((1, 6), 0.001, 5), ((0, 8), 0.2, 5), ((2, 0), 0.2, 5),
])
def test_noop_split_preserves_original_identity(shape, fraction, minimum):
    arrivals = np.ones(shape, dtype=np.float32)
    training, mask = validation.split_control_arrivals(
        arrivals, fraction=fraction, seed=0, min_training_picks=minimum,
    )
    assert training is arrivals
    assert mask.shape == arrivals.shape
    assert not mask.any()


def test_split_does_not_change_global_rng():
    np.random.seed(17)
    expected = np.random.random(5)
    np.random.seed(17)
    validation.split_control_arrivals(np.ones((5, 10)), fraction=0.2, seed=42, min_training_picks=5)
    np.testing.assert_array_equal(np.random.random(5), expected)


@pytest.mark.parametrize("fraction", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_split_fraction(fraction):
    with pytest.raises(ValueError, match="fraction"):
        validation.split_control_arrivals(np.ones((2, 8)), fraction=fraction, seed=0, min_training_picks=5)


@pytest.mark.parametrize("minimum", [1, -1, 2.5, True])
def test_invalid_training_minimum(minimum):
    with pytest.raises(ValueError, match="min_training_picks"):
        validation.split_control_arrivals(np.ones((2, 8)), fraction=0.2, seed=0, min_training_picks=minimum)


def test_split_rejects_infinite_and_non_table_arrivals():
    for arrivals, message in [(np.ones(8), "shape"), (np.array([[1., np.inf]]), "inf")]:
        with pytest.raises(ValueError, match=message):
            validation.split_control_arrivals(arrivals, fraction=0.2, seed=0, min_training_picks=5)


def _constant_problem(monkeypatch):
    predicted = np.array([1., 2., 4., 8., 16.])
    fields = np.broadcast_to(predicted[:, None, None, None], (5, 2, 2, 2)).copy()
    model = VelocityModel(np.full((2, 2, 2), 2000.), 100.)
    stations = np.zeros((5, 3))
    noise = PickNoise(0.2, 0.1, 0.3)
    mask = np.array([[False, False, False, False, True],
                     [False, True, False, True, True],
                     [False, False, False, False, False]])
    training = np.array([[11., 13., 13., 20., np.nan],
                         [21., np.nan, 26., np.nan, np.nan],
                         [np.nan] * 5])
    origins = []
    for row in training[:2]:
        indices = np.flatnonzero(np.isfinite(row))
        precision = 1 / noise.sigmas(predicted[indices]) ** 2
        origins.append(np.average(row[indices] - predicted[indices], weights=precision))
    arrivals = training.copy()
    arrivals[0, 4] = predicted[4] + origins[0] + 3.
    arrivals[1, mask[1]] = predicted[mask[1]] + origins[1] + np.array([1., 2., 6.])
    monkeypatch.setattr(validation, "station_travel_time_fields", lambda *args: fields)
    calls = []

    def refine(times, picks, cell, given_noise, *, accelerated, station_indices):
        assert times is fields  # Never materialize a station-subset volume.
        assert given_noise is noise
        calls.append((picks.copy(), station_indices.copy(), accelerated))
        return np.asarray(cell, dtype=float), 0.

    monkeypatch.setattr(validation, "refine_hypocentre_in_cell", refine)
    return model, training, arrivals, mask, stations, noise, fields, calls


@pytest.mark.parametrize("fast", [False, True])
def test_weighted_training_origin_and_raw_global_rmse(monkeypatch, fast):
    model, training, arrivals, mask, stations, noise, _, calls = _constant_problem(monkeypatch)
    before_training, before_arrivals, before_mask = training.copy(), arrivals.copy(), mask.copy()
    result = validation.evaluate_control_rmse(
        model, training, arrivals, mask, stations, noise, _config(fast_refinement=fast),
    )
    # Global pick weighting, not mean(event RMSE), likelihood or normalized residuals.
    assert result == pytest.approx(np.sqrt((3**2 + 1**2 + 2**2 + 6**2) / 4))
    assert len(calls) == 4  # Two candidates for each of two scored events, not the third.
    for picks, indices, accelerated in calls[:2]:
        np.testing.assert_array_equal(indices, [0, 1, 2, 3])
        np.testing.assert_array_equal(picks, training[0, indices])
        assert accelerated is fast
    for picks, indices, _ in calls[2:]:
        np.testing.assert_array_equal(indices, [0, 2])
        np.testing.assert_array_equal(picks, training[1, indices])
    np.testing.assert_array_equal(training, before_training)
    np.testing.assert_array_equal(arrivals, before_arrivals)
    np.testing.assert_array_equal(mask, before_mask)


def test_control_values_never_affect_localization_or_origin(monkeypatch):
    model, training, arrivals, mask, stations, noise, _, calls = _constant_problem(monkeypatch)
    args = (model, training, arrivals, mask, stations, noise, _config())
    baseline = validation.evaluate_control_rmse(*args)
    original_calls = list(calls)
    calls.clear()
    changed = arrivals.copy()
    changed[mask] += 10.
    # The full table's training values must not be consulted, either.
    changed[~mask] = -1e6
    result = validation.evaluate_control_rmse(model, training, changed, mask, stations, noise, _config())
    assert result == pytest.approx(np.sqrt(np.mean(np.array([13., 11., 12., 16.]) ** 2)))
    for actual, expected in zip(calls, original_calls):
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
    # A common absolute-origin shift per event leaves predictive errors unchanged.
    shifted_training = training + np.array([[100.], [-200.], [0.]])
    shifted_arrivals = arrivals + np.array([[100.], [-200.], [0.]])
    assert validation.evaluate_control_rmse(
        model, shifted_training, shifted_arrivals, mask, stations, noise, _config(),
    ) == pytest.approx(baseline)


@pytest.mark.parametrize("temperature,mode,top_n", [(1e-9, "hard", 1), (1e9, "soft", 2)])
def test_best_refined_candidate_not_best_grid_or_posterior_average(monkeypatch, temperature, mode, top_n):
    fields = np.array([1., 2., 2., 4., 100., 8.]).reshape(3, 2, 1, 1)
    model = SimpleNamespace(refined=lambda *args: model)
    monkeypatch.setattr(validation, "station_travel_time_fields", lambda *args: fields)
    training = np.array([[12., 14., np.nan]])
    arrivals = np.array([[12., 14., 18.]])
    mask = np.array([[False, False, True]])
    calls = []

    def likelihood(times, picks, noise, *, station_indices):
        assert times is fields
        np.testing.assert_array_equal(picks, [12., 14.])
        np.testing.assert_array_equal(station_indices, [0, 1])
        return np.array([0., 1.]).reshape(2, 1, 1)

    def refine(times, picks, cell, noise, **kwargs):
        calls.append(tuple(cell))
        return cell.astype(float), 5. if cell[0] == 0 else 0.

    monkeypatch.setattr(validation, "negative_log_likelihood", likelihood)
    monkeypatch.setattr(validation, "refine_hypocentre_in_cell", refine)
    config = _config(temperature=temperature, candidate_mode=mode, weights_top_n=top_n)
    assert validation.evaluate_control_rmse(
        model, training, arrivals, mask, np.zeros((3, 3)), PickNoise(), config,
    ) == pytest.approx(0.)
    assert calls == [(0, 0, 0), (1, 0, 0)]


def test_recomputes_fields_from_supplied_model_and_honors_settings(monkeypatch):
    _, training, arrivals, mask, stations, noise, fields, _ = _constant_problem(monkeypatch)
    fine = object()
    calls = []

    def refine_model(subdivision, interpolation):
        calls.append((subdivision, interpolation))
        return fine

    def fmm(model, coords, workers):
        assert model is fine
        np.testing.assert_array_equal(coords, stations)
        calls.append(workers)
        return fields

    select = validation.select_candidate_cells

    def shortlist(cost, n, distance):
        calls.append((n, distance))
        return select(cost, n, distance)

    # Execute callbacks in this process: the assertions below deliberately inspect
    # their side effects, which a real fork pool would keep in its children.
    import multiprocessing as mp

    class InlinePool:
        def __init__(self, processes, initializer, initargs):
            assert 1 < processes <= 4
            initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def imap(self, function, events, chunksize=1):
            assert chunksize == 1
            return map(function, events)

    def context(method):
        assert method == "fork"
        return SimpleNamespace(Pool=InlinePool)

    monkeypatch.setattr(mp, "get_context", context)
    monkeypatch.setattr(validation, "station_travel_time_fields", fmm)
    monkeypatch.setattr(validation, "select_candidate_cells", shortlist)
    model = SimpleNamespace(refined=refine_model)
    config = _config(subdivision=3, slowness_interpolation="trilinear", n_workers=4,
                     n_candidates=3, weights_min_distance=2)
    for _ in range(2):
        validation.evaluate_control_rmse(model, training, arrivals, mask, stations, noise, config)
    assert calls == [(3, "trilinear"), 4, (3, 2), (3, 2)] * 2


@pytest.mark.parametrize("problem,message", [
    ("empty", "at least one"), ("missing_control", "Control arrivals must be finite"),
    ("overlap", "excluded"), ("insufficient", "at least two training"),
    ("infinite_arrival", "inf"), ("field", "travel-time field"),
    ("likelihood", "training likelihood"), ("refinement", "refined candidate"),
    ("prediction", "predicted arrivals"),
])
def test_invalid_control_data_and_nonfinite_predictions_fail_clearly(monkeypatch, problem, message):
    model, training, arrivals, mask, stations, noise, fields, _ = _constant_problem(monkeypatch)
    if problem == "empty":
        mask[:] = False
    elif problem == "missing_control":
        arrivals[0, 4] = np.nan
    elif problem == "overlap":
        training[0, 4] = arrivals[0, 4]
    elif problem == "insufficient":
        training[0, :3] = np.nan
    elif problem == "infinite_arrival":
        arrivals[0, 4] = np.inf
    elif problem == "field":
        fields[4, 0, 0, 0] = np.nan
    elif problem == "likelihood":
        monkeypatch.setattr(validation, "negative_log_likelihood", lambda *args, **kwargs: np.full((2, 2, 2), np.nan))
    elif problem == "refinement":
        monkeypatch.setattr(validation, "refine_hypocentre_in_cell", lambda *args, **kwargs: (np.zeros(3), np.nan))
    elif problem == "prediction":
        monkeypatch.setattr(validation, "sample_cell_centered_trilinear_batch",
                            lambda *args, station_indices: np.full(len(station_indices), np.nan))
    with pytest.raises(ValueError, match=message):
        validation.evaluate_control_rmse(model, training, arrivals, mask, stations, noise, _config())


@pytest.mark.parametrize("fast", [False, True])
def test_real_tiny_fmm_localization_and_updated_model(fast):
    model = VelocityModel(np.full((4, 4, 4), 2000.), 100.)
    stations = np.array([[0., 0., 0.], [400., 0., 0.], [0., 400., 0.],
                         [0., 0., 400.], [400., 400., 400.], [400., 400., 0.]])
    config = _config(subdivision=2, slowness_interpolation="trilinear", n_candidates=4,
                     fast_refinement=fast)
    fields = station_travel_time_fields(model.refined(2, "trilinear"), stations)
    arrivals = sample_cell_centered_trilinear_batch(fields, [3., 4., 3.])[None, :].astype(float) + 2.5
    mask = np.array([[False, False, False, False, False, True]])
    training = arrivals.copy()
    training[mask] = np.nan
    noise = PickNoise(0., 0.01, 0.)
    rmse = validation.evaluate_control_rmse(model, training, arrivals, mask, stations, noise, config)
    assert rmse < 1e-5
    different_model = VelocityModel(np.full((4, 4, 4), 3000.), 100.)
    other_rmse = validation.evaluate_control_rmse(
        different_model, training, arrivals, mask, stations, noise, config,
    )
    assert other_rmse > rmse + 1e-4
