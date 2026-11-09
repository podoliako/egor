"""Forked held-out scoring preserves training-only inference and ordered SSEs."""
import multiprocessing as mp
import os
import re
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
                  n_candidates=3, weights_min_distance=1, fast_refinement=True,
                  temperature=1., weights_top_n=1, candidate_mode="soft")
    return SimpleNamespace(**{**values, **overrides})


@pytest.fixture
def tiny_problem():
    model = VelocityModel(np.linspace(1900., 2300., 64).reshape(4, 4, 4), 100.)
    stations = np.array([[x, y, z] for x in (0., 400.)
                         for y in (0., 400.) for z in (0., 400.)])
    times = station_travel_time_fields(model, stations, 1)
    positions = [[0.8, 1.2, 1.7], [2.1, 1.4, 0.9], [1.3, 2.2, 1.6]]
    observed = np.full((6, 8), np.nan)
    for event, position, origin in zip((1, 3, 4), positions, (2., -3., 10.)):
        observed[event] = sample_cell_centered_trilinear_batch(times, position) + origin
    # Sparse station rows, uneven held-out counts, and entirely unscored rows.
    observed[1, 0] = np.nan
    observed[3, 1] = np.nan
    mask = np.zeros_like(observed, dtype=bool)
    mask[1, 7] = True
    mask[3, [0, 6]] = True
    mask[4, [1, 5, 7]] = True
    observed[mask] += np.array([.03, -.02, .04, -.01, .05, -.06])
    training = observed.copy()
    training[mask] = np.nan
    noise = PickNoise(.03, .01, .02)
    return model, training, observed, mask, stations, noise


@pytest.mark.parametrize("fast", [False, True])
def test_real_fork_matches_serial_with_sparse_immutable_inputs(tiny_problem, fast, monkeypatch):
    model, training, observed, mask, stations, noise = tiny_problem
    arrays = (model.velocity, training, observed, mask, stations)
    before = [array.copy() for array in arrays]
    for array in arrays:
        array.flags.writeable = False
    # Warm NumPy/Numba and both refinement paths in the parent before any fork.
    serial = validation.evaluate_control_rmse(
        *tiny_problem, _config(fast_refinement=fast),
    )
    parent_pid = os.getpid()
    original = validation._control_event_squared_error

    def child_score(event, times, training, observed, mask, noise, config):
        assert os.getpid() != parent_pid, "event scoring must run in a real fork worker"
        assert mask[event].any()
        for array in (times, training, observed, mask):
            assert not array.flags.writeable
        return original(event, times, training, observed, mask, noise, config)

    monkeypatch.setattr(validation, "_control_event_squared_error", child_score)
    parallel = validation.evaluate_control_rmse(
        *tiny_problem, _config(n_workers=2, fast_refinement=fast),
    )
    assert np.isfinite(serial) and serial > 0
    assert parallel == pytest.approx(serial, rel=2e-15, abs=2e-15)
    for array, original in zip(arrays, before):
        np.testing.assert_array_equal(array, original)
        assert not array.flags.writeable


def _training_only_predictions(problem, config):
    """Independent explicit oracle; no full observed table enters localization."""
    model, training, _, mask, stations, noise = problem
    times = station_travel_time_fields(model, stations, 1)
    predicted = np.full(training.shape, np.nan)
    for event in np.flatnonzero(mask.any(axis=1)):
        indices = np.flatnonzero(np.isfinite(training[event]))
        picks = training[event, indices]
        cost = validation.negative_log_likelihood(times, picks, noise, station_indices=indices)
        cells = validation.select_candidate_cells(cost, config.n_candidates, config.weights_min_distance)
        candidates = [validation.refine_hypocentre_in_cell(
            times, picks, cell, noise, accelerated=config.fast_refinement,
            station_indices=indices,
        ) for cell in cells]
        position, _ = min(candidates, key=lambda candidate: candidate[1])
        travel = sample_cell_centered_trilinear_batch(times, position)
        precision = 1 / noise.sigmas(travel[indices]) ** 2
        origin = np.sum(precision * (picks - travel[indices])) / np.sum(precision)
        predicted[event] = travel + origin
    return predicted


def test_parallel_control_values_cannot_change_localization_or_origin(tiny_problem):
    model, training, observed, mask, stations, noise = tiny_problem
    config = _config()
    predictions = _training_only_predictions(tiny_problem, config)
    # The serial evaluation also warms the complete evaluator before forking.
    baseline = validation.evaluate_control_rmse(*tiny_problem, config)
    expected = np.sqrt(np.mean((observed[mask] - predictions[mask]) ** 2))
    assert baseline == pytest.approx(expected, rel=2e-14, abs=2e-15)
    for shift in (0., 7.):
        changed = observed.copy()
        changed[mask] += shift
        changed[~mask] = -1e6  # Not even the observed table's training picks may leak.
        expected = np.sqrt(np.mean((changed[mask] - predictions[mask]) ** 2))
        actual = validation.evaluate_control_rmse(
            model, training, changed, mask, stations, noise, _config(n_workers=2),
        )
        assert actual == pytest.approx(expected, rel=2e-14, abs=2e-15)


def test_serial_does_not_acquire_a_pool(tiny_problem, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("n_workers=1 must not acquire a multiprocessing context")

    monkeypatch.setattr(mp, "get_context", forbidden)
    assert np.isfinite(validation.evaluate_control_rmse(*tiny_problem, _config()))


def test_parallel_worker_error_identifies_event(tiny_problem, monkeypatch):
    validation.evaluate_control_rmse(*tiny_problem, _config())
    original = validation._control_event_squared_error

    def fail(event, *args):
        if event == 3:
            raise ValueError("deliberate scoring failure")
        return original(event, *args)

    # Fork inherits this local callback; no pickle-only/mock pool substitutes.
    monkeypatch.setattr(validation, "_control_event_squared_error", fail)
    with pytest.raises(RuntimeError, match=r"Failed to score control event 3") as error:
        validation.evaluate_control_rmse(*tiny_problem, _config(n_workers=2))
    assert "deliberate scoring failure" in str(error.value) or (
        "deliberate scoring failure" in str(error.value.__cause__)
    )


@pytest.mark.parametrize("workers", [1, 2])
def test_only_control_events_are_scored_in_order_and_progress_completes(
    tiny_problem, monkeypatch, capsys, workers,
):
    model, training, observed, mask, stations, noise = tiny_problem
    fields = station_travel_time_fields(model, stations, 1)
    monkeypatch.setattr(validation, "station_travel_time_fields", lambda *args: fields)
    calls = []
    submissions = []
    # Positive SSEs for which a different reduction order changes the RMSE.
    contributions = {1: 1e16, 3: 1., 4: 1.}

    def score(event, *args):
        assert mask[event].any(), "unscored sparse rows must never be localized"
        calls.append(int(event))
        return contributions[event]

    class InlinePool:
        def __init__(self, processes, initializer, initargs):
            assert processes == 2
            assert initializer is validation._init_control_worker
            initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def imap(self, function, events, chunksize=1):
            assert function is validation._score_control_event
            assert chunksize == 1
            events = list(events)
            submissions.extend(events)
            return map(function, events)

    def context(method):
        assert method == "fork"
        return SimpleNamespace(Pool=InlinePool)

    monkeypatch.setattr(mp, "get_context", context)
    monkeypatch.setattr(validation, "_control_event_squared_error", score)
    result = validation.evaluate_control_rmse(*tiny_problem, _config(n_workers=workers))
    assert calls == [1, 3, 4]
    assert submissions == ([1, 3, 4] if workers > 1 else [])
    expected_sse = 0.
    for event in (1, 3, 4):
        expected_sse += contributions[event]
    expected = float(np.sqrt(expected_sse / int(mask.sum())))
    reordered = float(np.sqrt((1. + 1. + 1e16) / int(mask.sum())))
    assert expected != reordered  # Ensure this test can detect reordering.
    assert result == expected
    lines = [line for line in capsys.readouterr().out.splitlines() if "[Control]" in line]
    assert len(lines) >= 3  # Field construction, scoring, terminal completion.
    terminal = lines[-1].lower()
    assert re.search(r"complete|finished|done", terminal)
    assert "3" in terminal and "6" in terminal  # Three events, six held-out picks.
    assert re.search(r"elapsed|\d+(?:\.\d+)?\s*s\b", terminal)
