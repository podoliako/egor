"""Hard and soft updates must use the same refined shortlist and noise model."""

import numpy as np
import pytest

import tomography.tomography_events as events
from instruments.instruments_ops import restriction_tables
from instruments.likelihood import PickNoise


def _setup(candidate_mode):
    times = np.array([[[[1.]], [[1.]]], [[[2.]], [[2.2]]]], dtype=np.float32)
    zeros = np.zeros_like(times)
    fields = events.IterationFields(times, zeros, zeros, zeros, np.zeros((2, 3)), 1., (2, 1, 1),
                                    restriction_tables((2, 1, 1), 1))
    settings = events.EventSettings(
        subdivision=1, slowness_interpolation="nearest", n_candidates=2, weights_top_n=2,
        weights_min_distance=1, candidate_mode=candidate_mode, temperature=1.,
        noise=PickNoise(0., 0.1, 0.),
    )
    return fields, settings


def test_hard_and_soft_modes_use_identical_refined_candidates(monkeypatch):
    monkeypatch.setattr(
        events, "refine_hypocentre_in_cell",
        lambda times, observed, cell, noise: (np.asarray(cell, dtype=float), 0.0),
    )
    calls = []

    def rays(gx, gy, gz, stations, position, *args):
        calls.append(tuple(position))
        G = np.zeros((2, 2, 1, 1))
        G[0, int(position[0]), 0, 0] = 1.0
        return G, np.array([True, True])

    results = {}
    for mode in ("soft", "hard"):
        fields, settings = _setup(mode)
        hessian, rhs = np.zeros((2, 2)), np.zeros(2)
        calls.clear()
        log = events.process_event(np.array([0., 1.]), fields, settings, hessian, rhs, rays)
        results[mode] = (hessian, log, list(calls))

    soft_h, soft_log, soft_calls = results["soft"]
    hard_h, hard_log, hard_calls = results["hard"]
    assert soft_calls == [(0., 0., 0.), (1., 0., 0.)]
    assert np.all(soft_log.weights > 0)
    assert soft_log.weights.sum() == pytest.approx(1.)
    assert hard_calls == [(0., 0., 0.)]
    np.testing.assert_array_equal(hard_log.weights, [1., 0.])
    np.testing.assert_array_equal(hard_log.positions, soft_log.positions)
    assert hard_h[0, 0] > soft_h[0, 0]


def test_invalid_candidate_mode_rejected_before_tracing():
    fields, settings = _setup("invalid")
    with pytest.raises(ValueError, match="candidate_mode"):
        events.process_event(np.array([0., 1.]), fields, settings, np.zeros((2, 2)), np.zeros(2), None)


def test_best_refined_candidates_are_kept_even_if_grid_ranked_lower(monkeypatch):
    # Grid ranks cell 0 first, but refinement makes cell 1 the better hypothesis.
    refined_cost = {0: 5.0, 1: 0.0}
    monkeypatch.setattr(
        events, "refine_hypocentre_in_cell",
        lambda times, observed, cell, noise: (np.asarray(cell, dtype=float) + 0.25,
                                              refined_cost[int(cell[0])]),
    )
    fields, settings = _setup("soft")
    settings = events.EventSettings(**{**settings.__dict__, "weights_top_n": 1})
    log = events.process_event(np.array([0., 1.]), fields, settings, np.zeros((2, 2)), np.zeros(2),
                               lambda *args: (np.zeros((2, 2, 1, 1)), np.array([True, True])))
    np.testing.assert_array_equal(log.candidate_cells, [[1, 0, 0]])
    np.testing.assert_allclose(log.positions, [[1.25, 0.25, 0.25]])
    np.testing.assert_array_equal(log.weights, [1.0])
