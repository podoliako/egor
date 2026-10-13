"""Hard and soft updates must use the same refined shortlist and noise model."""

from dataclasses import replace

import numpy as np
import pytest

import tomography.tomography_events as events
from instruments.instruments_ops import restriction_tables
from instruments.likelihood import PickNoise
from tomography.tomography_logging import summarize_ray_status


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
    assert summarize_ray_status([(0, soft_log)], 2)["expected"] == 4
    assert summarize_ray_status([(0, hard_log)], 2)["expected"] == 2


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


@pytest.mark.parametrize("mode", ["soft", "hard"])
@pytest.mark.parametrize("log_g", [False, True])
def test_ray_masks_are_copied_and_fine_retraces_are_not_counted(monkeypatch, mode, log_g):
    monkeypatch.setattr(
        events, "refine_hypocentre_in_cell",
        lambda times, observed, cell, noise: (np.asarray(cell, dtype=float), 0.0),
    )
    fields, settings = _setup(mode)
    settings = replace(settings, log_g_per_weight=log_g)
    shared = np.zeros(2, dtype=bool)
    calls = []

    def rays(gx, gy, gz, stations, position, *args):
        fine_retrace = log_g and len(calls) % 2 == 1
        calls.append(tuple(position))
        shared[:] = False if fine_retrace else [True, int(position[0]) == 0]
        # Zero-length reached rays must count as success even with all-zero G.
        return np.zeros((2, 2, 1, 1)), shared

    log = events.process_event(np.array([0., 1.]), fields, settings,
                               np.zeros((2, 2)), np.zeros(2), rays)
    n_active = 2 if mode == "soft" else 1
    assert len(calls) == n_active * (2 if log_g else 1)
    assert list(log.ray_reached_per_weight) == list(range(n_active))
    np.testing.assert_array_equal(log.ray_reached_per_weight[0], [True, True])
    if mode == "soft":
        np.testing.assert_array_equal(log.ray_reached_per_weight[1], [True, False])
        assert not np.shares_memory(log.ray_reached_per_weight[0], log.ray_reached_per_weight[1])
    shared[:] = False
    for reached in log.ray_reached_per_weight.values():
        assert reached.dtype == np.bool_
        assert not np.shares_memory(reached, shared)
    summary = summarize_ray_status([(7, log)], 2)
    assert summary["expected"] == n_active * 2
    assert summary["reached"] == (3 if mode == "soft" else 2)
    assert summary["used"] == 2
    assert summary["failed"] == (1 if mode == "soft" else 0)
    assert summary["failures"] == ([{"event": 7, "hypothesis": 1, "stations": [1]}]
                                   if mode == "soft" else [])


@pytest.mark.parametrize("mask", [[True], [[True, False]], [True, False, True]])
def test_process_event_validates_reached_mask_shape(mask):
    fields, settings = _setup("hard")
    with pytest.raises(ValueError, match="must have shape \\(2,\\)"):
        events.process_event(
            np.array([0., 1.]), fields, settings, np.zeros((2, 2)), np.zeros(2),
            lambda *args: (np.zeros((2, 2, 1, 1)), mask),
        )


def test_soft_zero_weight_hypotheses_are_not_expected(monkeypatch):
    monkeypatch.setattr(events, "candidate_weights", lambda *args: np.array([0., 1.]))
    fields, settings = _setup("soft")
    log = events.process_event(
        np.array([0., 1.]), fields, settings, np.zeros((2, 2)), np.zeros(2),
        lambda *args: (np.zeros((2, 2, 1, 1)), np.array([False, False])),
    )
    assert list(log.ray_reached_per_weight) == [1]
    summary = summarize_ray_status([(0, log)], 2)
    assert summary["expected"] == summary["failed"] == 2
    assert summary["reached"] == summary["used"] == 0
    assert summary["hypotheses_with_insufficient_stations"] == 1


def test_real_boundary_stall_is_counted_as_loss_but_zero_length_success_is_not(monkeypatch):
    from raytracing import compute_G_all_stations_serial

    shape = (5, 1, 1)
    times = np.ones((3, *shape), dtype=np.float32)
    gx, zeros = np.zeros_like(times), np.zeros_like(times)
    gx[0] = 1.0  # Reaches station 0 by moving left.
    gx[1] = -1.0  # Moves into the right boundary and stalls, never reaching station 1.
    stations = np.array([[0., 0., 0.], [0., 0., 0.], [4., 0., 0.]])
    fields = events.IterationFields(times, gx, zeros, zeros, stations, 1., shape,
                                    restriction_tables(shape, 1))
    _, settings = _setup('hard')
    settings = replace(settings, n_candidates=1, weights_top_n=1)
    monkeypatch.setattr(events, 'RAY_MAX_STEPS', 64)
    monkeypatch.setattr(events, 'refine_hypocentre_in_cell',
                        lambda *args: (np.array([4., 0., 0.]), 0.))
    log = events.process_event(np.zeros(3), fields, settings, np.zeros((5, 5)), np.zeros(5),
                               compute_G_all_stations_serial)
    np.testing.assert_array_equal(log.ray_reached_per_weight[0], [True, False, True])
    summary = summarize_ray_status([(0, log)], 3)
    assert summary['expected'] == 3
    assert summary['reached'] == summary['used'] == 2
    assert summary['failed'] == 1
    assert summary['failures'] == [{'event': 0, 'hypothesis': 0, 'stations': [1]}]
