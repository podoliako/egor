"""Regression tests for worker-side normal-equation aggregation."""

import numpy as np

import tomography.tomography_events as event_module
from tomography.tomography_events import (
    _aggregate_event_results,
    _mp_event_chunk_task,
    _partition_event_tasks,
)
from tomography.tomography_math import _normal_equation_contribution


def test_event_tasks_are_partitioned_once_into_balanced_ordered_chunks():
    arrivals = [[float(i)] for i in range(11)]
    chunks = _partition_event_tasks(arrivals, n_chunks=4)

    assert [chunk_idx for chunk_idx, _tasks in chunks] == [0, 1, 2, 3]
    assert [len(tasks) for _chunk_idx, tasks in chunks] == [3, 3, 3, 2]
    assert [
        event_idx
        for _chunk_idx, tasks in chunks
        for event_idx, _observed in tasks
    ] == list(range(11))


def test_more_workers_than_events_does_not_create_empty_chunks():
    chunks = _partition_event_tasks([[1.0], [2.0]], n_chunks=8)

    assert len(chunks) == 2
    assert all(len(tasks) == 1 for _chunk_idx, tasks in chunks)


def test_empty_arrival_table_is_rejected():
    try:
        _partition_event_tasks([], n_chunks=2)
    except ValueError as error:
        assert "at least one event" in str(error)
    else:
        raise AssertionError("An empty arrival table must be rejected")


def test_event_result_aggregation_preserves_logs_and_sums_equations():
    results = [
        (
            event_idx,
            np.full((2, 2), event_idx + 1.0),
            np.full(2, 10.0 + event_idx),
            {"marker": event_idx},
        )
        for event_idx in range(3)
    ]

    hessian, rhs, logs = _aggregate_event_results(iter(results))

    np.testing.assert_allclose(hessian, np.full((2, 2), 6.0))
    np.testing.assert_allclose(rhs, np.full(2, 33.0))
    assert logs == [
        (0, {"marker": 0}),
        (1, {"marker": 1}),
        (2, {"marker": 2}),
    ]


def test_chunk_worker_returns_one_normal_system_with_explicit_event_indices():
    original = event_module._mp_event_task

    def fake_event_task(task):
        event_idx, _observed = task
        return (
            np.full((2, 2), event_idx + 1.0),
            np.full(2, event_idx + 0.5),
            ("event", event_idx),
        )

    event_module._mp_event_task = fake_event_task
    try:
        chunk_idx, hessian, rhs, logs = _mp_event_chunk_task(
            (3, [(4, [1.0]), (5, [2.0])])
        )
    finally:
        event_module._mp_event_task = original

    assert chunk_idx == 3
    np.testing.assert_allclose(hessian, np.full((2, 2), 11.0))
    np.testing.assert_allclose(rhs, np.full(2, 10.0))
    assert logs == [(4, ("event", 4)), (5, ("event", 5))]


def test_sparse_normal_equations_match_explicit_dense_centering():
    rng = np.random.default_rng(17)
    sensitivities = np.zeros((7, 3, 2, 2), dtype=np.float64)
    flat = sensitivities.reshape(7, -1)
    flat[:, [1, 4, 9]] = rng.normal(size=(7, 3))
    residuals = rng.normal(size=7)
    valid = np.asarray([True, True, False, True, True, False, True])
    weight = 0.37

    hessian, rhs = _normal_equation_contribution(
        sensitivities, residuals, sensitivities.shape[1:], weight, valid
    )
    rows = flat[valid]
    centered_rows = rows - rows.mean(axis=0, keepdims=True)
    centered_residuals = residuals[valid] - residuals[valid].mean()
    scale = weight * len(rows)

    np.testing.assert_allclose(hessian, scale * centered_rows.T @ centered_rows)
    np.testing.assert_allclose(rhs, scale * centered_rows.T @ centered_residuals)
    assert not np.any(hessian[0])
    assert not np.any(hessian[:, 0])


def test_512_events_produce_at_most_one_dense_result_per_worker():
    chunks = _partition_event_tasks([[0.0]] * 512, n_chunks=24)

    assert len(chunks) == 24
    assert max(len(tasks) for _chunk_idx, tasks in chunks) == 22
    assert min(len(tasks) for _chunk_idx, tasks in chunks) == 21
