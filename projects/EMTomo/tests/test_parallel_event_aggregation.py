"""Event partitioning and serial/parallel equivalence of the summed normal equations."""

import numpy as np
import pytest

from eikonal import station_travel_time_fields
from instruments.instruments_coords import metric_to_cell_coord, sample_cell_centered_trilinear_batch
from instruments.likelihood import PickNoise
from tomography.tomography_em import warm_up_jit
from tomography.tomography_events import (
    EventSettings,
    IterationFields,
    partition_events,
    run_events,
)
from velocity_model import VelocityModel


def test_events_are_partitioned_into_balanced_ordered_chunks():
    chunks = partition_events(11, n_chunks=4)
    assert [len(chunk) for chunk in chunks] == [3, 3, 3, 2]
    assert [event for chunk in chunks for event in chunk] == list(range(11))
    assert len(partition_events(2, n_chunks=8)) == 2
    chunks = partition_events(512, n_chunks=24)
    assert {len(chunk) for chunk in chunks} == {21, 22}
    with pytest.raises(ValueError, match="At least one event"):
        partition_events(0, n_chunks=2)


def _small_problem(subdivision=2):
    rng = np.random.default_rng(4)
    coarse = VelocityModel(rng.uniform(4500., 5500., size=(4, 3, 3)), 1000.)
    fine = coarse.refined(subdivision)
    stations = np.array([[200., 300., 0.], [3700., 400., 0.], [1900., 2800., 0.],
                         [600., 2500., 0.], [3300., 2600., 0.]])
    times = station_travel_time_fields(fine, stations)
    events = rng.uniform([300., 300., 800.], [3700., 2700., 2700.], size=(7, 3))
    arrivals = np.stack([
        sample_cell_centered_trilinear_batch(times, metric_to_cell_coord(event, fine.cell_size))
        for event in events
    ])
    arrivals -= arrivals.min(axis=1, keepdims=True)
    noise = PickNoise(0.01, 0.05, 0.1)
    fields = IterationFields.from_times(
        times, metric_to_cell_coord(stations, fine.cell_size), fine.cell_size, subdivision, noise,
    )
    settings = EventSettings(
        subdivision=subdivision, slowness_interpolation="nearest", n_candidates=5, weights_top_n=3,
        weights_min_distance=2, candidate_mode="soft", temperature=1., noise=noise,
    )
    return arrivals, fields, settings


def test_parallel_events_match_serial_sum_and_logs():
    warm_up_jit()
    arrivals, fields, settings = _small_problem()
    serial = run_events(arrivals, fields, settings, n_workers=1)
    parallel = run_events(arrivals, fields, settings, n_workers=3)
    np.testing.assert_allclose(parallel[0], serial[0], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(parallel[1], serial[1], rtol=1e-10, atol=1e-12)
    assert [event for event, _ in parallel[2]] == list(range(len(arrivals)))
    for (_, a), (_, b) in zip(serial[2], parallel[2]):
        np.testing.assert_allclose(a.weights, b.weights)
        np.testing.assert_allclose(a.positions, b.positions)
    assert np.any(serial[0]) and np.allclose(serial[0], serial[0].T)
