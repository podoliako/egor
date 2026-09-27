"""Numerical checks independent of the inversion's SKFMM implementation."""
import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from projects.forward_modeling import (
    ForwardConfig, PointSet, VelocityGrid, check_convergence,
    compute_arrivals, compute_travel_times,
)
from projects.forward_modeling.solver import _nodal_velocity, _segment_time


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_invalid_velocity(value):
    with pytest.raises(ValueError, match="positive"):
        VelocityGrid(np.full((2, 2, 2), value), 10)


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_invalid_step(value):
    with pytest.raises(ValueError, match="cell_size"):
        VelocityGrid(np.ones((2, 2, 2)), value)


@pytest.mark.parametrize("shape", [(0, 2, 2), (2, 2), (2, 2, 2, 2)])
def test_invalid_shape(shape):
    with pytest.raises(ValueError, match="3-D"):
        VelocityGrid(np.ones(shape), 1)


@pytest.mark.parametrize("kwargs", [{"refinement": 0}, {"refinement": 1.5},
                                     {"refinement": True}, {"source_radius_cells": 0}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        ForwardConfig(**kwargs)


@pytest.mark.parametrize("ids,coords", [([], []), ([1, "1"], [[0, 0, 0]] * 2),
                                         ([""], [[0, 0, 0]]), (["x"], [[np.nan, 0, 0]]),
                                         (["x"], [[0, 0]])])
def test_invalid_points(ids, coords):
    with pytest.raises(ValueError):
        PointSet(ids, coords)


def test_inputs_are_copied_not_modified():
    velocity = np.full((3, 3, 3), 2000.)
    coordinates = np.array([[5., 6., 7.]])
    model = VelocityGrid(velocity, 10.)
    points = PointSet([1], coordinates)
    velocity[:] = 1
    coordinates[:] = 100
    assert np.all(model.velocity == 2000)
    np.testing.assert_array_equal(points.coordinates_m, [[5, 6, 7]])
    result = compute_arrivals(model, points, points)
    assert result[0].arrival_time_s == 0
    assert result[0].station_id == result[0].event_id == "1"


@pytest.mark.parametrize("position", [[-0.01, 0, 0], [30.01, 0, 0]])
def test_outside_domain_is_rejected_not_clipped(position):
    model = VelocityGrid(np.full((3, 3, 3), 2000.), 10.)
    inside = PointSet(["inside"], [[10, 10, 10]])
    outside = PointSet(["outside"], [position])
    with pytest.raises(ValueError, match="stations outside"):
        compute_arrivals(model, outside, inside)
    with pytest.raises(ValueError, match="events outside"):
        compute_arrivals(model, inside, outside)


def test_voxels_preserved_and_interface_averages_slowness():
    velocity = np.full((2, 2, 2), 2000.)
    velocity[1] = 4000.
    model = VelocityGrid(velocity, 100.)
    nodes = _nodal_velocity(model, 4)
    assert nodes.shape == (9, 9, 9)
    np.testing.assert_allclose(nodes[:4], 2000)
    np.testing.assert_allclose(nodes[4], 1 / ((1 / 2000 + 1 / 4000) / 2))
    np.testing.assert_allclose(nodes[5:], 4000)
    assert _segment_time(model, np.array([10., 50, 50]), np.array([150., 50, 50])) == pytest.approx(90 / 2000 + 50 / 4000)


def test_off_grid_homogeneous_accuracy_and_subcell_positions():
    model = VelocityGrid(np.full((12, 12, 12), 2000.), 100.)
    events = PointSet(["event"], [[321.1, 432.2, 543.3]])
    # First two receivers are in the same original AND numerical cells.
    stations = PointSet(["near", "nearer", "far", "corner", "same"],
                        [[325.1, 432.2, 543.3], [323.1, 432.2, 543.3],
                         [920, 830, 940], [0, 0, 0], [321.1, 432.2, 543.3]])
    exact = np.linalg.norm(stations.coordinates_m - events.coordinates_m, axis=1) / 2000
    times = compute_travel_times(model, stations, events, ForwardConfig(refinement=4))[0]
    np.testing.assert_allclose(times, exact, atol=0.003, rtol=0)
    np.testing.assert_allclose(times[:2], exact[:2], atol=1e-6, rtol=0)
    assert times[-1] == 0
    assert times[0] != times[1]
    moved_events = PointSet(["moved"], events.coordinates_m + [1., 0., 0.])
    moved = compute_travel_times(model, stations, moved_events, ForwardConfig(refinement=4))[0]
    assert moved[0] != times[0]


def test_relative_times_ids_order_and_translation_invariance():
    model = VelocityGrid(np.full((6, 6, 6), 2000.), 100.)
    stations = PointSet(["001", "north", "south"], [[0, 0, 0], [600, 600, 600], [311.2, 120.4, 90.3]])
    events = PointSet(["eq-b", "eq-a"], [[311.2, 120.4, 90.3], [490.1, 420.2, 470.3]])
    arrivals = compute_arrivals(model, stations, events)
    absolute = compute_travel_times(model, stations, events)
    relative = np.array([a.arrival_time_s for a in arrivals]).reshape(2, 3)
    np.testing.assert_allclose(relative, absolute - absolute.min(axis=1, keepdims=True))
    np.testing.assert_array_equal(relative.min(axis=1), 0)
    assert [(a.event_id, a.station_id) for a in arrivals] == [
        (e, s) for e in events.ids for s in stations.ids]
    offset = np.array([-1000., 700., 200.])
    shifted = compute_travel_times(
        VelocityGrid(model.velocity, 100., tuple(offset)),
        PointSet(stations.ids, stations.coordinates_m + offset),
        PointSet(events.ids, events.coordinates_m + offset),
    )
    np.testing.assert_allclose(shifted, absolute, atol=1e-12)
    reversed_times = compute_travel_times(model, stations, PointSet(events.ids[::-1], events.coordinates_m[::-1]))
    np.testing.assert_array_equal(reversed_times[::-1], absolute)


@pytest.mark.parametrize("source", [[0, 0, 0], [400, 400, 400], [0, 211.3, 117.4]])
def test_sources_and_receivers_on_closed_boundaries(source):
    model = VelocityGrid(np.full((4, 4, 4), 2000.), 100.)
    events = PointSet(["source"], [source])
    stations = PointSet(["same", "face"], [source, [400, 0, 155.5]])
    times = compute_travel_times(model, stations, events, ForwardConfig(refinement=4))[0]
    exact = np.linalg.norm(stations.coordinates_m - events.coordinates_m, axis=1) / 2000
    np.testing.assert_allclose(times, exact, atol=0.003, rtol=0)
    assert times[0] == 0


def test_two_layers_against_snell_refraction():
    velocity = np.full((6, 8, 8), 2000.)
    velocity[3:] = 3000.
    source = np.array([130.2, 211.3, 311.8])
    receiver = np.array([520.4, 600.1, 455.5])
    transverse = np.linalg.norm(receiver[1:] - source[1:])
    time_at_crossing = lambda distance: (
        np.hypot(300 - source[0], distance) / 2000
        + np.hypot(receiver[0] - 300, transverse - distance) / 3000
    )
    exact = minimize_scalar(time_at_crossing, bounds=(0, transverse), method="bounded").fun
    result = compute_travel_times(VelocityGrid(velocity, 100.),
                                 PointSet(["s"], [receiver]), PointSet(["e"], [source]),
                                 ForwardConfig(refinement=4))[0, 0]
    assert result == pytest.approx(exact, abs=0.004)


def test_head_wave_in_fast_lower_layer():
    velocity = np.full((24, 8, 12), 2000.)
    velocity[:, :, 6:] = 4000.
    source = np.array([311.2, 400.1, 450.])
    receiver = np.array([2000.2, 400.1, 450.])
    horizontal = receiver[0] - source[0]
    head_time = horizontal / 4000 + 300 * np.sqrt(1 / 2000**2 - 1 / 4000**2)
    exact = min(horizontal / 2000, head_time)
    result = compute_travel_times(VelocityGrid(velocity, 100.),
                                 PointSet(["s"], [receiver]), PointSet(["e"], [source]),
                                 ForwardConfig(refinement=4))[0, 0]
    assert result == pytest.approx(exact, abs=0.01)
    assert result < horizontal / 2000


def test_convergence_checks_absolute_times_even_for_single_station():
    model = VelocityGrid(np.full((6, 6, 6), 2000.), 100.)
    stations = PointSet(["s"], [[599.1, 400.4, 570.2]])
    events = PointSet(["e"], [[110.2, 150.5, 190.1]])
    stats = check_convergence(model, stations, events, ForwardConfig(refinement=1))
    assert stats["coarse_refinement"] == 1
    assert stats["fine_refinement"] == 2
    assert stats["max_abs_difference_s"] == 0
    assert stats["max_absolute_time_difference_s"] > 0
