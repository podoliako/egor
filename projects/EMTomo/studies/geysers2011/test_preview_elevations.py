"""Coordinate checks for the read-only raised-top preview."""
import numpy as np

from projects.EMTomo.studies.geysers2011.preview_elevations import relative_depths


def test_highest_station_is_on_top_and_others_inside():
    stations, events = relative_depths([327., 1166., 803.], [0., 1578., 6571.], 1166.)
    np.testing.assert_array_equal(stations, [839., 0., 363.])
    np.testing.assert_array_equal(events, [1166., 2744., 7737.])


def test_datum_change_preserves_station_event_vertical_separations():
    elevations = np.asarray([327., 1166., 803.])
    depths = np.asarray([0., 1578., 6571.])
    s0, e0 = relative_depths(elevations, depths, 0.)
    s1, e1 = relative_depths(elevations, depths, 1166.)
    np.testing.assert_array_equal(e0[:, None] - s0, e1[:, None] - s1)
    np.testing.assert_array_equal(elevations, [327., 1166., 803.])
    np.testing.assert_array_equal(depths, [0., 1578., 6571.])


def test_events_are_not_clipped_to_model_bottom():
    _, events = relative_depths([1166.], [5834., 5835., 6571.], 1166.)
    np.testing.assert_array_equal(events, [7000., 7001., 7737.])
    np.testing.assert_array_equal(events > 7000., [False, True, True])
