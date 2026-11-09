import numpy as np
import pytest

from projects.forward_modeling.one_sided_events import generate_events
from projects.forward_modeling.one_sided_pilot import select_events, spread_indices


def test_spread_selection_preserves_original_events_and_groups():
    catalog = generate_events()
    events, groups = select_events(catalog)
    assert len(events.ids) == len(set(events.ids)) == 10
    assert len(groups["far"]) == 7
    assert len(groups["near"]) == 3
    for identifier, xyz in zip(events.ids, events.coordinates_m):
        np.testing.assert_array_equal(xyz, catalog.coordinates_m[catalog.ids.index(identifier)])
    assert np.all(events.coordinates_m[:7, 0] >= 10000)
    assert np.all(events.coordinates_m[7:, 0] <= 9000)
    # Cover different positions and depths rather than taking consecutive rows.
    assert np.ptp(events.coordinates_m[:7], axis=0).min() > 5000
    repeated, repeated_groups = select_events(catalog)
    assert events.ids == repeated.ids
    assert groups == repeated_groups


def test_spread_indices_do_not_repeat_even_for_coincident_points():
    assert spread_indices(np.zeros((3, 3)), 3) == [0, 1, 2]
    with pytest.raises(ValueError):
        spread_indices(np.zeros((3, 3)), 4)
