import math

import numpy as np
import pytest

from projects.forward_modeling.geysers_checkerboard_generator import catalog_event_points


GEOMETRY = {
    "reference_latitude": 38.80275,
    "reference_longitude": -122.77070,
    "north_km_per_latitude_degree": 110.574,
    "east_km_per_longitude_degree": "111.320*cos(radians(38.80275))",
    "x_azimuth_degrees": 135.0,
    "geographic_reference_in_new_xy_m": [13000, 3500],
    "old_to_new_translation_m": [3000, 0, 0],
    "top_elevation_m": 1166,
    "z_direction": "positive_down",
}


def row(identifier, latitude=38.80275, longitude=-122.77070, depth=2):
    return {"event_id": identifier, "latitude": latitude, "longitude": longitude,
            "depth_km": depth}


def test_projection_preserves_order_and_applies_datum_and_translation_once():
    north_latitude = 38.80275 + 1 / 110.574
    rows = [row("origin"), row("north", latitude=north_latitude), row("unused")]
    events = catalog_event_points(rows, ["north", "origin"], GEOMETRY)
    assert events.ids == ("north", "origin")
    np.testing.assert_allclose(events.coordinates_m[1], [13000, 3500, 3166], atol=1e-7)
    np.testing.assert_allclose(events.coordinates_m[0],
                               [13000 - 1000 / math.sqrt(2),
                                3500 - 1000 / math.sqrt(2), 3166], atol=1e-7)


@pytest.mark.parametrize("rows, selected", [([row("a"), row("a")], ["a"]),
                                          ([row("a")], ["missing"])])
def test_invalid_catalog_selection_is_rejected(rows, selected):
    with pytest.raises(ValueError):
        catalog_event_points(rows, selected, GEOMETRY)
