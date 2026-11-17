import csv
import json

import numpy as np
import pytest

from projects.forward_modeling.one_sided_events import GROUPS, generate_events, main
from projects.forward_modeling.one_sided_stations import generate_stations


def test_stratified_groups_and_reproducibility():
    events = generate_events()
    assert events.coordinates_m.shape == (1200, 3)
    assert len(set(events.ids)) == 1200
    np.testing.assert_array_equal(events.coordinates_m, generate_events().coordinates_m)
    assert not np.array_equal(events.coordinates_m, generate_events(43).coordinates_m)
    start = 0
    for group in GROUPS:
        shape = np.asarray(group["grid_shape"])
        count = int(np.prod(shape))
        xyz = events.coordinates_m[start:start + count] / 1000
        lower, upper = np.array(group["lower_km"]), np.array(group["upper_km"])
        assert np.all(xyz >= lower)
        assert np.all(xyz < upper)
        tiles = np.floor((xyz - lower) / (upper - lower) * shape).astype(int)
        assert len(np.unique(tiles, axis=0)) == count
        start += count
    assert start == 1200


def test_outputs_preserve_stations_and_refuse_overwrite(tmp_path):
    station_path = tmp_path / "stations.csv"
    stations = generate_stations()
    with station_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("station_id", "x_m", "y_m", "z_m"))
        writer.writerows((identifier, *xyz) for identifier, xyz in
                         zip(stations.ids, stations.coordinates_m))
    original = station_path.read_bytes()
    args = ["--stations-csv", str(station_path), "--output-dir", str(tmp_path)]
    main(args)
    assert station_path.read_bytes() == original
    with (tmp_path / "events.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1200
    assert set(rows[0]) == {"event_id", "x_m", "y_m", "z_m"}
    np.testing.assert_allclose(
        [[float(row[key]) for key in ("x_m", "y_m", "z_m")] for row in rows],
        generate_events().coordinates_m,
    )
    metadata = json.loads((tmp_path / "events_generation.json").read_text())
    assert [group["count"] for group in metadata["groups"]] == [900, 300]
    for plane in ("xy", "xz", "yz"):
        assert (tmp_path / f"geometry_{plane}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    before = (tmp_path / "events.csv").read_bytes()
    with pytest.raises(SystemExit):
        main(args)
    assert (tmp_path / "events.csv").read_bytes() == before
