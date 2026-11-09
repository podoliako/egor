import csv
import json

import numpy as np
import pytest

from projects.forward_modeling.one_sided_stations import generate_stations, main


def test_geometry_and_reproducibility():
    stations = generate_stations()
    xyz = stations.coordinates_m
    assert xyz.shape == (18, 3)
    assert len(set(stations.ids)) == 18
    assert np.all((xyz[:, 0] > 1000) & (xyz[:, 0] < 5000))
    assert np.all((xyz[:, 1] > 1000) & (xyz[:, 1] < 11000))
    np.testing.assert_array_equal(xyz[:, 2], 0)
    np.testing.assert_array_equal(xyz, generate_stations().coordinates_m)
    assert not np.array_equal(xyz, generate_stations(seed=43).coordinates_m)
    tiles = np.floor((xyz[:, :2] / 1000 - [1, 1]) / [4 / 3, 10 / 6]).astype(int)
    assert len(np.unique(tiles, axis=0)) == 18


def test_cli_outputs_and_overwrite_protection(tmp_path):
    main(["--output-dir", str(tmp_path)])
    with (tmp_path / "stations.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 18
    assert set(rows[0]) == {"station_id", "x_m", "y_m", "z_m"}
    np.testing.assert_allclose(
        [[float(row[key]) for key in ("x_m", "y_m", "z_m")] for row in rows],
        generate_stations().coordinates_m,
    )
    assert (tmp_path / "stations_top_view.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert json.loads((tmp_path / "generation.json").read_text())["seed"] == 42
    before = (tmp_path / "stations.csv").read_bytes()
    with pytest.raises(SystemExit):
        main(["--output-dir", str(tmp_path), "--seed", "43"])
    assert (tmp_path / "stations.csv").read_bytes() == before
