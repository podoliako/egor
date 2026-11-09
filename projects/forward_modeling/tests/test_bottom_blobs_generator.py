import csv
import json

import numpy as np
import pytest

from projects.forward_modeling.bottom_blobs_generator import generate_geometry, main


def test_geometry_counts_bounds_and_reproducibility():
    stations, events = generate_geometry()
    xyz = events.coordinates_m / 1000
    assert xyz.shape == (600, 3)
    assert len(set(events.ids)) == 600
    assert np.all(xyz > 0) and np.all(xyz < [24, 8, 8])
    assert np.all((xyz[:500, 2] >= 7.2) & (xyz[:500, 2] < 7.8))
    tiles = np.floor((xyz[:500, :2] - [0.5, 0.5]) / [23, 7] * [50, 10]).astype(int)
    assert len(np.unique(tiles, axis=0)) == 500
    np.testing.assert_allclose(xyz[500:550].mean(axis=0), [6, 4, 4], atol=0.2)
    np.testing.assert_allclose(xyz[550:].mean(axis=0), [18, 4, 4], atol=0.2)
    station_xyz = stations.coordinates_m / 1000
    assert station_xyz.shape == (22, 3)
    assert len(np.unique(station_xyz, axis=0)) == 22
    np.testing.assert_array_equal(station_xyz[:, 2], 0)
    assert np.count_nonzero(station_xyz[:, 0] < 2) == 10
    assert np.count_nonzero(station_xyz[:, 0] > 22) == 10
    np.testing.assert_array_equal(station_xyz[-2:], [[12, 2, 0], [12, 6, 0]])
    np.testing.assert_array_equal(events.coordinates_m, generate_geometry()[1].coordinates_m)
    assert not np.array_equal(events.coordinates_m, generate_geometry(43)[1].coordinates_m)


def test_preview_preserves_cropped_velocity_and_protects_outputs(tmp_path):
    velocity = np.where(np.indices((6, 3, 3)).sum(axis=0) % 2, 5250.0, 4750.0)
    source = tmp_path / "source.npz"
    np.savez(source, velocity=velocity, cell_size_m=4000.0, origin_m=np.zeros(3))
    output = tmp_path / "preview"
    args = ["--source-model", str(source), "--output-dir", str(output)]
    main(args)
    with np.load(output / "model.npz", allow_pickle=False) as model:
        np.testing.assert_array_equal(model["velocity"], velocity[:, :2, :2])
        assert float(model["cell_size_m"]) == 4000
    for filename, count in (("stations.csv", 22), ("events.csv", 600)):
        with (output / filename).open(newline="") as stream:
            assert len(list(csv.DictReader(stream))) == count
    for view in ("xy", "xz", "yz"):
        assert (output / f"geometry_{view}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert json.loads((output / "generation.json").read_text())["blocks_xyz"] == [6, 2, 2]
    with pytest.raises(SystemExit):
        main(args)
