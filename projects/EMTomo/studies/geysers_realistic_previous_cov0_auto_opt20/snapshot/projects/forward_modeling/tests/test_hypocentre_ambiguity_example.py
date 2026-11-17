"""Lightweight checks for the small matched hypocentre-ambiguity inputs."""
from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from projects.forward_modeling.experiments import load_inputs
from projects.forward_modeling.hypocentre_ambiguity_example import (
    CHECKERBOARD_ID, CONFIG, HOMOGENEOUS_ID, main,
)


def test_matched_inputs_and_provenance(tmp_path, capsys):
    for options, experiment_id, velocities in (
        ([], CHECKERBOARD_ID, (4500., 5500.)),
        (["--homogeneous"], HOMOGENEOUS_ID, (5000., 5000.)),
    ):
        source = tmp_path / "input" / experiment_id
        assert main(["--root", str(tmp_path), *options]) == 0
        assert capsys.readouterr().out.strip() == str(source)
        assert {path.name for path in source.iterdir()} == {
            "model.npz", "stations.csv", "events.csv", "generation.json"
        }
        generation = json.loads((source / "generation.json").read_text())
        expected_config = replace(CONFIG, velocities_m_s=velocities)
        assert generation == {
            "generator": "checkerboard",
            "parameters": json.loads(json.dumps(asdict(expected_config))),
        }
        model, stations, events = load_inputs(tmp_path, experiment_id)
        assert model.velocity.shape == (4, 3, 2)
        assert model.cell_size_m == 6000.
        assert set(np.unique(model.velocity)) == set(velocities)
        assert len(stations.ids) == 8
        assert len(events.ids) == 80
        np.testing.assert_array_equal(np.unique(stations.coordinates_m[:, 0]), [3000, 9000, 15000, 21000])
        np.testing.assert_array_equal(np.unique(stations.coordinates_m[:, 1]), [4500, 13500])
        np.testing.assert_array_equal(stations.coordinates_m[:, 2], np.zeros(8))
        for axis, levels in enumerate(([3000, 9000, 15000, 21000],
                                       [2250, 6750, 11250, 15750],
                                       [3000, 5000, 7000, 9000, 11000])):
            np.testing.assert_array_equal(np.unique(events.coordinates_m[:, axis]), levels)
        assert len(np.unique(events.coordinates_m, axis=0)) == 80

    checker = tmp_path / "input" / CHECKERBOARD_ID
    control = tmp_path / "input" / HOMOGENEOUS_ID
    for name in ("stations.csv", "events.csv"):
        assert (checker / name).read_bytes() == (control / name).read_bytes()
    model, _, _ = load_inputs(tmp_path, CHECKERBOARD_ID)
    assert model.velocity[0, 0, 0] == 4500.
    assert model.velocity[1, 0, 0] == 5500.
    assert model.velocity[0, 1, 0] == 5500.
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("options,experiment_id", [
    ([], CHECKERBOARD_ID),
    (["--homogeneous"], HOMOGENEOUS_ID),
    (["custom-control", "--homogeneous"], "custom-control"),
])
def test_does_not_overwrite_existing_inputs(tmp_path, capsys, options, experiment_id):
    assert main(["--root", str(tmp_path), *options]) == 0
    capsys.readouterr()
    source = tmp_path / "input" / experiment_id
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    with pytest.raises(SystemExit) as error:
        main(["--root", str(tmp_path), *options])
    assert error.value.code == 1
    assert "Experiment already exists" in capsys.readouterr().err
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before
    assert not (tmp_path / "output").exists()
