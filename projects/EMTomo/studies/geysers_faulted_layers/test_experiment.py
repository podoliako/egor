"""Geometry checks only: no FMM or inverse solve."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('faulted_experiment', Path(__file__).with_name('experiment.py'))
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_fault_and_layer_velocities():
    model = experiment.build_model()
    assert model.velocity.shape == (160, 56, 56)
    assert model.cell_size_m == 125
    # x=6.5 km is exactly the face between cells 51 and 52.
    np.testing.assert_array_equal(model.velocity[51, 0, [0, 10, 30]], [3600, 4400, 5000])
    np.testing.assert_array_equal(model.velocity[52, 0, [0, 10, 30]], [4000, 4800, 5500])
    np.testing.assert_array_equal(model.velocity[:, 0], model.velocity[:, -1])
    assert set(np.unique(model.velocity)) == {3600, 4000, 4400, 4800, 5000, 5500}


def test_discretized_boundaries_and_right_thickness():
    model = experiment.build_model()
    depths = []
    for ix in (0, -1):
        profile = model.velocity[ix, 0]
        depths.append((np.flatnonzero(np.diff(profile)) + 1) * model.cell_size_m)
    np.testing.assert_array_equal(depths[0], [750, 1875])
    np.testing.assert_array_equal(depths[1], [1125, 2875])
    assert np.all(np.diff([0, *depths[1]]) > np.diff([0, *depths[0]]))


def test_initial_parameters_are_intermediate():
    p = experiment.PARAMETERS
    for key in ('boundaries_km', 'vp_km_s'):
        assert np.all(np.asarray(p['left'][key]) < p['initial'][key])
        assert np.all(np.asarray(p['initial'][key]) < p['right'][key])
    assert experiment.CONFIG.refinement == 2


def test_reject_nondividing_grid():
    with pytest.raises(ValueError, match='divide'):
        experiment.build_model(130)
