"""Cell-centred velocity model, refinement and reference averaging."""

import numpy as np
import pytest

from interpolation import prolongate_cell_centered_trilinear
from velocity_model import VelocityModel, block_average_slowness


def test_model_is_validated_and_read_only():
    model = VelocityModel(np.full((2, 3, 4), 5000.0), 100.0)
    assert model.shape == (2, 3, 4)
    assert model.velocity.dtype == np.float64
    with pytest.raises(ValueError):
        model.velocity[0, 0, 0] = 1.0
    for velocity, cell_size in ((np.zeros((2, 2, 2)), 1.0), (np.ones((2, 2)), 1.0),
                                (np.ones((2, 2, 2)), 0.0), (np.full((2, 2, 2), np.nan), 1.0)):
        with pytest.raises(ValueError):
            VelocityModel(velocity, cell_size)


def test_nearest_refinement_copies_parent_cells():
    velocity = np.arange(1.0, 9.0).reshape(2, 2, 2) * 1000
    fine = VelocityModel(velocity, 300.0).refined(3)
    assert fine.shape == (6, 6, 6)
    assert fine.cell_size == 100.0
    np.testing.assert_array_equal(fine.velocity[::3, ::3, ::3], velocity)
    np.testing.assert_array_equal(fine.velocity[2, 5, 4], velocity[0, 1, 1])
    assert VelocityModel(velocity, 300.0).refined(1).shape == (2, 2, 2)


def test_trilinear_refinement_interpolates_slowness():
    velocity = np.array([100.0, 200.0, 400.0]).reshape(3, 1, 1)
    fine = VelocityModel(velocity, 100.0).refined(2, "trilinear")
    np.testing.assert_allclose(fine.velocity, 1.0 / prolongate_cell_centered_trilinear(1.0 / velocity, 2))


def test_block_average_is_slowness_mean_over_overlaps():
    source = np.array([1000.0, 2000.0, 4000.0, 4000.0]).reshape(4, 1, 1)
    nested = block_average_slowness(np.broadcast_to(source, (4, 2, 2)), 1.0, (2, 1, 1), 2.0)
    np.testing.assert_allclose(nested.ravel(), [1 / ((1 / 1000 + 1 / 2000) / 2), 4000.0])
    # Target cells of 4/3 straddle source cells.
    straddling = block_average_slowness(np.broadcast_to(source, (4, 4, 4)), 1.0, (3, 3, 3), 4 / 3)
    expected_first = 1 / ((1 / 1000 + (1 / 3) / 2000) * 3 / 4)
    assert straddling[0, 0, 0] == pytest.approx(expected_first)
    rng = np.random.default_rng(0)
    velocity = rng.uniform(4000, 6000, size=(6, 4, 3))
    uniform = block_average_slowness(velocity, 2.0, (6, 4, 3), 2.0)
    np.testing.assert_allclose(uniform, velocity)
    with pytest.raises(ValueError, match="extent"):
        block_average_slowness(velocity, 2.0, (5, 4, 3), 2.0)
