"""Tests for the published-model representation reference; no seismic solvers."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_geysers_real_published_comparison import volume_slowness_oracle


class Field:
    def __init__(self, speed, xmax=2.):
        self.speed = speed
        self.xmax = xmax

    def sample(self, points):
        velocity = np.broadcast_to(self.speed(points), points.shape[:-1]).astype(float).copy()
        inside = ((points >= 0).all(axis=-1)
                  & (points <= [self.xmax, 1., 1.]).all(axis=-1))
        velocity[~inside] = np.nan
        return np.stack((velocity, np.ones_like(velocity)), axis=-1)


def test_constant_field_and_no_partial_cell_fill():
    oracle = volume_slowness_oracle(Field(lambda p: 5., xmax=1.75), [2., 1., 1.], samples_per_axis=4)
    assert oracle.shape == (2, 1, 1)
    assert oracle[0, 0, 0] == 5000.
    assert np.isnan(oracle[1, 0, 0])


def test_linear_vp_projects_slowness_not_arithmetic_velocity():
    oracle = volume_slowness_oracle(Field(lambda p: 4. + p[..., 0]), [1., 1., 1.], samples_per_axis=32)
    analytic = 1000. / np.log(5. / 4.)
    np.testing.assert_allclose(oracle, analytic, rtol=0, atol=.03)
    assert abs(oracle.item() - 4500.) > 10


def test_nonpositive_covered_values_rejected():
    with pytest.raises(ValueError, match='Invalid samples'):
        volume_slowness_oracle(Field(lambda p: 0.), [1., 1., 1.], samples_per_axis=2)
