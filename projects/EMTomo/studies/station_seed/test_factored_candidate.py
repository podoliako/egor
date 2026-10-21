"""Regression tests for the isolated experimental solver, not a production switch."""

import numpy as np
import pytest
from scipy.optimize import brentq

from factored_candidate import _hamiltonian, _root, travel_time_field


def _distance(shape, h, source):
    axes = [(np.arange(n) + 0.5) * h - s for n, s in zip(shape, source)]
    return np.sqrt(axes[0][:, None, None] ** 2 + axes[1][None, :, None] ** 2 + axes[2][None, None, :] ** 2)


@pytest.mark.parametrize("source", [(830.0, 615.0, 0.0), (830.0, 615.0, 730.0),
                                    (0.0, 0.0, 0.0), (2000.0, 1500.0, 1250.0)])
def test_homogeneous_boundary_and_offgrid_sources(source):
    h, shape = 125.0, (16, 12, 10)
    velocity = np.full(shape, 4000.0)
    velocity.flags.writeable = False
    actual, info = travel_time_field(velocity, h, source, dtype=np.float64, diagnostics=True)
    np.testing.assert_allclose(actual, _distance(shape, h, source) / 4000.0, rtol=0, atol=2e-14)
    assert info["relative_hamiltonian_residual"]["rms"] < 1e-10
    np.testing.assert_array_equal(velocity, np.full(shape, 4000.0))


@pytest.mark.parametrize("ulps", [-64, -8, -1, 1, 8, 64])
def test_heterogeneous_fields_are_continuous_under_ulp_perturbations(ulps):
    h, shape, source = 125.0, (16, 12, 10), (830.0, 615.0, 0.0)
    velocity = np.full(shape, 4000.0)
    velocity[:, :, 5:] = 5000.0
    original = travel_time_field(velocity, h, source, dtype=np.float64)
    perturbed = velocity.copy()
    perturbed[5:9, 4:8, :6] += ulps * np.spacing(4000.0)
    changed = travel_time_field(perturbed, h, source, dtype=np.float64)
    assert np.max(np.abs(changed - original)) < 1e-10


def test_inner_medium_is_ignored_but_external_speeds_remain_parameters():
    h, shape, source = 125.0, (16, 12, 10), (830.0, 615.0, 0.0)
    velocity = np.full(shape, 4000.0)
    distance = _distance(shape, h, source)
    inner = velocity.copy()
    inner[(distance > 250.0) & (distance <= 375.0)] = 2000.0
    original = travel_time_field(velocity, h, source, dtype=np.float64)
    np.testing.assert_array_equal(travel_time_field(inner, h, source, dtype=np.float64), original)
    external = velocity.copy()
    external[distance > 375.0] = 5000.0
    fast = travel_time_field(external, h, source, dtype=np.float64)
    assert np.max(np.abs(fast - original)) > 1e-3
    assert np.max(fast - original) < 1e-11
    np.testing.assert_allclose(fast[distance <= 375], distance[distance <= 375] / 4000.0, atol=1e-15)


def test_local_newton_matches_independent_bracketed_root_at_branch_switches():
    rng = np.random.default_rng(42)
    for _ in range(100):
        slopes = rng.uniform(2.0, 10.0, size=6)
        offsets = rng.uniform(0.0, 0.003, size=6)
        coefficients = tuple(value for pair in zip(slopes, offsets) for value in pair)
        slowness = rng.uniform(1 / 6000.0, 1 / 2000.0)
        upper = float(np.max(offsets / slopes) + slowness / np.min(slopes))
        expected = brentq(lambda u: _hamiltonian(u, coefficients)[0] - slowness ** 2,
                          0.0, upper, xtol=1e-18)
        assert _root(upper, slowness, coefficients) == pytest.approx(expected, rel=1e-11, abs=1e-17)
    # All branches meet at the root: the right derivative must remain valid.
    tied = (3.0, 0.0, 5.0, 0.0005) * 3
    expected = brentq(lambda u: _hamiltonian(u, tied)[0] - 1e-6, 0, 0.001, xtol=1e-18)
    assert _root(0.001, 0.001, tied) == pytest.approx(expected, rel=1e-11)


def test_failure_to_converge_is_not_silently_accepted():
    velocity = np.full((18, 14, 12), 4000.0)
    velocity[:, :, 5:] = 6000.0
    with pytest.raises(RuntimeError, match="did not converge"):
        travel_time_field(velocity, 125.0, (830.0, 615.0, 0.0), max_cycles=1)


@pytest.mark.parametrize("options", [
    {"radius_cells": np.inf}, {"radius_cells": np.nan},
    {"tolerance": np.nan}, {"tolerance": np.inf}, {"tolerance": 0},
    {"max_cycles": 1.5}, {"max_cycles": True}, {"max_cycles": 0},
    {"dtype": np.int64},
])
def test_invalid_numerical_parameters_are_rejected(options):
    with pytest.raises(ValueError):
        travel_time_field(np.full((8, 6, 6), 4000.0), 125.0, (500.0, 375.0, 0.0), **options)
