"""The homogeneous source sphere must be the same medium in FMM and ray deposition."""

import numpy as np
import pytest
import skfmm

from eikonal import SOURCE_RADIUS_CELLS, travel_time_field


def _geometry(shape, h, source):
    axes = [(np.arange(n) + 0.5) * h - s for n, s in zip(shape, source)]
    return np.sqrt(axes[0][:, None, None] ** 2
                   + axes[1][None, :, None] ** 2
                   + axes[2][None, None, :] ** 2)


@pytest.mark.parametrize("source", [
    (1900.0, 1450.0, 0.0),
    (1900.0, 1450.0, 1300.0),
    (0.0, 0.0, 0.0),
    (4000.0, 3000.0, 3000.0),
])
@pytest.mark.parametrize("shell_velocity", [2000.0, 6000.0])
def test_non_source_velocities_inside_seed_sphere_do_not_leak_out(source, shell_velocity):
    h, shape = 125.0, (32, 24, 24)
    distance = _geometry(shape, h, source)
    radius = SOURCE_RADIUS_CELLS * h
    velocity = np.full(shape, 4000.0)
    perturbed = velocity.copy()
    perturbed[(distance > 2 * h) & (distance <= radius)] = shell_velocity

    original = travel_time_field(velocity, h, source)
    changed = travel_time_field(perturbed, h, source)
    np.testing.assert_array_equal(changed, original)


def test_seed_homogenization_does_not_change_velocity_input_or_outside_medium():
    h, shape, source = 125.0, (32, 24, 24), (1900.0, 1450.0, 0.0)
    distance = _geometry(shape, h, source)
    radius = SOURCE_RADIUS_CELLS * h
    velocity = np.full(shape, 4000.0)
    velocity[(distance > 2 * h) & (distance <= radius)] = 2000.0
    velocity[(distance > radius) & (distance < 6 * h)] = 4500.0
    before = velocity.copy()
    velocity.flags.writeable = False

    actual = travel_time_field(velocity, h, source)
    np.testing.assert_array_equal(velocity, before)
    explicit_medium = before.copy()
    explicit_medium[distance <= radius] = 4000.0
    expected = np.asarray(skfmm.travel_time(distance - radius, explicit_medium, dx=h, order=2))
    expected += radius / 4000.0
    expected[distance <= radius] = distance[distance <= radius] / 4000.0
    np.testing.assert_array_equal(actual, expected.astype(np.float32))
    homogeneous = travel_time_field(np.full(shape, 4000.0), h, source)
    assert np.max(np.abs(actual[distance > 6 * h] - homogeneous[distance > 6 * h])) > 1e-3


def test_homogeneous_seed_keeps_previous_homogeneous_solution():
    h, shape, source = 125.0, (32, 24, 24), (1900.0, 1450.0, 0.0)
    velocity = np.full(shape, 4000.0)
    distance = _geometry(shape, h, source)
    radius = SOURCE_RADIUS_CELLS * h
    old = np.asarray(skfmm.travel_time(distance - radius, velocity, dx=h, order=2))
    old += radius / 4000.0
    old[distance <= radius] = distance[distance <= radius] / 4000.0
    np.testing.assert_array_equal(travel_time_field(velocity, h, source), old.astype(np.float32))


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="scikit-fmm order=2 switches stencils at roundoff-level time ties; see studies/station_seed/report.md",
)
def test_second_order_field_is_continuous_under_one_ulp_velocity_change():
    h, shape, source = 125.0, (48, 32, 32), (1000.0, 875.0, 0.0)
    velocity = np.full(shape, 4000.0)
    perturbed = velocity.copy()
    perturbed[4:8, 4:8, :4] = np.nextafter(4000.0, np.inf)
    difference = travel_time_field(perturbed, h, source) - travel_time_field(velocity, h, source)
    assert np.max(np.abs(difference)) < 1e-9


def test_seed_sphere_covering_entire_domain_uses_source_velocity():
    h, source = 125.0, (0.0, 0.0, 0.0)
    velocity = np.full((2, 2, 2), 6000.0)
    velocity[0, 0, 0] = 4000.0
    expected = _geometry(velocity.shape, h, source) / 4000.0
    np.testing.assert_array_equal(travel_time_field(velocity, h, source), expected.astype(np.float32))
