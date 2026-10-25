"""Compact construction must retain the dense kernels' geometry and arithmetic."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import raytracing as rays
from instruments.instruments_ops import restriction_tables


def _args(shape=(12, 10, 8), stations=None, epic=None, interpolation="identity",
          radius=3.0, dtype=np.float32, step=0.13, tol=0.01, max_steps=1000):
    if stations is None:
        stations = [[0.0, 0.0, 0.0], [2.2, 3.7, -0.5],
                    [11.5, 9.5, 7.5], [-0.5, -0.5, -0.5], [6.3, 4.2, 3.1]]
    stations = np.asarray(stations, dtype=np.float64).reshape(-1, 3)
    epic = np.asarray([8.5, 6.5, 5.5] if epic is None else epic, dtype=np.float64)
    # Linear gradients interpolate exactly and trace straight rays, including
    # surface stations a half-cell outside the clipped centre domain.
    coords = np.indices(shape, dtype=np.float64)
    gradients = tuple(np.ascontiguousarray(
        coords[axis][None] - stations[:, axis, None, None, None], dtype=dtype,
    ) for axis in range(3))
    sub = 1 if interpolation == "identity" else 2
    out_shape = tuple(n // sub for n in shape)
    return (*gradients, stations, epic, 137.5, step, tol, max_steps,
            np.zeros(3), np.asarray(shape, dtype=float) - 1.0, radius,
            *restriction_tables(out_shape, sub, interpolation), out_shape)


def _compare(args):
    dense, expected_reached = rays.compute_G_all_stations_serial(*args)
    compact, reached = rays.compute_G_all_stations_compact(*args)
    assert isinstance(compact, csr_matrix)
    assert compact.shape == (len(reached), int(np.prod(args[-1])))
    assert compact.dtype == np.float64
    assert reached.dtype == np.bool_
    assert reached.shape == (args[0].shape[0],)
    assert compact.has_canonical_format
    assert np.all(compact.data != 0.0)
    np.testing.assert_array_equal(reached, expected_reached)
    np.testing.assert_allclose(compact.toarray(), dense.reshape(compact.shape),
                               rtol=2e-13, atol=2e-12)
    assert np.all(np.diff(compact.indptr)[~reached] == 0)
    return compact, reached


@pytest.mark.parametrize("radius", [0.0, 3.0])
@pytest.mark.parametrize("interpolation", ["identity", "nearest", "trilinear"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compact_matches_dense_surface_corners_and_restriction(radius, interpolation, dtype):
    compact, reached = _compare(_args(radius=radius, interpolation=interpolation, dtype=dtype))
    assert reached.all()
    assert np.all(np.diff(compact.indptr) > 0)


@pytest.mark.parametrize("radius", [0.0, 3.0])
@pytest.mark.parametrize("interpolation", ["identity", "nearest", "trilinear"])
def test_inside_source_sphere_and_zero_length_success(radius, interpolation):
    epic = np.array([4.2, 3.3, 2.1])
    stations = [epic, epic + [0.2, 0.1, 0.1], epic + [1.7, 0.0, 0.0],
                epic + [3.0, 0.0, 0.0]]  # exactly on sphere as well as inside
    compact, reached = _compare(_args(epic=epic, stations=stations, radius=radius,
                                     interpolation=interpolation))
    assert reached.all()
    assert compact.indptr[1] == 0
    np.testing.assert_allclose(np.asarray(compact.sum(axis=1)).ravel(),
                               np.linalg.norm(np.asarray(stations) - epic, axis=1) * 137.5,
                               rtol=2e-13, atol=2e-12)


@pytest.mark.parametrize("radius", [0.0, 3.0])
@pytest.mark.parametrize("interpolation", ["identity", "nearest", "trilinear"])
def test_success_failure_success_rows_do_not_leak_workspace(radius, interpolation):
    stations = [[0., 2., 2.], [0., 2., 2.], [0., 2., 2.],
                [5., 2., 2.], [4., 2., 2.], [0., 2., 2.]]
    args = list(_args(shape=(6, 4, 4), stations=stations, epic=[5., 2., 2.],
                      radius=radius, interpolation=interpolation))
    for gradient in args[:3]:
        gradient[1] = 0.0  # zero gradient before reaching
        gradient[2] *= -1.0  # outward gradient stalls at the clipped boundary
    compact, reached = _compare(tuple(args))
    np.testing.assert_array_equal(reached, [True, False, False, True, True, True])
    np.testing.assert_array_equal(compact.getrow(0).toarray(), compact.getrow(5).toarray())
    np.testing.assert_array_equal(np.diff(compact.indptr)[1:4], [0, 0, 0])


@pytest.mark.parametrize("max_steps, step, tol, expected", [
    (0, 0.1, 0.01, False), (1, 0.1, 0.01, False),
    (20, 0.0, 0.01, False), (1, 0.1, 10.0, True),
    (1, 0.1, 0.6, True), (1, 0.1, 0.59, False),
])
def test_max_steps_stalled_ray_and_tolerance(max_steps, step, tol, expected):
    _, reached = _compare(_args(shape=(6, 2, 2), stations=[[0., 0., 0.]],
                               epic=[0.6, 0., 0.], max_steps=max_steps, step=step, tol=tol))
    assert reached[0] == expected


@pytest.mark.parametrize("radius", [0.0, 3.0])
def test_singleton_axes_and_clipped_hypocentre(radius):
    compact, reached = _compare(_args(shape=(8, 1, 1), stations=[[0., 0., -0.5], [7.5, 0., 0.]],
                                     epic=[10., -2., 3.], radius=radius))
    assert reached.all()
    assert compact.shape == (2, 8)


@pytest.mark.parametrize("radius", [0.0, 3.0])
@pytest.mark.parametrize("interpolation", ["nearest", "trilinear"])
def test_restriction_with_singleton_output_axes(radius, interpolation):
    compact, reached = _compare(_args(shape=(8, 2, 2), stations=[[0., 0., -0.5], [7.5, 1., 1.]],
                                     epic=[5.5, 0.5, 0.5], radius=radius,
                                     interpolation=interpolation))
    assert reached.all()
    assert compact.shape == (2, 4)


def test_dense_parallel_reference_matches_compact():
    args = _args(interpolation="trilinear")
    compact, reached = rays.compute_G_all_stations_compact(*args)
    dense, expected_reached = rays.compute_G_all_stations(*args)
    np.testing.assert_array_equal(reached, expected_reached)
    np.testing.assert_allclose(compact.toarray(), dense.reshape(compact.shape),
                               rtol=2e-13, atol=2e-12)


def test_empty_station_stack():
    compact, reached = _compare(_args(stations=[]))
    assert compact.shape == (0, 960)
    assert compact.nnz == reached.size == 0
    np.testing.assert_array_equal(compact.indptr, [0])


@pytest.mark.parametrize("interpolation", ["identity", "nearest", "trilinear"])
@pytest.mark.parametrize("radius", [0.0, 3.0])
def test_random_curved_gradients_and_repeated_calls(interpolation, radius):
    rng = np.random.default_rng(921)
    shape = (12, 10, 8)
    stations = rng.uniform(0, np.asarray(shape) - 1, size=(12, 3))
    args = list(_args(stations=stations, radius=radius, interpolation=interpolation, dtype=np.float64))
    # Smooth anisotropy bends rays and exercises unequal DDA crossing times.
    for axis, gradient in enumerate(args[:3]):
        gradient *= (0.7, 1.1, 1.4)[axis]
        gradient += rng.normal(0.0, 0.03, size=gradient.shape)
    first, reached = _compare(tuple(args))
    assert reached.all()
    saved = first.copy()
    args[4] = np.array([1.5, 2.5, 3.5])
    _compare(tuple(args))
    np.testing.assert_array_equal(first.indptr, saved.indptr)
    np.testing.assert_array_equal(first.indices, saved.indices)
    np.testing.assert_array_equal(first.data, saved.data)


@pytest.mark.parametrize("radius", [0.0, 3.0])
@pytest.mark.parametrize("interpolation", ["identity", "nearest", "trilinear"])
def test_shared_deposition_specialization_at_dda_boundaries(radius, interpolation):
    shape = (8, 8, 8)
    sub = 1 if interpolation == "identity" else 2
    out_shape = tuple(n // sub for n in shape)
    tables = restriction_tables(out_shape, sub, interpolation)
    paths = [
        # Tied crossings, negative travel from exact boundaries, repeat visits.
        [[6.5, 6.5, 6.5], [0.5, 0.5, 0.5], [6.5, 6.5, 6.5], [0., 0., 0.]],
        # Grid clipping, parallel exterior segment, and zero-length segment.
        [[-2., 1., 1.], [-2., 5., 1.], [9., 5., 1.], [9., 5., 1.], [7., 7., 7.]],
        # No point enters the source sphere (deposit's fallback branch).
        [[6., 7., 7.], [7., 7., 7.]],
    ]
    for points in paths:
        path = np.asarray(points, dtype=np.float64)
        dense = np.zeros(out_shape)
        row = rays._TouchedRow(out_shape)
        station = np.zeros(3)
        rays._deposit_ray_nb(path, dense, 137.5, *tables, station, radius)
        rays._deposit_ray_nb(path, row, 137.5, *tables, station, radius)
        np.testing.assert_allclose(row.values, dense, rtol=2e-13, atol=2e-12)
        touched = row.touched[:row.count]
        assert len(np.unique(touched)) == row.count
        np.testing.assert_array_equal(np.sort(touched), np.flatnonzero(row.seen))
        assert np.all(row.values.ravel()[~row.seen] == 0.0)


def test_compact_traces_once_and_reuses_one_cleared_row(monkeypatch):
    args = _args()
    dense, expected_reached = rays.compute_G_all_stations_serial(*args)
    trace = rays._trace_ray_nb
    deposit = rays._deposit_ray_nb
    traced = []
    rows = []

    def trace_once(*args):
        traced.append(args[3].copy())
        return trace(*args)

    def deposit_into_row(path, row, *args):
        assert row.count == 0
        assert not row.seen.any()
        assert not row.values.any()
        rows.append(row)
        return deposit(path, row, *args)

    def no_dense_builder(*args):
        pytest.fail("compact construction must not call a dense station builder")

    monkeypatch.setattr(rays, "_trace_ray_nb", trace_once)
    monkeypatch.setattr(rays, "_deposit_ray_nb", deposit_into_row)
    monkeypatch.setattr(rays, "compute_G_all_stations_serial", no_dense_builder)
    monkeypatch.setattr(rays, "compute_G_all_stations", no_dense_builder)
    # Exercise the kernel's Python control flow to observe calls; deposition
    # still uses the actual compiled specialization, not a substitute algorithm.
    data, indices, indptr, reached = rays._compute_G_compact_nb.py_func(*args)
    assert len(traced) == len(args[3])
    np.testing.assert_array_equal(traced, args[3])
    assert len(rows) == reached.sum()
    assert all(row is rows[0] for row in rows)
    assert rows[0].values.shape == args[-1]
    np.testing.assert_array_equal(reached, expected_reached)
    actual = csr_matrix((data, indices, indptr), shape=(len(reached), int(np.prod(args[-1]))))
    np.testing.assert_allclose(actual.toarray(), dense.reshape(actual.shape), rtol=2e-13, atol=2e-12)


@pytest.mark.parametrize("interpolation", ["identity", "nearest", "trilinear"])
def test_heterogeneous_travel_time_gradients(interpolation):
    from eikonal import travel_time_field

    shape = (12, 10, 8)
    stations = np.array([[1.3, 2.1, -0.5], [11.5, 8.7, 3.3]])
    velocity = np.full(shape, 3500.0)
    velocity[:, :, 3:] = 5100.0
    args = list(_args(stations=stations, interpolation=interpolation))
    fields = [travel_time_field(velocity, args[5], (station + 0.5) * args[5])
              for station in stations]
    gradients = [np.gradient(field) for field in fields]
    args[:3] = [np.ascontiguousarray([gradient[axis] for gradient in gradients], dtype=np.float32)
                for axis in range(3)]
    _, reached = _compare(tuple(args))
    assert reached.all()
