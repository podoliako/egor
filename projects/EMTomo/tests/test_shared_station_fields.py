"""Shared station outputs preserve FMM results, failures, and buffer ownership."""

import gc
import multiprocessing as mp
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

import eikonal
from velocity_model import VelocityModel


pytestmark = pytest.mark.skipif("fork" not in mp.get_all_start_methods(), reason="station workers use fork")


def _model(shape=(12, 10, 8)):
    velocity = np.full(shape, 3500.0)
    velocity[:, :, shape[2] // 2:] = 5100.0
    return VelocityModel(velocity, 100.0)


def _stations(model):
    extent = np.asarray(model.shape) * model.cell_size
    return np.array([[0., 0., 0.], extent, extent * [0.31, 0.62, 0.0],
                     extent * [0.83, 0.21, 0.57], extent * [0.31, 0.62, 0.0]])


@pytest.mark.parametrize("shape", [(2, 2, 2), (12, 10, 8)])
@pytest.mark.parametrize("count, workers", [(2, 2), (2, 8), (5, 3)])
def test_parallel_fields_exactly_match_serial_in_station_order(shape, count, workers):
    model = _model(shape)
    stations = _stations(model)[:count]
    original_velocity = model.velocity.copy()
    original_stations = stations.copy()
    serial = eikonal.station_travel_time_fields(model, stations)
    parallel = eikonal.station_travel_time_fields(model, stations, n_workers=workers)
    assert parallel.shape == (count, *shape)
    assert parallel.dtype == np.float32
    assert parallel.flags.c_contiguous
    assert parallel.flags.writeable
    np.testing.assert_array_equal(parallel, serial)
    np.testing.assert_array_equal(model.velocity, original_velocity)
    np.testing.assert_array_equal(stations, original_stations)
    assert not model.velocity.flags.writeable
    if count == 5:
        np.testing.assert_array_equal(parallel[2], parallel[4])


@pytest.mark.parametrize("count, workers", [(1, 1), (1, 8), (2, 1), (2, 0), (2, -1)])
def test_single_station_or_serial_execution_does_not_allocate_shared_memory(monkeypatch, count, workers):
    model = _model()
    stations = _stations(model)[:count]
    expected = np.stack([eikonal.travel_time_field(model.velocity, model.cell_size, s) for s in stations])

    def forbidden_context(*args):
        pytest.fail("serial execution must not create a multiprocessing context")

    monkeypatch.setattr(eikonal, "mp", SimpleNamespace(get_context=forbidden_context))
    result = eikonal.station_travel_time_fields(model, stations.tolist(), n_workers=workers)
    assert result.flags.owndata
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("workers", [1, 8])
def test_empty_station_list_still_raises(workers):
    with pytest.raises(ValueError, match="At least one station"):
        eikonal.station_travel_time_fields(_model(), [], n_workers=workers)


@pytest.mark.parametrize("bad_index", [0, 1, 2])
def test_invalid_worker_source_raises_instead_of_returning_partial_output(bad_index):
    model = _model()
    stations = _stations(model)[:3]
    stations[bad_index] = [-1., 0., 0.]
    with pytest.raises(ValueError, match="outside the model domain"):
        eikonal.station_travel_time_fields(model, stations, n_workers=2)
    # A failed invocation must not poison a subsequent pool or its output.
    valid = _stations(model)[:2]
    np.testing.assert_array_equal(
        eikonal.station_travel_time_fields(model, valid, n_workers=2),
        eikonal.station_travel_time_fields(model, valid),
    )


def test_fmm_exception_propagates_from_worker(monkeypatch):
    def failing_fmm(*args, **kwargs):
        raise RuntimeError("synthetic fast-marching failure")

    monkeypatch.setattr(eikonal.skfmm, "travel_time", failing_fmm)
    model = _model()
    with pytest.raises(RuntimeError, match="synthetic fast-marching failure"):
        eikonal.station_travel_time_fields(model, _stations(model), n_workers=2)


def test_worker_writes_only_assigned_slice_and_returns_small_index(monkeypatch):
    model = _model()
    shape = (3, *model.shape)
    buffer = mp.get_context("fork").RawArray("f", int(np.prod(shape)))
    fields = np.frombuffer(buffer, dtype=np.float32).reshape(shape)
    fields.fill(-123.0)
    monkeypatch.setattr(eikonal, "_WORKER", {})
    eikonal._init_shared_worker(model.velocity, model.cell_size, buffer, shape)
    station = _stations(model)[2]
    expected = eikonal.travel_time_field(model.velocity, model.cell_size, station)
    status = eikonal._solve_into_shared_worker((1, station))
    assert type(status) is int
    assert status == 1
    np.testing.assert_array_equal(fields[1], expected)
    assert np.all(fields[[0, 2]] == -123.0)
    # Retain compatibility of the original array-returning private solver.
    np.testing.assert_array_equal(eikonal._solve_in_worker(station), expected)
    with pytest.raises(ValueError, match="outside the model domain"):
        eikonal._solve_into_shared_worker((2, [-1., 0., 0.]))
    assert np.all(fields[2] == -123.0)


def test_returned_array_and_derived_view_keep_raw_buffer_alive(monkeypatch):
    context = mp.get_context("fork")
    buffer_refs = []

    def tracked_raw_array(*args):
        buffer = context.RawArray(*args)
        buffer_refs.append(weakref.ref(buffer))
        return buffer

    proxy = SimpleNamespace(RawArray=tracked_raw_array, Pool=context.Pool)
    monkeypatch.setattr(eikonal, "mp", SimpleNamespace(get_context=lambda method: proxy))
    model = _model()
    stations = _stations(model)
    expected = eikonal.station_travel_time_fields(model, stations)
    fields = eikonal.station_travel_time_fields(model, stations, n_workers=2)
    gc.collect()
    assert len(buffer_refs) == 1
    assert buffer_refs[0]() is not None
    assert np.shares_memory(fields, np.frombuffer(buffer_refs[0](), dtype=np.float32))
    np.testing.assert_array_equal(fields, expected)

    view = fields[1:, ::2, :, :]
    del fields
    gc.collect()
    assert buffer_refs[0]() is not None
    # A new call must own a distinct buffer, not reuse the surviving view's.
    other = eikonal.station_travel_time_fields(model, stations[::-1], n_workers=3)
    assert not np.shares_memory(view, other)
    other.fill(-7.0)
    np.testing.assert_array_equal(view, expected[1:, ::2, :, :])
    view[0, 0, 0, 0] = 123.0
    assert view[0, 0, 0, 0] == 123.0
    del view
    gc.collect()
    assert buffer_refs[0]() is None


def test_raw_array_works_with_tempfile_backing_without_dev_shm(monkeypatch, tmp_path):
    # Force a fresh CPython RawArray heap with no /dev/shm candidate. This
    # exercises its normal disk-backed fallback without allocating >64 MiB.
    from multiprocessing import heap, util

    if not hasattr(heap.Arena, "_dir_candidates"):
        pytest.skip("this Python does not expose POSIX RawArray arena selection")
    monkeypatch.setattr(heap.BufferWrapper, "_heap", heap.Heap())
    monkeypatch.setattr(heap.Arena, "_dir_candidates", [])
    monkeypatch.setattr(util, "get_temp_dir", lambda: str(tmp_path))
    choose_dir = heap.Arena._choose_dir
    directories = []

    def tracked_choose_dir(self, size):
        directory = choose_dir(self, size)
        directories.append(directory)
        return directory

    monkeypatch.setattr(heap.Arena, "_choose_dir", tracked_choose_dir)
    model = _model()
    stations = _stations(model)[:2]
    parallel = eikonal.station_travel_time_fields(model, stations, n_workers=2)
    assert directories and all(path == str(tmp_path) for path in directories)
    np.testing.assert_array_equal(parallel, eikonal.station_travel_time_fields(model, stations))
