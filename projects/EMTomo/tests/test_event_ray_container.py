"""Lossless event ray archives, old-run compatibility, and viewer contracts."""
import hashlib
import json
from shutil import copyfile
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

import server
from tomography.tomography_events import EventLog
from tomography.tomography_logging import TomographyLogger


def _log(with_g=True):
    shape = (4, 3, 2)
    weights = np.zeros(12)
    weights[[2, 10]] = [0.25, 0.75]
    fields = {
        "shape": np.array(shape, dtype=np.int32),
        "offsets": np.array([0, 2, 2, 3], dtype=np.int64),
        "coords": np.array([[1, 0, 1], [3, 2, 0], [2, 1, 1]], dtype=np.int32),
        "values": np.array([1.25, 2.5, 0.75], dtype=np.float32),
    }
    empty = {**fields, "offsets": np.zeros(4, dtype=np.int64),
             "coords": np.empty((0, 3), dtype=np.int32), "values": np.empty(0, dtype=np.float32)}
    return EventLog(
        candidate_cells=np.tile([[1, 0, 1]], (12, 1)), misfit_shape=np.array(shape),
        positions=np.arange(36, dtype=float).reshape(12, 3), weights=weights,
        misfit=None, residuals=np.array([[0., np.nan, 2.], [np.nan]*3, [-2., np.nan, 0.]]),
        # Include a G-only ID and a coverage-only ID; don't infer IDs from weights.
        G_per_weight={10: empty, 2: fields, 7: fields} if with_g else None,
        ray_count_per_weight={10: np.full(shape, 3, dtype=np.int16),
                              2: np.arange(24, dtype=np.int16).reshape(shape),
                              11: np.zeros(shape, dtype=np.int16)},
        ray_reached_per_weight={2: np.array([True, False, True]), 10: np.zeros(3, dtype=bool)},
        observed_mask=np.array([True, False, True]),
    )


def _legacy_rays(event, log):
    for index, counts in log.ray_count_per_weight.items():
        weight = event / f"weight_{index}"
        weight.mkdir(exist_ok=True)
        np.save(weight / "ray_count.npy", counts)
    for index, fields in (log.G_per_weight or {}).items():
        weight = event / f"weight_{index}"
        weight.mkdir(exist_ok=True)
        np.savez_compressed(weight / "G_stations_sparse.npz", **fields)


def _assert_exact(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("with_g", [True, False])
def test_event_archive_preserves_every_array_and_reduces_files(tmp_path, with_g):
    log = _log(with_g)
    # Exercise endian, strides, Fortran order and floating-point bit patterns too.
    log.ray_count_per_weight[11] = np.asfortranarray(np.arange(24, dtype=">i4").reshape(4, 3, 2))
    if with_g:
        log.G_per_weight[2]["values"] = np.array([0x80000000, 0x7fc00001, 0x3fa00000], dtype=np.uint32).view(np.float32)
        log.G_per_weight[7] = {**log.G_per_weight[7], "values": np.arange(6, dtype=np.float64)[::2]}
    logger = TomographyLogger(base_dir=tmp_path)
    logger.save_event_data(0, 0, log)
    event = logger.run_dir / "iter_0/event_0"
    assert {p.name for p in event.iterdir()} == {"rays.npz", "weights.npz", "residuals.npy", "ray_status.npz"}
    with np.load(event / "rays.npz", allow_pickle=False) as data:
        assert data["schema_version"] == 1
        np.testing.assert_array_equal(data["ray_count_indices"], [2, 10, 11])
        np.testing.assert_array_equal(data["g_indices"], [2, 7, 10] if with_g else [])
        for index, counts in log.ray_count_per_weight.items():
            _assert_exact(data[f"ray_count_{index}"], counts)
        for index, fields in (log.G_per_weight or {}).items():
            for name, array in fields.items():
                _assert_exact(data[f"g_{index}_{name}"], array)
    with ZipFile(event / "rays.npz") as archive:
        assert all(member.compress_type == ZIP_DEFLATED for member in archive.infolist())
    with np.load(event / "weights.npz") as data:
        np.testing.assert_array_equal(data["positions"], log.positions)
        np.testing.assert_array_equal(data["weight_values"], log.weights)
        np.testing.assert_array_equal(data["observed_mask"], log.observed_mask)
    with np.load(event / "ray_status.npz") as data:
        np.testing.assert_array_equal(data["hypothesis_indices"], [2, 10])
        np.testing.assert_array_equal(data["reached"], [log.ray_reached_per_weight[i] for i in [2, 10]])
        np.testing.assert_array_equal(data["observed_mask"], log.observed_mask)
    _assert_exact(np.load(event / "residuals.npy"), log.residuals)


def _runs(tmp_path, monkeypatch, with_g):
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    monkeypatch.setitem(server.app.config, "TESTING", True)
    log = _log(with_g)
    experiments = tmp_path / "experiments"
    model = experiments / "input/sample/model.npz"
    model.parent.mkdir(parents=True)
    np.savez(model, velocity=np.ones((4, 3, 2)), cell_size_m=1000., origin_m=np.zeros(3))
    monkeypatch.setattr(server, "EXPERIMENTS_ROOT", experiments)
    source = {"id": "sample", "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest()}
    logger = TomographyLogger(base_dir=tmp_path)
    logger.save_meta({"n_events": 1, "log_g_per_weight": with_g, "viewer_quality_expected": False}, [], [],
                     {"coarse_shape": [4, 3, 2], "coarse_side_m": [4000, 3000, 2000]}, source)
    logger.save_event_data(0, 0, log)
    logger.save_ray_status(0, [(0, log)], 3)
    logger.save_iteration_model(0, np.full((4, 3, 2), 5000.))
    logger.save_delta_s(0, np.ones((4, 3, 2)))
    logger.complete_iteration(0)
    old = tmp_path / "old"
    event = old / "iter_0/event_0"
    event.mkdir(parents=True)
    meta = json.loads((logger.run_dir / "meta.json").read_text())
    meta.pop("viewer_event_rays_schema")
    (old / "meta.json").write_text(json.dumps(meta))
    for name in ("complete.json", "model.npy", "delta_s.npy", "ray_status.json"):
        copyfile(logger.run_dir / "iter_0" / name, old / "iter_0" / name)
    for name in ("weights.npz", "residuals.npy", "ray_status.npz"):
        copyfile(logger.run_dir / "iter_0/event_0" / name, event / name)
    _legacy_rays(event, log)
    return logger, old, log, server.app.test_client()


@pytest.mark.parametrize("with_g", [True, False])
def test_old_and_new_viewer_responses_are_identical(tmp_path, monkeypatch, with_g):
    logger, old, log, client = _runs(tmp_path, monkeypatch, with_g)
    endpoints = ["iters_list", "events_list?iter=0", "weights_list?iter=0&event=0", "ray_status?iter=0"]
    endpoints += [f"slice?type={kind}&iter=0&event=0&y_km={y + 0.5}"
                  for kind in ("weights", "ray_count") for y in range(3)]
    endpoints += [f"slice?type=G&iter=0&event=0&weight={weight}&station={station}&y_km={y + 0.5}"
                  for weight in (0, 2, 7, 10, 11, 99) for station in (0, 1, 2, 3) for y in range(3)]
    endpoints += ["slice?type=G&event=0&weight=2&station=0&y_km=2.5"]
    for endpoint in endpoints:
        new_response = client.get(f"/api/runs/{logger.run_id}/{endpoint}")
        old_response = client.get(f"/api/runs/{old.name}/{endpoint}")
        assert new_response.status_code == old_response.status_code == 200, endpoint
        assert new_response.json == old_response.json, endpoint
    assert client.get(f"/api/runs/{logger.run_id}/weights_list").json == (
        ["weight_2", "weight_7", "weight_10", "weight_11"] if with_g else ["weight_2", "weight_10", "weight_11"])
    # New archives can infer stations from a nonzero hypothesis ID without metadata.
    assert client.get(f"/api/runs/{logger.run_id}/info").json["n_stations"] == (3 if with_g else 0)
    for index in (2, 10):
        for station in range(3):
            stem = f"iter_0/event_0/weight_{index}/G_station_{station}"
            np.testing.assert_array_equal(server._load_G_station(logger.run_dir / stem, 1),
                                          server._load_G_station(old / stem, 1))
    # No iteration-level aggregate: both layouts use the per-hypothesis fallback.
    np.testing.assert_array_equal(server._cached_ray_count(logger.run_dir / "iter_0"),
                                  server._cached_ray_count(old / "iter_0"))
    logger.save_ray_count(0, np.full((4, 3, 2), 99, dtype=np.int64))
    np.testing.assert_array_equal(server._cached_ray_count(logger.run_dir / "iter_0"), 99)


def test_selective_g_reads_do_not_inflate_other_hypotheses_or_coverage(tmp_path, monkeypatch):
    logger, _, _, client = _runs(tmp_path, monkeypatch, True)
    original = np.lib.npyio.NpzFile.__getitem__
    accessed = []

    def tracked(data, key):
        if "schema_version" in data.files:
            accessed.append(key)
            assert key in {"schema_version", "g_indices", "g_2_shape", "g_2_offsets", "g_2_coords", "g_2_values"}
        return original(data, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", tracked)
    response = client.get(f"/api/runs/{logger.run_id}/slice?type=G&weight=2&station=0")
    assert response.status_code == 200
    assert "g_2_values" in accessed


def test_atomic_archive_failure_preserves_previous_and_removes_temp(tmp_path, monkeypatch):
    path = tmp_path / "rays.npz"
    log = _log()
    TomographyLogger._save_event_rays(path, log.ray_count_per_weight, log.G_per_weight)
    before = path.read_bytes()
    original = np.lib.format.write_array
    calls = []

    def fail(stream, array, **kwargs):
        calls.append(1)
        if len(calls) == 5:
            raise OSError("simulated full disk")
        return original(stream, array, **kwargs)

    monkeypatch.setattr(np.lib.format, "write_array", fail)
    with pytest.raises(OSError, match="full disk"):
        TomographyLogger._save_event_rays(path, log.ray_count_per_weight, log.G_per_weight)
    assert path.read_bytes() == before
    assert not path.with_suffix(".npz.tmp").exists()


def test_empty_archive_and_replacement_invalidate_fallback_cache(tmp_path):
    event = tmp_path / "event_0"
    event.mkdir()
    path = event / "rays.npz"
    TomographyLogger._save_event_rays(path, {}, {})
    assert server._cached_ray_count(tmp_path) is None
    TomographyLogger._save_event_rays(path, {5: np.ones((2, 2, 2), dtype=np.int16)}, {})
    np.testing.assert_array_equal(server._cached_ray_count(tmp_path), 1)
    # An authoritative archive must not double-count stale weight directories.
    _legacy_rays(event, _log(False))
    np.testing.assert_array_equal(server._cached_ray_count(tmp_path), 1)
    other = tmp_path / "event_1"
    other.mkdir()
    counts = _log(False)
    counts.ray_count_per_weight = {0: np.full((2, 2, 2), 2, dtype=np.int16)}
    _legacy_rays(other, counts)
    np.testing.assert_array_equal(server._cached_ray_count(tmp_path), 3)


def test_compact_completion_requires_marker_and_supports_legacy_inference(tmp_path, monkeypatch):
    logger, _, _, client = _runs(tmp_path, monkeypatch, True)
    marker = logger.run_dir / "iter_0/complete.json"
    marker.unlink()
    prefix = f"/api/runs/{logger.run_id}"
    assert client.get(prefix + "/weights_list").json == []
    assert client.get(prefix + "/slice?type=G&weight=2").status_code == 404
    logger.complete_iteration(0)
    assert client.get(prefix + "/iters_list").json == ["iter_0"]
    # A copied archive in an unmarked layout is checked for all active IDs.
    meta_path = logger.run_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta.pop("viewer_completion_protocol")
    meta_path.write_text(json.dumps(meta))
    (logger.run_dir / "timing.jsonl").write_text('{"iter": 0, "elapsed_s": 1}\n')
    assert client.get(prefix + "/iters_list").json == ["iter_0"]


@pytest.mark.parametrize("missing_kind", ["ray_count", "g"])
def test_unmarked_compact_run_waits_for_missing_ids_and_detects_publication(tmp_path, monkeypatch, missing_kind):
    logger, _, log, client = _runs(tmp_path, monkeypatch, True)
    meta_path = logger.run_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta.pop("viewer_completion_protocol")
    meta_path.write_text(json.dumps(meta))
    (logger.run_dir / "timing.jsonl").write_text('{"iter": 0, "elapsed_s": 1}\n')
    path = logger.run_dir / "iter_0/event_0/rays.npz"
    counts, rays = dict(log.ray_count_per_weight), dict(log.G_per_weight)
    (counts if missing_kind == "ray_count" else rays).pop(2)
    logger._save_event_rays(path, counts, rays)
    url = f"/api/runs/{logger.run_id}/iters_list"
    assert client.get(url).json == []
    logger._save_event_rays(path, log.ray_count_per_weight, log.G_per_weight)
    assert client.get(url).json == ["iter_0"]


def test_archive_symlink_cannot_escape_run(tmp_path, monkeypatch):
    logger, _, _, client = _runs(tmp_path, monkeypatch, True)
    path = logger.run_dir / "iter_0/event_0/rays.npz"
    outside = tmp_path / "outside.npz"
    path.replace(outside)
    path.symlink_to(outside)
    prefix = f"/api/runs/{logger.run_id}"
    for endpoint in ("weights_list", "info", "slice?type=G&weight=2"):
        assert client.get(prefix + "/" + endpoint).status_code == 404
    assert server._cached_ray_count(logger.run_dir / "iter_0") is None


def test_fifty_hypotheses_use_one_ray_file_and_no_weight_directories(tmp_path):
    log = _log()
    log.ray_count_per_weight = {i: log.ray_count_per_weight[2] for i in range(50)}
    log.G_per_weight = {i: log.G_per_weight[2] for i in range(50)}
    logger = TomographyLogger(base_dir=tmp_path)
    logger.save_event_data(0, 0, log)
    event = logger.run_dir / "iter_0/event_0"
    assert len(list(event.iterdir())) == 4
    assert not list(event.glob("weight_*"))
    old = tmp_path / "old"
    old.mkdir()
    _legacy_rays(old, log)
    assert len(list(old.glob("weight_*"))) == 50
    assert len([p for p in old.rglob("*") if p.is_file()]) == 100
    with np.load(event / "rays.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["ray_count_indices"], np.arange(50))
        np.testing.assert_array_equal(data["g_indices"], np.arange(50))


def test_fallback_preserves_legacy_float32_accumulation_order(tmp_path):
    shape = (1, 1, 1)
    log = _log(False)
    log.ray_count_per_weight = {2: np.full(shape, -1e8), 1: np.full(shape, 1e8), 10: np.ones(shape)}
    old = tmp_path / "old/event_0"
    new = tmp_path / "new/event_0"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    _legacy_rays(old, log)
    TomographyLogger._save_event_rays(new / "rays.npz", log.ray_count_per_weight, {})
    _assert_exact(server._cached_ray_count(new.parent), server._cached_ray_count(old.parent))
