"""Tests for descriptive, date-groupable tomography run identifiers."""

import json
import re

import numpy as np
import pytest

from server import _load_G_station, _model_grid_step
from tomography.tomography_events import EventLog, _sparsify_G_stations
from tomography.tomography_logging import TomographyLogger


def _event_log(shape, indices, positions=None, weights=None, G_per_weight=None, ray_counts=None,
               ray_reached=None):
    indices = np.asarray(indices, dtype=np.int32)
    return EventLog(
        candidate_cells=indices,
        misfit_shape=np.asarray(shape, dtype=np.int32),
        positions=np.asarray(indices if positions is None else positions, dtype=np.float64),
        weights=np.ones(len(indices)) / len(indices) if weights is None else np.asarray(weights),
        misfit=None,
        residuals=np.array([]),
        G_per_weight=G_per_weight,
        ray_count_per_weight=ray_counts or {},
        ray_reached_per_weight=ray_reached or {},
    )


def test_logger_uses_descriptive_run_id_and_persists_identity(tmp_path):
    logger = TomographyLogger(
        base_dir=tmp_path,
        run_name="EM release",
        run_version="1.2.3",
        run_tags={"topn": 3, "lam": 0.01},
    )
    logger.save_meta({}, [], [])

    assert re.fullmatch(
        r"run_em-release_v1\.2\.3_topn-3_lam-0\.01_\d{8}_\d{6}_[0-9a-f]{32}", logger.run_id
    )

    meta = json.loads((logger.run_dir / "meta.json").read_text())
    assert meta["run_id"] == logger.run_dir.name == logger.run_id
    assert meta["run_name"] == "em-release"
    assert meta["run_version"] == "1.2.3"
    assert meta["run_tags"] == {"topn": "3", "lam": "0.01"}
    assert "started_at" in meta
    assert meta["viewer_completion_protocol"] == 1
    assert meta["viewer_ray_status_schema"] == 1


def test_runs_with_same_name_get_distinct_ids(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from tomography import tomography_logging

    class FixedDatetime:
        @staticmethod
        def now(tz):
            return datetime(2026, 10, 18, 12, 0, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(tomography_logging, "datetime", FixedDatetime)
    first = TomographyLogger(base_dir=tmp_path, run_name="em")
    second = TomographyLogger(base_dir=tmp_path, run_name="em")
    assert first.run_id != second.run_id
    assert first.run_id.rsplit("_", 1)[0] == second.run_id.rsplit("_", 1)[0]
    assert first.run_dir.is_dir() and second.run_dir.is_dir()


def test_event_log_keeps_refined_positions_and_coverage_without_g(tmp_path):
    logger = TomographyLogger(base_dir=tmp_path)
    weights = {
        "shape": np.array([2, 2, 2], dtype=np.int32),
        "indices": np.array([[1, 0, 1]], dtype=np.int32),
    }
    positions = np.array([[1.25, 0.1, 0.75]])
    ray_count = np.ones((1, 1, 1), dtype=np.int16)

    logger.save_event_data(0, 0, _event_log(
        weights["shape"], weights["indices"], positions, [1.0], ray_counts={0: ray_count},
    ))

    event_dir = logger.run_dir / "iter_0" / "event_0"
    with np.load(event_dir / "weights.npz") as saved:
        np.testing.assert_array_equal(saved["weight_shape"], weights["shape"])
        np.testing.assert_array_equal(saved["weight_indices"], weights["indices"])
        np.testing.assert_array_equal(saved["positions"], positions)
        np.testing.assert_array_equal(saved["weight_values"], [1.0])
    np.testing.assert_array_equal(
        np.load(event_dir / "weight_0" / "ray_count.npy"), ray_count
    )
    logger.save_ray_count(0, ray_count * 3)
    np.testing.assert_array_equal(
        np.load(logger.run_dir / "iter_0" / "ray_count.npy"), ray_count * 3
    )


def test_nearest_model_uses_coarse_grid_boundaries():
    meta = {"run_params": {"slowness_interpolation": "nearest"}}
    assert _model_grid_step(meta, (20, 7, 7), (180, 63, 63)) == [9, 9]
    meta["run_params"]["slowness_interpolation"] = "trilinear"
    assert _model_grid_step(meta, (20, 7, 7), (180, 63, 63)) == [1, 1]


def test_sparse_g_log_round_trip(tmp_path):
    logger = TomographyLogger(base_dir=tmp_path)
    dense = np.zeros((3, 8, 6, 5), dtype=np.float64)
    dense[0, 1, 2, 3] = 1.25
    dense[0, 2, 2, 3] = 0.75
    dense[2, 7, 5, 4] = 2.5

    logger.save_event_data(0, 0, _event_log(
        np.ones(3), [[0, 0, 0]], G_per_weight={0: _sparsify_G_stations(dense)},
    ))

    weight_dir = logger.run_dir / "iter_0" / "event_0" / "weight_0"
    assert (weight_dir / "G_stations_sparse.npz").exists()
    for station in range(dense.shape[0]):
        for y in range(dense.shape[2]):
            restored = _load_G_station(weight_dir / f"G_station_{station}", y)
            np.testing.assert_array_equal(restored, dense[station, :, y, :].astype(np.float32))


def _mock_em_cycle(monkeypatch):
    from tomography import tomography_em as em

    def step(model, arrivals, stations, noise, config, *, iteration, logger):
        shape = model.shape
        logger.save_station_fields(iteration, np.ones((len(stations), *shape)))
        log = _event_log(shape, [[0, 0, 0]],
                         ray_counts={0: np.ones(shape)},
                         ray_reached={0: np.ones(len(stations), dtype=bool)},
                         G_per_weight={0: _sparsify_G_stations(np.ones((len(stations), *shape)))})
        log.misfit = np.ones(shape)
        log.residuals = np.zeros((len(stations), len(stations)))
        logger.save_event_data(iteration, 0, log)
        logger.save_ray_count(iteration, np.ones(shape))
        logger.save_ray_status(iteration, [(0, log)], len(stations))
        return np.full(shape, 1e-6), np.full(shape, 2.), np.full(shape, 0.5)

    monkeypatch.setattr(em, "make_tomography_step", step)
    return em


@pytest.mark.parametrize("with_reference", [False, True])
def test_em_publishes_marker_last_preserving_pre_update_model(tmp_path, monkeypatch, with_reference):
    from config import InversionConfig
    from instruments.likelihood import PickNoise
    from velocity_model import VelocityModel

    em = _mock_em_cycle(monkeypatch)
    logger = TomographyLogger(base_dir=tmp_path, save_misfit=True, save_timefields=True)
    model = VelocityModel(np.full((2, 2, 2), 5000.), 1000.)
    original = logger._save_json
    publications = []

    def checked_save(path, values):
        if path.name == "complete.json":
            directory = path.parent
            expected = ["model.npy", "delta_s.npy", "sensitivity_diagonal.npy", "coverage_confidence.npy",
                        "ray_count.npy", "station_fields.npy", "event_0/weights.npz", "event_0/residuals.npy",
                        "event_0/misfit.npy", "event_0/weight_0/ray_count.npy",
                        "event_0/weight_0/G_stations_sparse.npz", "event_0/ray_status.npz",
                        "ray_status.json"]
            assert all((directory / name).is_file() for name in expected)
            assert json.loads((logger.run_dir / "timing.jsonl").read_text())["iter"] == 0
            assert (logger.run_dir / "quality.jsonl").exists() is with_reference
            assert not path.exists()
            publications.append(values)
        original(path, values)

    monkeypatch.setattr(logger, "_save_json", checked_save)
    config = InversionConfig(n_cycles=1, subdivision=1, n_workers=1, log_g_per_weight=True)
    em.run_em(config, model, [[0., 1.]], [[0, 0, 0], [1000, 0, 0]], PickNoise(), logger=logger,
              reference_model=model if with_reference else None)
    assert publications == [{"iter": 0, "viewer_completion_protocol": 1}]
    np.testing.assert_array_equal(np.load(logger.run_dir / "iter_0/model.npy"), model.velocity)
    np.testing.assert_allclose(np.load(logger.run_dir / "final_model.npy"), 1 / (1 / model.velocity + 1e-6))
    meta = json.loads((logger.run_dir / "meta.json").read_text())
    assert meta["viewer_completion_protocol"] == 1
    assert meta["run_params"]["viewer_quality_expected"] is with_reference
    assert meta["viewer_saved_artifacts"] == {"save_misfit": True, "save_timefields": True}
    assert not list(logger.run_dir.rglob("*.tmp"))


@pytest.mark.parametrize("failure", ["update_velocity", "save_station_fields", "save_event_data",
                                      "save_ray_count", "save_ray_status", "end_iteration", "save_delta_s",
                                      "save_inversion_diagnostics", "save_quality", "complete_iteration"])
def test_failed_update_or_save_never_publishes_cycle(tmp_path, monkeypatch, failure):
    from config import InversionConfig
    from instruments.likelihood import PickNoise
    from velocity_model import VelocityModel

    em = _mock_em_cycle(monkeypatch)
    logger = TomographyLogger(base_dir=tmp_path, save_timefields=True)
    model = VelocityModel(np.full((2, 2, 2), 5000.), 1000.)

    def fail(*args, **kwargs):
        raise OSError("injected failure")

    monkeypatch.setattr(em if failure == "update_velocity" else logger, failure, fail)
    with pytest.raises(OSError, match="injected failure"):
        em.run_em(InversionConfig(n_cycles=1, subdivision=1), model, [[0., 1.]],
                  [[0, 0, 0], [1000, 0, 0]], PickNoise(), reference_model=model, logger=logger)
    assert (logger.run_dir / "iter_0/model.npy").is_file()
    assert not (logger.run_dir / "iter_0/complete.json").exists()
    assert not (logger.run_dir / "final_model.npy").exists()


@pytest.mark.parametrize("extension", ["npy", "npz", "json"])
def test_atomic_save_keeps_previous_file_on_writer_failure(tmp_path, monkeypatch, extension):
    path = tmp_path / f"saved.{extension}"
    path.write_bytes(b"previous complete artifact")

    def fail(stream, *args, **kwargs):
        stream.write("partial" if extension == "json" else b"partial")
        raise OSError("disk failure")

    if extension == "json":
        monkeypatch.setattr(json, "dump", lambda values, stream, **kwargs: fail(stream))
        save = lambda: TomographyLogger._save_json(path, {"iter": 0})
    elif extension == "npz":
        monkeypatch.setattr(np, "savez_compressed", fail)
        save = lambda: TomographyLogger._save_npz(path, values=np.ones(2))
    else:
        monkeypatch.setattr(np, "save", fail)
        save = lambda: TomographyLogger._save_npy(path, np.ones(2))
    with pytest.raises(OSError, match="disk failure"):
        save()
    assert path.read_bytes() == b"previous complete artifact"


@pytest.mark.parametrize("masks, reached, used, insufficient", [
    ({0: [True, True, True], 2: [True, False, False]}, 4, 3, 1),
    ({0: [False, False, False], 2: [False, False, False]}, 0, 0, 2),
    ({0: [True, True, True], 2: [True, True, True]}, 6, 6, 0),
])
def test_ray_status_round_trip_without_optional_g(tmp_path, masks, reached, used, insufficient):
    logger = TomographyLogger(base_dir=tmp_path)
    log = _event_log((2, 2, 2), [[0, 0, 0]] * 3, weights=[0.25, 0., 0.75], ray_reached=masks)
    logger.save_event_data(4, 9, log)
    summary = logger.save_ray_status(4, [(9, log)], 3)
    directory = logger.run_dir / "iter_4"
    with np.load(directory / "event_9/ray_status.npz") as saved:
        assert set(saved.files) == {"hypothesis_indices", "reached"}
        assert saved["hypothesis_indices"].dtype.kind == "i"
        assert saved["reached"].dtype == np.bool_
        np.testing.assert_array_equal(saved["hypothesis_indices"], [0, 2])
        np.testing.assert_array_equal(saved["reached"], [masks[0], masks[2]])
    failures = [
        {"event": 9, "hypothesis": index, "stations": np.flatnonzero(~np.asarray(mask)).tolist()}
        for index, mask in masks.items() if not all(mask)
    ]
    assert summary == {
        "schema_version": 1, "expected": 6, "reached": reached, "failed": 6 - reached,
        "used": used, "n_events": 1, "n_hypotheses": 2,
        "events_with_failed_rays": int(reached < 6),
        "hypotheses_with_insufficient_stations": insufficient, "failures": failures,
    }
    assert json.loads((directory / "ray_status.json").read_text()) == summary
    assert not list(directory.rglob("G_stations*"))
    assert not list(directory.rglob("*.tmp"))


@pytest.mark.parametrize("partial", [False, True])
def test_missing_active_masks_never_fabricate_aggregate(tmp_path, partial):
    logger = TomographyLogger(base_dir=tmp_path)
    complete = _event_log((2, 2, 2), [[0, 0, 0]], ray_reached={0: [True, True]})
    legacy = _event_log((2, 2, 2), [[0, 0, 0]] * 2,
                        ray_reached={0: [True, True]} if partial else None)
    logger.save_event_data(0, 1, legacy)
    assert logger.save_ray_status(0, [(0, complete), (1, legacy)], 2) is None
    directory = logger.run_dir / "iter_0"
    assert not (directory / "ray_status.json").exists()
    assert (directory / "event_1/ray_status.npz").exists() is partial


def test_legacy_event_log_default_masks_are_independent():
    first = EventLog(np.zeros((1, 3)), np.ones(3), np.zeros((1, 3)), np.ones(1),
                     None, np.array([]), None, {})
    second = EventLog(np.zeros((1, 3)), np.ones(3), np.zeros((1, 3)), np.ones(1),
                      None, np.array([]), None, {})
    first.ray_reached_per_weight[0] = [True]
    assert second.ray_reached_per_weight == {}


@pytest.mark.parametrize("mask", [[True], [[True, False]], [True, False, True]])
def test_aggregate_validates_station_count(tmp_path, mask):
    logger = TomographyLogger(base_dir=tmp_path)
    log = _event_log((2, 2, 2), [[0, 0, 0]], ray_reached={0: mask})
    with pytest.raises(ValueError, match="must have shape \\(2,\\)"):
        logger.save_ray_status(0, [(0, log)], 2)
    assert not (logger.run_dir / "iter_0/ray_status.json").exists()


def _mock_event_computation(monkeypatch, masks):
    from tomography import tomography_em as em

    log = _event_log((2, 2, 2), [[0, 0, 0]] * 2, weights=[0.5, 0.5],
                     ray_counts={0: np.zeros((2, 2, 2)), 1: np.zeros((2, 2, 2))},
                     ray_reached=masks)
    monkeypatch.setattr(em, "station_travel_time_fields",
                        lambda fine, stations, workers: np.ones((len(stations), *fine.shape)))
    monkeypatch.setattr(em, "run_events",
                        lambda *args: (np.zeros((8, 8)), np.zeros(8), [(0, log)]))
    monkeypatch.setattr(em, "solve_slowness_update",
                        lambda *args, **kwargs: (np.zeros((2, 2, 2)), np.ones((2, 2, 2)),
                                                np.ones((2, 2, 2))))
    return em


@pytest.mark.parametrize("log_g", [False, True])
@pytest.mark.parametrize("all_fail", [False, True])
def test_real_step_saves_status_before_marker_and_warns_without_stopping(
    tmp_path, monkeypatch, capsys, log_g, all_fail,
):
    from config import InversionConfig
    from instruments.likelihood import PickNoise
    from velocity_model import VelocityModel

    masks = {0: [False, False], 1: [False, False]} if all_fail else {
        0: [True, True], 1: [True, False],
    }
    em = _mock_event_computation(monkeypatch, masks)
    logger = TomographyLogger(base_dir=tmp_path)
    model = VelocityModel(np.full((2, 2, 2), 5000.), 1000.)
    original_json = logger._save_json
    original_status = logger.save_ray_status
    publications = []

    def checked_status(iteration, event_logs, n_stations):
        directory = logger.run_dir / f"iter_{iteration}"
        assert (directory / "event_0/ray_status.npz").exists()
        assert (directory / "ray_count.npy").exists()
        return original_status(iteration, event_logs, n_stations)

    def checked_json(path, values):
        publications.append(path.name)
        if path.name == "complete.json":
            assert (path.parent / "ray_status.json").exists()
        original_json(path, values)

    monkeypatch.setattr(logger, "save_ray_status", checked_status)
    monkeypatch.setattr(logger, "_save_json", checked_json)
    em.run_em(InversionConfig(n_cycles=1, subdivision=1, log_g_per_weight=log_g), model,
              [[0., 1.]], [[0, 0, 0], [1000, 0, 0]], PickNoise(), logger=logger)
    assert publications.index("ray_status.json") < publications.index("complete.json")
    summary = json.loads((logger.run_dir / "iter_0/ray_status.json").read_text())
    assert summary["expected"] == 4
    assert summary["reached"] == (0 if all_fail else 3)
    assert summary["used"] == (0 if all_fail else 2)
    output = capsys.readouterr().out
    assert f"[Rays] {summary['reached']}/4 reached, {summary['failed']} failed" in output
    assert "WARNING" in output
    assert (logger.run_dir / "iter_0/complete.json").exists()


@pytest.mark.parametrize("artifact", ["ray_status.npz", "ray_status.json"])
def test_failed_ray_status_write_never_publishes_cycle(tmp_path, monkeypatch, artifact):
    from config import InversionConfig
    from instruments.likelihood import PickNoise
    from velocity_model import VelocityModel

    em = _mock_event_computation(monkeypatch, {0: [True, False], 1: [False, False]})
    logger = TomographyLogger(base_dir=tmp_path)
    method = "_save_npz" if artifact.endswith("npz") else "_save_json"
    original = getattr(logger, method)

    def fail_status(path, *args, **kwargs):
        if path.name == artifact:
            raise OSError("ray status disk failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(logger, method, fail_status)
    model = VelocityModel(np.full((2, 2, 2), 5000.), 1000.)
    with pytest.raises(OSError, match="ray status disk failure"):
        em.run_em(InversionConfig(n_cycles=1, subdivision=1), model, [[0., 1.]],
                  [[0, 0, 0], [1000, 0, 0]], PickNoise(), logger=logger)
    assert not (logger.run_dir / "iter_0/complete.json").exists()
    assert not (logger.run_dir / "final_model.npy").exists()


def test_step_prints_ray_status_without_logger(monkeypatch, capsys):
    from config import InversionConfig
    from instruments.likelihood import PickNoise
    from velocity_model import VelocityModel

    em = _mock_event_computation(monkeypatch, {0: [True, True], 1: [True, True]})
    model = VelocityModel(np.full((2, 2, 2), 5000.), 1000.)
    em.make_tomography_step(model, np.array([[0., 1.]]), np.array([[0, 0, 0], [1000, 0, 0]]),
                            PickNoise(), InversionConfig(subdivision=1, save_runs=False))
    output = capsys.readouterr().out
    assert "[Rays] 4/4 reached, 0 failed, 4 used" in output
    assert "WARNING" not in output
