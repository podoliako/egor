"""Tests for descriptive, date-groupable tomography run identifiers."""

import json
import re

import numpy as np
import pytest

from server import _load_G_station, _model_grid_step
from tomography.tomography_events import EventLog, _sparsify_G_stations
from tomography.tomography_logging import TomographyLogger


def _event_log(shape, indices, positions=None, weights=None, G_per_weight=None, ray_counts=None):
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
        r"run_em-release_v1\.2\.3_topn-3_lam-0\.01_\d{8}_\d{6}", logger.run_id
    )

    meta = json.loads((logger.run_dir / "meta.json").read_text())
    assert meta["run_name"] == "em-release"
    assert meta["run_version"] == "1.2.3"
    assert meta["run_tags"] == {"topn": "3", "lam": "0.01"}
    assert "started_at" in meta
    assert meta["viewer_completion_protocol"] == 1


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
                         G_per_weight={0: _sparsify_G_stations(np.ones((len(stations), *shape)))})
        log.misfit = np.ones(shape)
        log.residuals = np.zeros((len(stations), len(stations)))
        logger.save_event_data(iteration, 0, log)
        logger.save_ray_count(iteration, np.ones(shape))
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
                        "event_0/weight_0/G_stations_sparse.npz"]
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
                                      "save_ray_count", "end_iteration", "save_delta_s",
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
