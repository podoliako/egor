"""Tests for descriptive, date-groupable tomography run identifiers."""

import json
import re

import numpy as np

from server import _load_G_station
from tomography.tomography_events import _sparsify_G_stations
from tomography.tomography_logging import TomographyLogger


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


def test_event_log_keeps_refined_positions_and_coverage_without_g(tmp_path):
    logger = TomographyLogger(base_dir=tmp_path)
    weights = np.zeros((2, 2, 2), dtype=np.float64)
    weights[1, 0, 1] = 1.0
    positions = np.array([[1.25, 0.1, 0.75]])
    ray_count = np.ones((1, 1, 1), dtype=np.int16)

    logger.save_event_data(
        iteration=0,
        event_idx=0,
        weights=weights,
        positions=positions,
        weight_values=np.array([1.0]),
        G_per_weight=None,
        ray_count_per_weight={0: ray_count},
    )

    event_dir = logger.run_dir / "iter_0" / "event_0"
    with np.load(event_dir / "weights.npz") as saved:
        np.testing.assert_array_equal(saved["weights"], weights)
        np.testing.assert_array_equal(saved["positions"], positions)
        np.testing.assert_array_equal(saved["weight_values"], [1.0])
    np.testing.assert_array_equal(
        np.load(event_dir / "weight_0" / "ray_count.npy"), ray_count
    )


def test_sparse_g_log_round_trip(tmp_path):
    logger = TomographyLogger(base_dir=tmp_path)
    dense = np.zeros((3, 8, 6, 5), dtype=np.float64)
    dense[0, 1, 2, 3] = 1.25
    dense[0, 2, 2, 3] = 0.75
    dense[2, 7, 5, 4] = 2.5

    logger.save_event_data(
        iteration=0,
        event_idx=0,
        weights=np.ones((1, 1, 1)),
        G_per_weight={0: _sparsify_G_stations(dense)},
    )

    weight_dir = logger.run_dir / "iter_0" / "event_0" / "weight_0"
    assert (weight_dir / "G_stations_sparse.npz").exists()
    for station in range(dense.shape[0]):
        restored = _load_G_station(weight_dir / f"G_station_{station}")
        np.testing.assert_array_equal(restored, dense[station].astype(np.float32))
