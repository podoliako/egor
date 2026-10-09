"""Tests for compact event-weight logging and viewer compatibility."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from server import _load_weights
from tomography.tomography_events import EventLog
from tomography.tomography_logging import TomographyLogger


def test_compact_weights_round_trip_without_dense_storage():
    with TemporaryDirectory() as directory:
        logger = TomographyLogger(base_dir=directory)
        compact = {
            "shape": np.asarray([8, 6, 5], dtype=np.int32),
            "indices": np.asarray([[1, 2, 3], [7, 5, 4]], dtype=np.int32),
        }
        values = np.asarray([0.25, 0.75], dtype=np.float64)
        logger.save_event_data(0, 0, EventLog(
            candidate_cells=compact["indices"],
            misfit_shape=compact["shape"],
            positions=np.asarray([[1.1, 2.2, 3.3], [7.0, 5.0, 4.0]]),
            weights=values,
            misfit=None,
            residuals=np.array([]),
            G_per_weight=None,
            ray_count_per_weight={},
        ))
        path = Path(logger.run_dir) / "iter_0" / "event_0" / "weights.npz"

        with np.load(path) as saved:
            assert "weights" not in saved
            np.testing.assert_array_equal(saved["weight_shape"], [8, 6, 5])
            np.testing.assert_array_equal(saved["weight_indices"], compact["indices"])

        restored = _load_weights(path)
        expected = np.zeros((8, 6, 5), dtype=np.float64)
        expected[1, 2, 3] = 0.25
        expected[7, 5, 4] = 0.75
        np.testing.assert_array_equal(restored, expected)


def test_unlinked_dense_weights_are_not_loaded_as_new_run_data():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "weights.npz"
        expected = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        np.savez_compressed(path, weights=expected)

        assert _load_weights(path) is None
