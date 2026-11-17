"""Focused synthetic tests; no archive or inversion dependencies."""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("residual_metrics", Path(__file__).with_name("residual_metrics.py"))
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


class ResidualMetricsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        (self.run / "meta.json").write_text(json.dumps({"n_events": 2}))
        self.directory = self.run / "iter_0"
        self.directory.mkdir()
        (self.directory / "complete.json").write_text("{}")
        self.save(0, [[99, 3], [-3, 99]])
        self.save(1, [[0, 4, 4], [-4, 0, 4], [-4, -4, 0]])
        self.row = {"run_id": "test", "run_dir": "test", "section": "2", "n_events": 2}

    def save(self, event, matrix):
        p = self.directory / f"event_{event}"
        p.mkdir(exist_ok=True)
        np.save(p / "residuals.npy", np.array(matrix, dtype=float))

    def test_pooled_and_mean_varying_station_counts(self):
        point = metrics.compute_iteration(self.directory, 2)
        self.assertAlmostEqual(point["rmse_s"], math.sqrt(57 / 4))
        self.assertEqual(point["mean_event_rms_s"], 3.5)
        self.assertEqual(point["n_pairs"], 4)

    def test_nan_mask(self):
        self.save(1, [[0, 4, np.nan], [-4, 0, 4], [np.nan, -4, 0]])
        point = metrics.compute_iteration(self.directory, 2)
        self.assertAlmostEqual(point["rmse_s"], math.sqrt(41 / 3))
        self.assertEqual(point["n_pairs"], 3)

    def test_missing_event_fails(self):
        (self.directory / "event_1/residuals.npy").unlink()
        with self.assertRaises(FileNotFoundError):
            metrics.compute_iteration(self.directory, 2)

    def test_inf_rejected(self):
        self.save(0, [[0, np.inf], [0, 0]])
        with self.assertRaisesRegex(ValueError, "Infinite"):
            metrics.compute_iteration(self.directory, 2)

    def test_empty_pairs_rejected(self):
        self.save(0, [[0, np.nan], [0, 0]])
        with self.assertRaisesRegex(ValueError, "No finite"):
            metrics.compute_iteration(self.directory, 2)

    def test_incomplete_excluded_and_absent(self):
        incomplete = self.run / "iter_1/event_0"
        incomplete.mkdir(parents=True)
        np.save(incomplete / "residuals.npy", np.zeros((2, 2)))
        result = metrics.extract_run(self.run, self.row)
        self.assertEqual(result["n_complete_iterations"], 1)
        self.assertEqual(result["best"]["iteration"], 0)
        (self.directory / "complete.json").unlink()
        self.assertEqual(metrics.extract_run(self.run, self.row)["reason"], "zero_complete_iterations")
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            metrics.compute_iteration(self.directory, 2)

    def test_tie_and_global_selection(self):
        points = [{"iteration": 3, "rmse_s": 1, "mean_event_rms_s": 0.1},
                  {"iteration": 1, "rmse_s": 1, "mean_event_rms_s": 0.9},
                  {"iteration": 0, "rmse_s": 2, "mean_event_rms_s": 0.01}]
        self.assertEqual(metrics.select_best(points)["iteration"], 1)
        self.assertIsNone(metrics.select_best([]))

    def test_signature_detects_metadata_and_marker_edits(self):
        before = metrics.snapshot_signature(self.run, self.row)
        (self.run / "meta.json").write_text('{"changed": true}')
        self.assertNotEqual(before, metrics.snapshot_signature(self.run, self.row))
        before = metrics.snapshot_signature(self.run, self.row)
        (self.directory / "complete.json").write_text('{"changed": true}')
        self.assertNotEqual(before, metrics.snapshot_signature(self.run, self.row))


if __name__ == "__main__":
    unittest.main()
