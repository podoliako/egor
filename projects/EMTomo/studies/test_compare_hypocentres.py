"""Fast fixture tests: saved-run analysis only, never run EM or forward modeling."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

import numpy as np

import compare_hypocentres as study


class SavedPairTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        experiment = self.root / "experiments" / "input" / "fixture"
        experiment.mkdir(parents=True)
        (experiment / "model.npz").write_bytes(b"saved forward model")
        output = self.root / "experiments" / "output" / "fixture"
        output.mkdir(parents=True)
        (output / "metadata.json").write_text(json.dumps({"noise": {"enabled": False}}))
        digest = hashlib.sha256(b"saved forward model").hexdigest()
        self.reference = np.full((1, 1, 1), 5500.)
        params = {
            "candidate_mode": "soft", "weights_top_n": 2, "n_cycles": 2,
            "n_events": 2, "subdivision": 6, "coarse_side_m": 3000.,
            "weight_likelihood": "gaussian_independent_absolute_picks_marginal_origin_uniform_candidates",
            "normal_equations": "station_precision_weighted_profiled_origin",
            "max_velocity_step_fraction": None,
        }
        self.directories = {}
        for mode, values in (("hard", [5100., 5300.]), ("soft", [5200., 5400.])):
            directory = self.root / mode
            directory.mkdir()
            self.directories[mode] = directory
            np.save(directory / "initial_model.npy", np.full((1, 1, 1), 5000.))
            metadata = {
                "source_experiment": {"id": "fixture", "model_sha256": digest},
                "run_version": "1.3", "station_locs": [[0, 0, 0], [1, 0, 0]],
                "grid_info": {"coarse_shape": [1, 1, 1]},
                "run_params": {**params, "candidate_mode": mode},
            }
            (directory / "meta.json").write_text(json.dumps(metadata))
            pre = 5000.
            quality = []
            for i, post in enumerate(values):
                iteration = directory / f"iter_{i}"
                iteration.mkdir()
                np.save(iteration / "model.npy", np.full((1, 1, 1), pre))
                np.save(iteration / "delta_s.npy", np.full((1, 1, 1), 1 / post - 1 / pre))
                quality.append(json.dumps({"iter": i, "avg_abs_pct_dev": abs(post - 5500) / 5500 * 100}))
                for event in range(2):
                    event_dir = iteration / f"event_{event}"
                    event_dir.mkdir()
                    weights = [0.6, 0.4] if mode == "soft" else [1., 0.]
                    np.savez(event_dir / "weights.npz", weight_values=weights,
                             positions=[[0, 0, 0], [1, 0, 0]])
                pre = post
            (directory / "quality.jsonl").write_text("\n".join(quality) + "\n")

    def report(self):
        with patch.object(study, "prepare_inversion") as prepared:
            prepared.return_value.reference_model.velocity = self.reference
            return study.analyze("fixture", self.root / "experiments",
                                 self.directories["soft"], self.directories["hard"])

    def test_aligned_post_update_metrics_best_and_posterior(self):
        report = self.report()
        self.assertEqual(report["predefined_terminal_cycle"], 2)
        self.assertEqual(report["best_observed_cycle_post_hoc"], 1)
        self.assertAlmostEqual(report["cycles"][0]["rmse_gain_m_s"], 100.)
        self.assertAlmostEqual(report["cycles"][1]["soft"]["mape_pct"], 100 / 5500 * 100)
        self.assertAlmostEqual(report["cycles"][0]["soft_posterior"]["mean_effective_count"], 1 / (0.6**2 + 0.4**2))
        self.assertAlmostEqual(report["cycles"][0]["soft_posterior"]["mean_max_non_map_separation_gt_0_01_km"], 0.5)
        self.assertEqual(report["terminal_rmse_gain_m_s"], 100.)

    def test_mismatched_pair_rejected(self):
        path = self.directories["hard"] / "meta.json"
        meta = json.loads(path.read_text())
        meta["run_params"]["weights_top_n"] = 3
        path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "settings differ"):
            self.report()

    def test_quality_alignment_checked(self):
        path = self.directories["soft"] / "quality.jsonl"
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        rows[0]["avg_abs_pct_dev"] = 0.
        path.write_text("\n".join(map(json.dumps, rows)))
        with self.assertRaisesRegex(ValueError, "differs from quality.jsonl"):
            self.report()

    def test_positive_slowness_and_trust_region(self):
        updated = study._post_update(np.full((1, 1, 1), 5000.), np.full((1, 1, 1), -1.),
                                     {"max_velocity_step_fraction": 0.03})
        np.testing.assert_array_equal(updated, np.full((1, 1, 1), 5150.))

    def test_post_update_matches_float32_velocity_grid_storage(self):
        pre = np.full((1, 1, 1), 5000., dtype=np.float32)
        delta = np.full((1, 1, 1), 1 / 5100 - 1 / 5000)
        updated = study._post_update(pre, delta, {"max_velocity_step_fraction": None})
        self.assertEqual(updated.dtype, np.float32)
        np.testing.assert_array_equal(updated, np.array([[[5100.]]], dtype=np.float32))

    def test_launcher_uses_one_validated_saved_experiment_and_matched_configs(self):
        report = self.report()
        calls = []

        def fake_main(config, *, experiment_id, experiments_root, validate_only=False):
            calls.append((config, experiment_id, experiments_root, validate_only))
            if not validate_only:
                return SimpleNamespace(run_dir=self.directories[config.candidate_mode])

        with patch.object(study, "run_main", side_effect=fake_main), patch.object(study, "analyze", return_value=report), redirect_stdout(StringIO()):
            self.assertEqual(study.cli(["fixture", "--experiments-root", str(self.root / "experiments"),
                                        "--runs-dir", str(self.root / "reports"), "--cycles", "2"]), 0)
        self.assertEqual([call[3] for call in calls], [True, False, False])
        self.assertEqual([call[0].candidate_mode for call in calls[1:]], ["hard", "soft"])
        self.assertEqual([call[0].weights_top_n for call in calls[1:]], [8, 8])
        self.assertEqual([call[0].cell_size for call in calls[1:]], [3000., 3000.])
        self.assertTrue(all(call[1] == "fixture" for call in calls))
        self.assertEqual(len(list((self.root / "reports").glob("*.json"))), 1)

    def test_no_gain_and_zero_hard_rmse_are_reported(self):
        # Switch the hard final step to reach truth exactly, keeping the quality log aligned.
        directory = self.directories["hard"]
        np.save(directory / "iter_0" / "delta_s.npy", np.full((1, 1, 1), 1 / 5400 - 1 / 5000))
        np.save(directory / "iter_1" / "model.npy", np.full((1, 1, 1), 5400.))
        np.save(directory / "iter_1" / "delta_s.npy", np.full((1, 1, 1), 1 / 5500 - 1 / 5400))
        quality = [json.loads(s) for s in (directory / "quality.jsonl").read_text().splitlines()]
        quality[0]["avg_abs_pct_dev"] = 100 / 5500 * 100
        quality[1]["avg_abs_pct_dev"] = 0.
        (directory / "quality.jsonl").write_text("\n".join(map(json.dumps, quality)))
        report = self.report()
        self.assertIsNone(report["terminal_rmse_gain_pct_of_hard"])
        self.assertLess(report["best_observed_rmse_gain_m_s"], 0)
        with patch.object(study, "analyze", return_value=report), redirect_stdout(StringIO()) as output:
            self.assertEqual(study.cli(["fixture", "--soft-run", str(self.directories["soft"]),
                                        "--hard-run", str(directory), "--runs-dir", str(self.root / "reports")]), 0)
        self.assertIn("No observed soft RMSE gain", output.getvalue())
        self.assertIn("hard RMSE is zero", output.getvalue())
        self.assertEqual(len(list((self.root / "reports").glob("*.json"))), 1)


if __name__ == "__main__":
    unittest.main()
