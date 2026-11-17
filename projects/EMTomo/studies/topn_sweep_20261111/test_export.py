"""Small artifact fixtures: no inversion engine or production outputs required."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location("topn_export", Path(__file__).with_name("export.py"))
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.study = self.project / "studies/topn"
        self.study.mkdir(parents=True)
        self.data = self.project / "paper/exp_data"
        self.data.mkdir(parents=True)
        self.params = dict(n_candidates=50, weights_top_n=50, n_events=1, n_cycles=3,
                           temperature=1., lambda_reg=1., weights_min_distance=7)
        cache = {"runs": {}}
        for run_id in ("old_h", "old_s"):
            self.make_run(run_id)
            cache["runs"][run_id] = export.residual.extract_run(
                self.project / "runs" / run_id,
                dict(run_id=run_id, run_dir=run_id, n_events=1))
        write_json(self.data / "residual_metrics_cache.json", cache)
        self.old = dict(pair_id="old_pair", section="2", experiment_id="experiment",
                        study_manifest="", hard_control_reuse_count="1", n_candidates="50",
                        weights_top_n="50", hard_run_id="old_h", soft_run_id="old_s",
                        soft_gain_percent="1.234000", metric="avg_abs_pct_dev",
                        selection_uses_truth="False", evaluation_uses_truth="True")
        for mode in ("hard", "soft"):
            for key in export.METRICS:
                self.old[f"{mode}_{key}"] = "old_value"
            self.old[f"{mode}_temperature"] = "1.0"
            self.old[f"{mode}_temperature_schedule"] = "null"
        for key in ("lambda_reg", "weights_min_distance", "n_events", "requested_updates"):
            self.old[key] = "1"
        with (self.data / "hard_soft_lcurve_avg_abs_pct_dev.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.old))
            writer.writeheader()
            writer.writerow(self.old)
        self.manifest = dict(cases={"soft": dict(config={**self.params, "n_candidates": 100,
                              "weights_top_n": 10}, experiment_id="experiment", source_run_id="old_s")},
                             pairs=[dict(pair_id="new_pair", source_pair_id="old_pair",
                                         hard_case_id=None, hard_run_id="old_h", soft_case_id="soft")])
        self.save_manifest()

    def save_manifest(self):
        write_json(self.study / "manifest.json", self.manifest)

    def make_run(self, run_id):
        directory = self.project / "runs" / run_id
        write_json(directory / "meta.json", dict(run_params=self.params))
        np.save(directory / "initial_model.npy", np.array([5000.]))
        for k in range(3):
            iteration = directory / f"iter_{k}"
            write_json(iteration / "complete.json", {})
            (iteration / "event_0").mkdir()
            np.save(iteration / "event_0/residuals.npy", np.array([[0., .1 / (k + 1)], [0., 0.]]))
            np.save(iteration / "model.npy", np.array([5000. + k * 100.]))
        (directory / "quality.jsonl").write_text('\n'.join(json.dumps(dict(iter=k, avg_abs_pct_dev=5. - k)) for k in range(3)))

    def run_export(self):
        return export.export_study(self.study, self.project)

    def test_pending_blanks_all_metrics_and_preserves_old_values(self):
        source = self.data / "hard_soft_lcurve_avg_abs_pct_dev.csv"
        before = source.read_bytes()
        with patch.object(export.residual, "extract_run", side_effect=AssertionError("no extraction")), \
             patch.object(export.lcurve, "selected_quality", side_effect=AssertionError("no truth")):
            rows = self.run_export()
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual({k: rows[0][k] for k in self.old}, self.old)
        self.assertEqual(rows[1]["pair_status"], "pending")
        self.assertEqual(rows[1]["soft_run_id"], "")
        for mode in ("hard", "soft"):
            for key in export.METRICS:
                self.assertEqual(rows[1][f"{mode}_{key}"], "")
        self.assertEqual(rows[1]["soft_gain_percent"], "")
        self.assertEqual([rows[1][k] for k in export.EXTRA_FIELDS[:4]], [50, 100, 50, 10])
        self.assertEqual(rows[1]["n_candidates"], 100)
        self.assertEqual(rows[1]["weights_top_n"], 10)

    def test_running_failed_and_refresh(self):
        for status in ("running", "failed", "running"):
            write_json(self.study / "cases/soft/result.json", dict(status=status, run_id="new_s"))
            row = self.run_export()[1]
            self.assertEqual(row["pair_status"], status)
            self.assertEqual(row["soft_run_id"], "new_s")
            self.assertEqual(row["soft_avg_abs_pct_dev_at_lcurve"], "")

    def test_completed_extracts_only_new_freezes_before_truth_and_reuses_cache(self):
        self.make_run("new_s")
        write_json(self.study / "cases/soft/result.json", dict(status="completed", run_id="new_s"))
        original = export.lcurve.selected_quality

        def quality(directory, selection):
            frozen = export.read(self.study / "lcurve_selection.json")
            self.assertEqual(set(frozen["runs"]), {"old_h", "new_s"})
            return original(directory, selection)

        old_cache = (self.data / "residual_metrics_cache.json").read_bytes()
        with patch.object(export.residual, "extract_run", wraps=export.residual.extract_run) as extract, \
             patch.object(export.lcurve, "selected_quality", side_effect=quality):
            row = self.run_export()[1]
            self.assertEqual(extract.call_count, 1)
            self.assertEqual(extract.call_args.args[0].name, "new_s")
            self.assertEqual(row["pair_status"], "completed")
            self.assertEqual(row["soft_selected_update_number"], 2)
            self.assertEqual(row["soft_quality_iteration_index"], 1)
            self.assertEqual(row["soft_avg_abs_pct_dev_at_lcurve"], 4.)
            self.assertEqual(row["soft_gain_percent"], 0.)
            self.run_export()
            self.assertEqual(extract.call_count, 1)
        self.assertEqual((self.data / "residual_metrics_cache.json").read_bytes(), old_cache)
        self.assertEqual(set(export.read(self.study / "residual_metrics_cache.json")["runs"]), {"new_s"})

    def test_new_hard_case_and_bad_quality_blank_entire_pair(self):
        self.manifest["cases"]["hard"] = dict(config={**self.params, "weights_top_n": 1},
                                             experiment_id="experiment", source_run_id="old_h")
        self.manifest["pairs"][0].update(hard_case_id="hard", hard_run_id=None)
        self.save_manifest()
        for mode in ("hard", "soft"):
            self.make_run(mode)
            write_json(self.study / f"cases/{mode}/result.json", dict(status="completed", run_id=mode))
        (self.project / "runs/soft/quality.jsonl").write_text("")
        row = self.run_export()[1]
        self.assertEqual(row["hard_weights_top_n"], 1)
        self.assertEqual(row["pair_status"], "error")
        self.assertEqual(row["hard_avg_abs_pct_dev_at_lcurve"], "")
        self.assertIn("soft", export.read(self.study / "export_provenance.json")["errors"])

    def test_expanded_38_plus_14_preserves_every_original_cell(self):
        originals = [{**self.old, "pair_id": f"old_{i}", "soft_gain_percent": f"{i}.00100"}
                     for i in range(38)]
        source = self.data / "hard_soft_lcurve_avg_abs_pct_dev.csv"
        with source.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.old))
            writer.writeheader()
            writer.writerows(originals)
        self.manifest["pairs"] = [{**self.manifest["pairs"][0], "pair_id": f"new_{i}",
                                   "source_pair_id": f"old_{i}"} for i in range(14)]
        self.save_manifest()
        self.run_export()
        target = self.data / "hard_soft_lcurve_avg_abs_pct_dev_topn_sweep.csv"
        with target.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 52)
        self.assertEqual([{k: row[k] for k in self.old} for row in rows[:38]], originals)
        self.assertTrue(all(row["pair_status"] == "pending" for row in rows[38:]))

    def test_missing_historical_cache_does_not_reextract_old_residuals(self):
        self.make_run("new_s")
        write_json(self.study / "cases/soft/result.json", dict(status="completed", run_id="new_s"))
        write_json(self.data / "residual_metrics_cache.json", {"runs": {}})
        with patch.object(export.residual, "extract_run", wraps=export.residual.extract_run) as extract:
            row = self.run_export()[1]
        self.assertEqual(extract.call_count, 1)
        self.assertEqual(extract.call_args.args[0].name, "new_s")
        self.assertEqual(row["pair_status"], "error")
        self.assertEqual(row["soft_gain_percent"], "")

    def test_refuses_historical_overwrite(self):
        with self.assertRaisesRegex(ValueError, "historical CSV"):
            export.export_study(self.study, self.project, self.data / "hard_soft_lcurve_avg_abs_pct_dev.csv")


if __name__ == "__main__":
    unittest.main()
