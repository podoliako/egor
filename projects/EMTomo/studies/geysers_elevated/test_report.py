"""Fixture/mock-shared-report tests: no production imports or real solves."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from projects.EMTomo.studies.geysers_elevated import report


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def tree_bytes(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()}


@pytest.fixture
def study(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    state = tmp_path / "state"
    dataset = {"station_count": 2, "event_count": 2, "pair_count": 4,
               "start_utc": "2011-04-01T00:00:00Z", "end_utc_exclusive": "2011-04-21T00:00:00Z"}
    baseline = {"cell_size": 1000., "subdivision": 8, "n_cycles": 2,
                "initial_layer_boundaries_km": [0.7], "initial_layer_velocities_m_s": [4200., 4600.],
                "candidate_mode": "hard", "n_candidates": 15, "weights_top_n": 1,
                "temperature": 1., "lambda_reg": 1., "coverage_damping_power": 2.}
    cases = [{"id": f"{mode}-k{k}", "run_name": f"elev-{mode}-{k}", "is_reference": i == 0,
              "lambda_reg": 1., "coverage_damping_power": 2., "candidate_mode": mode,
              "n_candidates": k, "weights_top_n": 1 if mode == "hard" else k,
              "temperature": 1. if mode == "hard" else None}
             for i, (mode, k) in enumerate([(m, k) for m, ks in
                 (("hard", (1, 5, 10, 15)), ("soft", (5, 10, 15))) for k in ks])]
    manifest = {"schema_version": 1, "experiment_id": "elevated-fixture", "dataset": dataset,
                "baseline_config": baseline, "experiments_root": str(tmp_path / "experiments"),
                "runs_dir": str(tmp_path / "runs"), "reference_run": str(tmp_path / "historical"),
                "runtime_root": str(runtime), "cases": cases}
    calibration = {"temperature": 4.25, "selection": {"reason": "upper_bound_saturated", "saturated": True},
                   "per_k": {str(k): {"at_temperature": {
                       "ess": {"p10": 1., "median": min(k, 3.), "p90": k},
                       "unique_location_ess": {"median": 1.5 if k > 1 else 1.}}}
                             for k in (1, 5, 10, 15)},
                   "warnings": ["fixture saturation warning", "label ESS is not spatial diversity"]}
    path = state / "manifest.json"
    write_json(path, manifest)
    write_json(state / "calibration.json", calibration)
    stations = SimpleNamespace(ids=("S2", "S1"), coordinates_m=[[250., 250., 0.], [1750., 1750., 416.]])
    events = SimpleNamespace(ids=("E2", "E1"))
    calls = SimpleNamespace(analyze=[], source=[], atomic=[], imports=[])

    class Truth:
        def __init__(self, root):
            assert root == manifest["experiments_root"]

        def source(self, source_id):
            calls.source.append(source_id)
            return None, stations, events

    def row(case):
        return {**copy.deepcopy(case), "status": "pending", "run_dir": None, "warnings": [],
                "completed_iterations": [], "completed_cycles": 0, "rmse_curve": [],
                "initial_rmse_m_s": None, "final_rmse_m_s": None, "best_cycle": None,
                "min_rmse_m_s": None, "hypocentres": None, "assessment_complete": False,
                "runtime": {"mean_cycle_s": None, "timed_cycles": 0},
                "rays": {"failed_total": None, "available_cycles": 0, "counts_complete": False}}

    def analyze(row, run, selected, truth, *, historical=False):
        assert historical  # Deliberately bypass only the obsolete flat-z branch.
        assert selected["experiment_id"] == manifest["experiment_id"]
        calls.analyze.append((row["id"], run))
        source_id = report._object(run / "meta.json")["source_experiment"]["id"]
        truth.source(source_id)
        row.update(source_experiment_id=source_id, expected_cycles=2, completed_cycles=1,
                   completed_iterations=[0], initial_rmse_m_s=100., best_cycle=1, min_rmse_m_s=80.,
                   rmse_curve=[{"cycle": 0, "rmse_m_s": 100.}, {"cycle": 1, "rmse_m_s": 80.}])
        if row["status"] == "complete":
            row.update(final_rmse_m_s=90., assessment_complete=True,
                       hypocentres={"median_m": 20., "p90_m": 40.})

    def atomic(path, content):
        calls.atomic.append(path.name)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)

    def plot(rows, output):
        raise ImportError("fixture: plotting unavailable")

    def dataset_for(selected):
        return {key: selected.get("dataset", dataset)[key] for key in dataset}

    shared = SimpleNamespace(_row=row, _analyze_run=analyze, Truth=Truth, _dataset=dataset_for,
                             _atomic=atomic, _plot=plot,
                             HYPO_PHASE="last logged E-step; not post-final relocation")

    def import_module(name):
        calls.imports.append(name)
        assert sys.path[:3] == [str(runtime / "projects/EMTomo"), str(runtime), str(runtime / "runner")]
        assert name == "_shared_report"
        return shared

    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(report.importlib, "import_module", import_module)
    runs = []
    for case in cases:
        run = Path(manifest["runs_dir"]) / case["run_name"]
        params = {**baseline, **{key: case[key] for key in report._SETTINGS + report._REGULARIZATION}}
        params["temperature"] = 1. if case["candidate_mode"] == "hard" else calibration["temperature"]
        meta = {"run_params": params, "station_locs": stations.coordinates_m,
                "source_experiment": {"id": manifest["experiment_id"]}}
        write_json(run / "meta.json", meta)
        result = {"case_id": case["id"], "status": "complete", "run_dir": str(run),
                  "provenance": {"runtime_root": str(runtime), "manifest_sha256": "fixture"}}
        write_json(state / "cases" / case["id"] / "result.json", result)
        runs.append(run)
    historical = {**baseline, "candidate_mode": "hard", "n_candidates": 10,
                  "weights_top_n": 1, "temperature": 1., "lambda_reg": 0.5, "coverage_damping_power": 1.}
    write_json(Path(manifest["reference_run"]) / "meta.json",
               {"run_params": historical, "source_experiment": {"id": "own-historical-source"}})
    return SimpleNamespace(path=path, manifest=manifest, calibration=calibration, runs=runs,
                           calls=calls, shared=shared, state=state, historical=historical,
                           stations=stations, output=tmp_path / "output")


def test_seven_modes_resolved_temperature_historical_config_and_read_only(study):
    s = study
    before = tree_bytes(s.state), tree_bytes(Path(s.manifest["runs_dir"])), tree_bytes(Path(s.manifest["reference_run"]))
    summary = report.generate_report(s.path, s.output)
    assert len(summary["cases"]) == 7
    for row in summary["cases"]:
        assert row["assessment_complete"]
        assert row["temperature"] == (1. if row["candidate_mode"] == "hard" else 4.25)
        assert row["weights_top_n"] == (1 if row["candidate_mode"] == "hard" else row["n_candidates"])
        assert row["provenance"] == row["result"]["provenance"]
        assert row["final_rmse_m_s"] == 90. and row["min_rmse_m_s"] == 80.
    old = summary["reference_run"]
    assert old["configuration"] == s.historical
    assert old["lambda_reg"] == 0.5 and old["coverage_damping_power"] == 1.
    assert old["n_candidates"] == 10 and old["source_experiment_id"] == "own-historical-source"
    assert s.calls.imports == ["_shared_report"] and len(s.calls.analyze) == 8
    assert "own-historical-source" in s.calls.source
    assert summary == json.loads((s.output / "summary.json").read_text())
    json.dumps(summary, allow_nan=False)
    markdown = (s.output / "report.md").read_text()
    assert markdown.startswith("# Elevated Geysers — hard/soft candidate comparison")
    assert "| case | mode | K | top_n | T | status | final RMSE" in markdown
    assert "| soft-k15 | soft | 15 | 15 | 4.25 | complete | 90 | 80 @1 |" in markdown
    assert "not a new damping sweep" in markdown and "Latest historical configuration" in markdown
    assert "mode=hard, K=10, top_n=1, T=1" in markdown
    assert "fixture saturation warning" in markdown and "median unique-location ESS" in markdown
    assert "not verified" in markdown and "Stations remain at flat z=0" not in markdown
    assert "not early stopping" in markdown and "not post-final relocation" in markdown
    ledger = (s.output / "geometry_start.csv").read_text()
    assert "2011-04-21T00:00:00Z" in ledger and "baseline_config_json" in ledger
    assert set(s.calls.atomic) == {"summary.json", "report.md", "geometry_start.csv"}
    assert any("plot unavailable" in w for w in summary["warnings"])
    assert before == (tree_bytes(s.state), tree_bytes(Path(s.manifest["runs_dir"])),
                      tree_bytes(Path(s.manifest["reference_run"])))


def test_partial_failed_pending_truncated_and_missing_meta(study):
    s = study
    results = [s.state / "cases" / case["id"] / "result.json" for case in s.manifest["cases"]]
    write_json(results[0], {"case_id": "hard-k1", "status": "running", "run_dir": str(s.runs[0])})
    write_json(results[1], {"status": "failed", "run_dir": str(s.runs[1]), "error": "fixture failure"})
    write_json(results[2], {"status": "running", "run_dir": None})
    results[3].write_text('{"status":')
    results[4].unlink()
    (s.runs[5] / "meta.json").unlink()
    summary = report.generate_report(s.path, s.output)
    rows = summary["cases"]
    assert [r["status"] for r in rows] == ["running", "failed", "running", "pending", "pending", "complete", "complete"]
    assert rows[0]["min_rmse_m_s"] == 80. and rows[0]["final_rmse_m_s"] is None
    assert not rows[0]["assessment_complete"]
    assert rows[1]["error"] == "fixture failure" and rows[1]["final_rmse_m_s"] is None
    assert rows[2]["warnings"] and rows[3]["warnings"] and rows[5]["warnings"]
    assert not rows[4]["warnings"] and rows[4]["temperature"] == 4.25
    assert [r[0] for r in s.calls.analyze] == ["hard-k1", "soft-k15", "historical-reference"]
    assert "fixture failure" in (s.output / "report.md").read_text()


@pytest.mark.parametrize("contents", [None, '{"temperature":', '[]',
                                      '{"temperature":NaN}', '{"temperature":true}',
                                      '{"temperature":2,"selection":[],"per_k":{}}'])
def test_missing_or_invalid_calibration_never_invents_soft_temperature(study, contents):
    s = study
    path = s.state / "calibration.json"
    if contents is None:
        path.unlink()
    else:
        path.write_text(contents)
    (s.state / "cases/soft-k15/result.json").unlink()
    summary = report.generate_report(s.path, s.output)
    assert summary["calibration"]["temperature"] is None
    assert summary["calibration"]["status"] == "unavailable"
    assert summary["warnings"]
    assert summary["cases"][0]["temperature"] == 1.
    assert summary["cases"][4]["temperature"] == 4.25
    assert summary["cases"][4]["temperature_source"] == "run_meta"
    assert summary["cases"][6]["temperature"] is None
    assert "| soft-k15 | soft | 15 | 15 | — | pending | — | — @— |" in (s.output / "report.md").read_text()


@pytest.mark.parametrize("problem", ["mode", "k", "top_n", "temperature", "lambda", "baseline",
                                    "source", "geometry", "effective", "case_id", "relative_run"])
def test_config_identity_or_geometry_mismatch_is_not_assessed(study, problem):
    s = study
    meta_path = s.runs[0] / "meta.json"
    meta = report._object(meta_path)
    result_path = s.state / "cases/hard-k1/result.json"
    result = report._object(result_path)
    if problem in ("mode", "k", "top_n", "temperature", "lambda", "baseline"):
        key, value = {"mode": ("candidate_mode", "soft"), "k": ("n_candidates", 15),
                      "top_n": ("weights_top_n", 5), "temperature": ("temperature", 2.),
                      "lambda": ("lambda_reg", 0.5), "baseline": ("subdivision", 4)}[problem]
        meta["run_params"][key] = value
    elif problem == "source":
        meta["source_experiment"]["id"] = "wrong-truth"
    elif problem == "geometry":
        meta["station_locs"] = list(reversed(meta["station_locs"]))
    elif problem == "effective":
        result["effective_config"] = {**meta["run_params"], "temperature": 2.}
    elif problem == "case_id":
        result["case_id"] = "another-case"
    else:
        result["run_dir"] = "relative/run"
    write_json(meta_path, meta)
    write_json(result_path, result)
    summary = report.generate_report(s.path, s.output)
    row = summary["cases"][0]
    assert not row["assessment_complete"] and row["final_rmse_m_s"] is None
    assert row["warnings"] and "hard-k1" not in [r[0] for r in s.calls.analyze]


def test_optional_result_effective_config_matches_resolved_soft_config(study):
    s = study
    path = s.state / "cases/soft-k5/result.json"
    result = report._object(path)
    result["effective_config"] = report._object(s.runs[4] / "meta.json")["run_params"]
    write_json(path, result)
    summary = report.generate_report(s.path, s.output)
    assert summary["cases"][4]["assessment_complete"]
    assert summary["cases"][4]["effective_config"] == result["effective_config"]


def test_missing_historical_artifact_and_dataset_are_tolerated(study):
    s = study
    del s.manifest["dataset"]
    write_json(s.path, s.manifest)
    Path(s.manifest["reference_run"], "meta.json").write_text('{"run_params":')
    summary = report.generate_report(s.path, s.output)
    assert summary["dataset"]["event_count"] == 2
    assert summary["reference_run"]["warnings"]
    assert summary["reference_run"]["temperature"] is None
    assert summary["cases"][0]["assessment_complete"]


@pytest.mark.parametrize("field,value", [("temperature", "corrupt"), ("candidate_mode", "invalid"),
                                        ("weights_top_n", 10), ("lambda_reg", 0.5)])
def test_invalid_case_config_degrades_without_breaking_other_rows(study, field, value):
    s = study
    s.manifest["cases"][0][field] = value
    write_json(s.path, s.manifest)
    summary = report.generate_report(s.path, s.output)
    assert not summary["cases"][0]["assessment_complete"]
    assert summary["cases"][0]["warnings"]
    assert summary["cases"][1]["assessment_complete"]


def test_partial_ess_diagnostics_keep_temperature_and_warnings(study):
    s = study
    s.calibration["per_k"]["5"] = {"at_temperature": {"ess": {"median": "corrupt"}}}
    s.calibration["per_k"]["10"] = None
    write_json(s.state / "calibration.json", s.calibration)
    summary = report.generate_report(s.path, s.output)
    assert summary["calibration"]["temperature"] == 4.25
    markdown = (s.output / "report.md").read_text()
    assert "| 5 | — | — |" in markdown and "| 10 | — | — |" in markdown
    assert "fixture saturation warning" in markdown


def test_saved_run_temperature_must_match_optional_effective_config_without_calibration(study):
    s = study
    (s.state / "calibration.json").unlink()
    path = s.state / "cases/soft-k5/result.json"
    result = report._object(path)
    result["effective_config"] = {**report._object(s.runs[4] / "meta.json")["run_params"], "temperature": 2.}
    write_json(path, result)
    summary = report.generate_report(s.path, s.output)
    assert not summary["cases"][4]["assessment_complete"]
    assert any("temperature differs" in warning for warning in summary["cases"][4]["warnings"])


def test_cli_default_output_and_no_reference(study):
    s = study
    del s.manifest["reference_run"]
    write_json(s.path, s.manifest)
    assert report.main(["--manifest", str(s.path)]) == 0
    summary = report._object(s.state / "summary.json")
    assert "reference_run" not in summary
    assert any("no reference_run" in warning for warning in summary["warnings"])
    assert report.main(["--manifest", str(s.path), "--output-dir", str(s.output)]) == 0


def test_cli_invalid_manifest_exits_two(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("null")
    with pytest.raises(SystemExit) as error:
        report.main(["--manifest", str(path)])
    assert error.value.code == 2


def test_live_loader_fallback_and_dependency_import_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "path", list(sys.path))
    seen = []
    shared = object()

    def live(name):
        seen.append(name)
        if name == "_shared_report":
            raise ModuleNotFoundError(name="_shared_report")
        assert name == "projects.EMTomo.studies.geysers2011.report"
        return shared

    monkeypatch.setattr(report.importlib, "import_module", live)
    assert report._load_shared({"runtime_root": str(tmp_path)}) is shared
    assert len(seen) == 2

    def broken(name):
        raise ModuleNotFoundError(name="missing_shared_dependency")

    monkeypatch.setattr(report.importlib, "import_module", broken)
    with pytest.raises(ModuleNotFoundError) as error:
        report._load_shared({"runtime_root": str(tmp_path)})
    assert error.value.name == "missing_shared_dependency"
    with pytest.raises(ValueError, match="absolute"):
        report._load_shared({"runtime_root": "relative"})


def test_import_and_frozen_loader_are_snapshot_safe(tmp_path):
    runner = tmp_path / "runtime/runner"
    runner.mkdir(parents=True)
    shutil.copyfile(report.__file__, runner / "report.py")
    (runner / "_shared_report.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "root = Path(__file__).resolve().parents[1]\n"
        "assert sys.path[:2] == [str(root / 'projects/EMTomo'), str(root)]\n"
        "FROZEN = True\n"
    )
    script = (
        "import importlib.util, sys; "
        f"spec=importlib.util.spec_from_file_location('frozen_report', {str(runner / 'report.py')!r}); "
        "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
        "assert not any(name in sys.modules for name in "
        "('numpy', '_shared_report', 'config', 'eikonal', 'tomography', 'experiment_runner')); "
        f"assert module._load_shared({{'runtime_root': {str(runner.parent)!r}}}).FROZEN"
    )
    subprocess.run([sys.executable, "-S", "-c", script], check=True, timeout=10,
                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
