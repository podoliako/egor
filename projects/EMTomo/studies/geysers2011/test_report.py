"""Small native production logs; reporting must never solve fields or relocate."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.EMTomo.studies.geysers2011 import report
from projects.EMTomo.studies.geysers2011.test_run_case import manifest_for, write_json
from projects.forward_modeling.experiments import save_inputs
from projects.forward_modeling.model import PointSet, VelocityGrid


def tree_bytes(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def study(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "projects/EMTomo"))
    from config import InversionConfig
    from instruments.likelihood import PickNoise
    from tomography import tomography_em as em
    from tomography.tomography_events import EventLog
    from tomography.tomography_logging import TomographyLogger
    from velocity_model import VelocityModel, block_average_slowness

    manifest = manifest_for(tmp_path, ROOT)
    root = Path(manifest["experiments_root"])
    source_id = manifest["experiment_id"]
    stations = PointSet(("S2", "S1"), [[250., 250., 0.], [1750., 1750., 0.]])
    # Fractional fine-grid coordinates catch a missing +0.5 or an axis swap.
    events = PointSet(("E2", "E1"), [[406.25, 562.5, 718.75], [1187.5, 1093.75, 1312.5]])
    source = np.full((4, 4, 4), 4000.)
    source[::2] = 6000.
    dataset = {"station_count": 2, "event_count": 2, "pair_count": 4,
               "start_utc": "2011-04-01T00:00:00Z", "end_utc_exclusive": "2011-04-21T00:00:00Z"}
    inputs = save_inputs(root, source_id, VelocityGrid(source, 500.), stations, events, dataset)
    reference = block_average_slowness(source, 500., (2, 2, 2), 1000.)
    config = InversionConfig(cell_size=1000., subdivision=8, n_cycles=2, candidate_mode="hard",
                             lambda_reg=.1, coverage_damping_power=0., n_workers=12,
                             initial_layer_boundaries_km=(.7,), initial_layer_velocities_m_s=(4200., 4600.))
    manifest["baseline_config"] = report.json.loads(json.dumps(asdict(config)))
    manifest["input_sha256"] = {name: hashlib.sha256((inputs / name).read_bytes()).hexdigest()
                                for name in manifest["input_sha256"]}
    manifest["dataset"] = dataset
    case = manifest["cases"][0]
    logger = TomographyLogger(base_dir=manifest["runs_dir"], run_name="fixture")

    def forbidden(*args, **kwargs):
        pytest.fail("report attempted a forward/time-field solve or JIT")

    monkeypatch.setattr(em, "station_travel_time_fields", forbidden)
    monkeypatch.setattr(em, "warm_up_jit", forbidden)

    def step(model, arrivals, station_locs, noise, cfg, *, iteration, logger):
        logs = []
        for event, xyz in enumerate(events.coordinates_m):
            log = EventLog(
                candidate_cells=np.asarray([[0, 0, 0]]), misfit_shape=(16, 16, 16),
                positions=np.asarray([xyz / 125. - .5]), weights=np.asarray([1.]),
                misfit=None, residuals=np.zeros((2, 2)), G_per_weight=None,
                ray_count_per_weight={}, ray_reached_per_weight={0: np.asarray([True, event == 0])},
            )
            logger.save_event_data(iteration, event, log)
            logs.append((event, log))
        logger.save_ray_status(iteration, logs, 2)
        return np.zeros(model.shape), np.ones(model.shape), np.ones(model.shape)

    monkeypatch.setattr(em, "make_tomography_step", step)
    initial = VelocityModel(np.broadcast_to([4200., 4600.], (2, 2, 2)), 1000.)
    em.run_em(config, initial, np.zeros((2, 2)), stations.coordinates_m, PickNoise(.01, .01, .02),
              reference_model=VelocityModel(reference, 1000.),
              source_experiment={"id": source_id, "model_sha256": manifest["input_sha256"]["model.npz"]},
              logger=logger)
    result_path = tmp_path / "cases" / case["id"] / "result.json"
    write_json(result_path, {"case_id": case["id"], "status": "complete", "run_dir": str(logger.run_dir)})
    # A historical run must use its own source, not the new geometry's truth.
    historical_id = "historical-synthetic"
    old_inputs = save_inputs(root, historical_id, VelocityGrid(np.full((2, 2, 2), 3000.), 1000.), stations, events)
    old = Path(manifest["reference_run"])
    meta = json.loads((logger.run_dir / "meta.json").read_text())
    meta["source_experiment"] = {"id": historical_id,
                                  "model_sha256": hashlib.sha256((old_inputs / "model.npz").read_bytes()).hexdigest()}
    write_json(old / "meta.json", meta)
    np.save(old / "initial_model.npy", np.full((2, 2, 2), 3100.))
    np.save(old / "final_model.npy", np.full((2, 2, 2), 3050.))
    path = tmp_path / "manifest.json"
    write_json(path, manifest)
    # Reporting should have no access to even the fake numerical step.
    monkeypatch.setattr(em, "make_tomography_step", forbidden)
    monkeypatch.setattr(em, "run_em", forbidden)
    return path, manifest, logger.run_dir, reference


def test_native_logs_complete_metrics_historical_truth_and_read_only(study, tmp_path, monkeypatch):
    path, manifest, run, reference = study
    before_runs = tree_bytes(Path(manifest["runs_dir"]))
    before_inputs = tree_bytes(Path(manifest["experiments_root"]))
    output = tmp_path / "report"
    with monkeypatch.context() as scoped:
        def no_scan(*args, **kwargs):
            pytest.fail("report recursively scanned per-hypothesis artifacts")
        scoped.setattr(Path, "rglob", no_scan)
        summary = report.generate_report(path, output)
    row = summary["cases"][0]
    expected = np.sqrt(np.mean((np.load(run / "initial_model.npy") - reference) ** 2))
    assert row["assessment_complete"]
    assert row["initial_rmse_m_s"] == pytest.approx(expected)
    assert row["final_rmse_m_s"] == pytest.approx(expected)
    assert [p["cycle"] for p in row["rmse_curve"]] == [0, 1, 2]
    assert row["best_cycle"] == 0
    assert row["hypocentres"]["event_count"] == 2
    assert row["hypocentres"]["median_m"] == pytest.approx(0.)
    assert row["hypocentres"]["p90_m"] == pytest.approx(0.)
    assert row["hypocentres"]["iteration"] == 1
    assert row["hypocentres"]["phase"] == report.HYPO_PHASE
    assert row["rays"]["failed_total"] == 2 and row["rays"]["counts_complete"]
    assert row["runtime"]["timed_cycles"] == 2
    old = summary["reference_run"]
    assert old["source_experiment_id"] == "historical-synthetic"
    assert old["initial_rmse_m_s"] == pytest.approx(100.)
    assert old["final_rmse_m_s"] == pytest.approx(50.)
    assert not old["assessment_complete"] and old["hypocentres"] is None
    assert tree_bytes(Path(manifest["runs_dir"])) == before_runs
    assert tree_bytes(Path(manifest["experiments_root"])) == before_inputs
    assert {p.name for p in output.iterdir()} == {"summary.json", "report.md", "geometry_start.csv", "rmse_curve.png"}
    assert json.loads((output / "summary.json").read_text()) == summary
    markdown = (output / "report.md").read_text()
    assert "not post-final relocation" in markdown and "not early stopping" in markdown
    assert "2011-04-21T00:00:00Z" in (output / "geometry_start.csv").read_text()


def test_manifest_schema_does_not_require_dataset(study, tmp_path):
    path, manifest, _, _ = study
    del manifest["dataset"]
    write_json(path, manifest)
    summary = report.generate_report(path, tmp_path / "report")
    assert summary["dataset"]["station_count"] == 2
    assert summary["dataset"]["event_count"] == 2
    assert summary["cases"][0]["assessment_complete"]


def test_all_nine_queue_states_and_partial_json_are_tolerated(study, tmp_path):
    path, manifest, run, _ = study
    for index, status in ((1, "running"), (2, "failed"), (3, "running")):
        case = manifest["cases"][index]
        write_json(path.parent / "cases" / case["id"] / "result.json",
                   {"status": status, "run_dir": str(run) if index != 3 else None, "error": "fixture"})
    # A queue reader may catch a truncated result/meta or last JSONL line.
    malformed = path.parent / "cases" / manifest["cases"][4]["id"] / "result.json"
    malformed.parent.mkdir(parents=True)
    malformed.write_text('{"status":')
    with (run / "quality.jsonl").open("a") as stream:
        stream.write('{"iter":2,')
    (run / "iter_1" / "complete.json").unlink()
    (run / "iter_0" / "ray_status.json").unlink()
    summary = report.generate_report(path, tmp_path / "report")
    assert len(summary["cases"]) == 9
    assert [r["status"] for r in summary["cases"][:4]] == ["complete", "running", "failed", "running"]
    row = summary["cases"][0]
    assert row["completed_iterations"] == [0]
    assert [p["cycle"] for p in row["rmse_curve"]] == [0, 1]
    assert row["hypocentres"] is None and not row["assessment_complete"]
    assert row["rays"]["failed_total"] is None  # unavailable is not zero
    assert any("incomplete/invalid" in warning for warning in row["warnings"])
    assert summary["cases"][4]["warnings"]
    assert all(r["status"] == "pending" for r in summary["cases"][5:])


@pytest.mark.parametrize("artifact", ["weights", "initial", "final", "quality", "meta", "params", "subdivision"])
def test_corrupt_artifacts_degrade_assessment_not_report(study, tmp_path, artifact):
    path, _, run, _ = study
    if artifact == "weights":
        np.savez(run / "iter_1/event_0/weights.npz", positions=[[0., 0., 0.]], weight_values=[np.nan])
    elif artifact in ("initial", "final"):
        np.save(run / f"{artifact}_model.npy", np.ones((1, 1, 1)))
    elif artifact == "quality":
        (run / "quality.jsonl").write_text('null\n[]\n{"iter":0,"rms_m_s":true}\n')
    else:
        meta = json.loads((run / "meta.json").read_text())
        if artifact == "meta":
            meta["source_experiment"] = ["not an object"]
        elif artifact == "params":
            meta["run_params"] = []
        else:
            meta["run_params"]["subdivision"] = 0
        write_json(run / "meta.json", meta)
    summary = report.generate_report(path, tmp_path / "report")
    assert not summary["cases"][0]["assessment_complete"]
    assert summary["cases"][0]["warnings"]
    json.dumps(summary, allow_nan=False)


def test_jsonl_ignores_nonobject_and_truncated_lines(tmp_path):
    path = tmp_path / "quality.jsonl"
    path.write_text('null\n[]\n{"iter":false}\n{"iter":0,"rms_m_s":3}\n{"iter":')
    warnings = []
    assert report._jsonl(path, warnings) == {0: {"iter": 0, "rms_m_s": 3}}
    assert len(warnings) == 4


def test_truth_uses_slowness_average_and_caches_inputs(study, monkeypatch):
    _, manifest, _, reference = study
    calls = []
    original = report._load_inputs
    monkeypatch.setattr(report, "_load_inputs", lambda *args: (calls.append(args), original(*args))[1])
    truth = report.Truth(manifest["experiments_root"])
    grid = truth.grid(manifest["experiment_id"], (2, 2, 2), 1000.)
    np.testing.assert_allclose(grid, reference)
    np.testing.assert_allclose(grid, 4800.)  # harmonic mean of 4000 and 6000, not 5000
    assert truth.grid(manifest["experiment_id"], (2, 2, 2), 1000.) is grid
    assert len(calls) == 1


def test_cli_and_optional_plot_failure(study, tmp_path, monkeypatch):
    path, _, _, _ = study
    def unavailable(*args):
        raise ImportError("matplotlib unavailable")
    monkeypatch.setattr(report, "_plot", unavailable)
    output = tmp_path / "report"
    assert report.main(["--manifest", str(path), "--output-dir", str(output)]) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["cases"][0]["assessment_complete"]
    assert any("plot unavailable" in w for w in summary["warnings"])
