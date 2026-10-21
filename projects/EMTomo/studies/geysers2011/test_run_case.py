"""Production wiring tests: real preparation/logging, no forward, JIT or ray solves."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from dataclasses import asdict

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.EMTomo.config import InversionConfig
from projects.EMTomo.studies.geysers2011 import run_case
from projects.forward_modeling.experiments import save_inputs
from projects.forward_modeling.model import PointSet, VelocityGrid


CASES = [
    {"id": f"lam-{lam}-cov-{cov}", "lambda_reg": lam, "coverage_damping_power": cov,
     "run_name": f"geysers-lam-{lam}-cov-{cov}", "is_reference": lam == .5 and cov == 1.}
    for lam in (.1, .5, 1.) for cov in (0., 1., 2.)
]


def baseline_config():
    config = InversionConfig(
        cell_size=1000., subdivision=8, n_cycles=12, candidate_mode="hard",
        initial_layer_boundaries_km=(.7, 1.9, 3.4),
        initial_layer_velocities_m_s=(3800., 4600., 5200., 5700.),
        lambda_reg=.5, coverage_damping_power=1., max_velocity_step_fraction=.02,
        weight_noise_relative_sigma=.01, weight_noise_absolute_sigma_s=.01,
        weight_model_sigma_s=.02, n_workers=12,
    )
    return run_case._json({k: v for k, v in asdict(config).items()
                           if k not in {"runs_dir", "save_runs"}})


def manifest_for(tmp_path, runtime=None):
    return {
        "schema_version": 1, "experiment_id": run_case.EXPERIMENT_ID,
        "forward_implementation": run_case.FORWARD_IMPLEMENTATION,
        "created_utc": "2026-10-17T00:00:00Z",
        "experiments_root": str(tmp_path / "experiments"), "runs_dir": str(tmp_path / "runs"),
        "runtime_root": str(runtime or tmp_path / "runtime"),
        "reference_run": str(tmp_path / "reference"),
        "resources": dict(run_case.RESOURCE_VALUES), "baseline_config": baseline_config(),
        "input_sha256": {name: "0" * 64 for name in run_case.INPUT_FILES},
        "code_sha256": {"projects/EMTomo/eikonal.py": "0" * 64},
        "cases": copy.deepcopy(CASES),
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


@pytest.fixture(scope="module")
def runtime_snapshot(tmp_path_factory):
    runtime = tmp_path_factory.mktemp("geysers-runtime")
    for relative in ("projects/EMTomo", "projects/EMTomo/instruments",
                     "projects/EMTomo/tomography", "projects/forward_modeling"):
        destination = runtime / relative
        destination.mkdir(parents=True, exist_ok=True)
        for source in (ROOT / relative).glob("*.py"):
            shutil.copy2(source, destination / source.name)
    legacy = subprocess.run(
        ["git", "--no-pager", "show", "c381cfd:projects/EMTomo/eikonal.py"],
        cwd=ROOT, capture_output=True, check=True, timeout=10,
    ).stdout
    (runtime / "projects/EMTomo/eikonal.py").write_bytes(legacy)
    (runtime / "runner").mkdir()
    shutil.copy2(run_case.__file__, runtime / "runner/run_case.py")
    shutil.copy2(Path(run_case.__file__).with_name("report.py"), runtime / "runner/report.py")
    return runtime


@pytest.fixture
def saved_study(tmp_path, runtime_snapshot):
    """Full pair count but a tiny coarse truth; observations are hand-written fixtures."""
    manifest = manifest_for(tmp_path, runtime_snapshot)
    root = Path(manifest["experiments_root"])
    station_xyz = np.column_stack((np.linspace(100., 19900., 26), np.full(26, 3500.), np.zeros(26)))
    event_xyz = np.tile([9000., 3000., 2000.], (3780, 1))
    stations = PointSet(tuple(f"s{i:02}" for i in range(26)), station_xyz)
    events = PointSet(tuple(f"e{i:04}" for i in range(3780)), event_xyz)
    generation = {"start_utc": "2011-04-01T00:00:00Z", "end_utc_exclusive": "2011-04-21T00:00:00Z",
                  "station_count": 26, "event_count": 3780, "pair_count": 98280}
    directory = save_inputs(root, manifest["experiment_id"],
                            VelocityGrid(np.full((20, 7, 7), 5000.), 1000.), stations, events, generation)
    manifest["input_sha256"] = {name: run_case.sha256(directory / name) for name in run_case.INPUT_FILES}
    manifest["code_sha256"] = {path.relative_to(runtime_snapshot).as_posix(): run_case.sha256(path)
                               for path in runtime_snapshot.rglob("*.py")}
    output = root / "output" / manifest["experiment_id"]
    output.mkdir(parents=True)
    with (output / "arrivals.csv").open("w") as stream:
        stream.write("station_id,event_id,arrival_time_s\n")
        for event in events.ids:
            stream.writelines(f"{station},{event},{i / 100}\n" for i, station in enumerate(stations.ids))
    write_json(output / "metadata.json", {
        "time_reference": "earliest_station_arrival_per_event",
        "units": {"coordinates": "m", "cell_size": "m", "velocity": "m/s", "arrival_time": "s", "elapsed_time": "s"},
        "input_sha256": manifest["input_sha256"],
        "noise": {"enabled": True, "model": "independent_gaussian_relative_plus_absolute",
                  "config": {"relative_sigma": .01, "absolute_sigma_s": .01, "seed": 42}},
    })
    reference = Path(manifest["reference_run"])
    write_json(reference / "meta.json", {"run_params": {
        **manifest["baseline_config"], "travel_times": run_case.SAVED_TRAVEL_TIMES}})
    profile = np.asarray([3800., 4600., 5200., 5700., 5700., 5700., 5700.])
    np.save(reference / "initial_model.npy", np.broadcast_to(profile, (20, 7, 7)))
    for case in manifest["cases"]:
        (tmp_path / "cases" / case["id"]).mkdir(parents=True)
    path = tmp_path / "manifest.json"
    write_json(path, manifest)
    return path, manifest


# Child process keeps bare production module names and BLAS settings out of pytest.
# Only the expensive numerical boundary is stubbed; run_em and the logger are real.
RUN_CHILD = r'''
import json, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import run_case as study
manifest_path = Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text())
study.verify_hashes(manifest)
assert 'numpy' not in sys.modules, 'numerical import occurred before runtime selection'
runtime = study.load_runtime(manifest)
import eikonal
assert Path(eikonal.__file__).resolve().is_relative_to(Path(manifest['runtime_root']))
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    assert os.environ[key] == '12'
old_description = runtime.em.METHOD_DESCRIPTION
np = runtime.np
calls = []
def forbidden(*args, **kwargs):
    raise AssertionError('unexpected heavy calculation')
eikonal.station_travel_time_fields = forbidden
runtime.em.station_travel_time_fields = forbidden
runtime.em.warm_up_jit = lambda: None
def step(model, arrivals, stations, noise, config, *, iteration, logger):
    assert arrivals.shape == (3780, 26)
    assert stations.shape == (26, 3) and np.all(stations[:, 2] == 0)
    assert config.n_workers == 12
    assert runtime.em.METHOD_DESCRIPTION['travel_times'] == study.SAVED_TRAVEL_TIMES
    calls.append(iteration)
    return np.zeros(model.shape), np.ones(model.shape), np.ones(model.shape)
runtime.em.make_tomography_step = step
for case_id in sys.argv[3:]:
    result = study.run_case(manifest_path, case_id)
    assert result['status'] == 'complete'
    assert runtime.em.METHOD_DESCRIPTION is old_description
    before = len(calls)
    assert study.run_case(manifest_path, case_id) == result
    assert len(calls) == before, 'completed case was recomputed'
assert len(calls) == 12 * len(sys.argv[3:])
'''


def run_child(runtime, manifest_path, cases, code=RUN_CHILD):
    result = subprocess.run(
        [sys.executable, "-c", code, str(runtime / "runner"), str(manifest_path), *cases],
        cwd=manifest_path.parent, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_all_nine_cases_use_production_api_and_resume(saved_study):
    path, manifest = saved_study
    code = RUN_CHILD + r'''
import report
summary = report.generate_report(manifest_path, manifest_path.parent / 'report')
assert len(summary['cases']) == 9
assert summary['dataset']['pair_count'] == 98280
for row in summary['cases']:
    assert row['status'] == 'complete'
    assert row['completed_cycles'] == 12
    assert len(row['rmse_curve']) == 13
    assert row['final_rmse_m_s'] is not None
    assert row['hypocentres'] is None  # numerical stub deliberately emits no E-step weights
'''
    run_child(Path(manifest["runtime_root"]), path, [case["id"] for case in manifest["cases"]], code)
    for case in manifest["cases"]:
        result = json.loads((path.parent / "cases" / case["id"] / "result.json").read_text())
        assert result["status"] == "complete"
        meta = json.loads((Path(result["run_dir"]) / "meta.json").read_text())
        params = meta["run_params"]
        assert params["effective_config"]["lambda_reg"] == case["lambda_reg"]
        assert params["effective_config"]["coverage_damping_power"] == case["coverage_damping_power"]
        assert params["initial_model_source"]["used_as_checkpoint"] is False
        assert params["provenance"]["manifest_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert params["travel_times"] == run_case.SAVED_TRAVEL_TIMES


@pytest.mark.parametrize("change,match", [
    ({"resources": {**run_case.RESOURCE_VALUES, "forward_workers": 31}}, "resources"),
    ({"resources": {**run_case.RESOURCE_VALUES, "max_parallel_runs": 2.}}, "resources"),
    ({"runtime_root": "relative"}, "absolute"),
    ({"created_utc": "2026-10-17T00:00:00+01:00"}, "UTC"),
    ({"created_utc": "2026-10-17T00:00:00"}, "UTC"),
    ({"created_utc": None}, "UTC"),
    ({"code_sha256": {"../escape": "0" * 64}}, "Unsafe"),
    ({"cases": [{**CASES[1], "n_workers": 1}]}, "no extra config"),
    ({"cases": [CASES[1], CASES[1]]}, "Duplicate"),
    ({"cases": [{**CASES[1], "lambda_reg": True}]}, "finite"),
    ({"cases": [{**CASES[1], "lambda_reg": float("nan")}]}, "finite"),
    ({"cases": [{**CASES[1], "is_reference": False}]}, "exactly one"),
])
def test_manifest_rejects_unsafe_or_changed_contract(tmp_path, change, match):
    manifest = manifest_for(tmp_path)
    manifest.update(change)
    with pytest.raises(ValueError, match=match):
        run_case.validate_manifest(manifest)


def test_build_config_preserves_every_non_case_field(tmp_path):
    manifest = manifest_for(tmp_path)
    run_case.validate_manifest(manifest)  # dataset is intentionally not required
    meta = {"run_params": {**manifest["baseline_config"], "travel_times": run_case.SAVED_TRAVEL_TIMES}}
    for case in manifest["cases"]:
        config = run_case.build_config(manifest, case, InversionConfig, meta)
        actual = run_case._json(asdict(config))
        for key, value in manifest["baseline_config"].items():
            if key not in run_case.RUNTIME_FIELDS | {"lambda_reg", "coverage_damping_power"}:
                assert actual[key] == value
        assert config.n_workers == 12 and config.save_runs
    manifest["baseline_config"]["temperature"] = .25
    with pytest.raises(ValueError, match="differs from saved"):
        run_case.build_config(manifest, CASES[0], InversionConfig, meta)


def test_verify_initial_requires_bitwise_identity(tmp_path):
    initial = np.full((20, 7, 7), 5000.)
    path = tmp_path / "initial_model.npy"
    np.save(path, initial)
    run_case.verify_initial(initial, path, np)
    for bad in (initial.astype(np.float32), initial[:1], np.nextafter(initial, np.inf)):
        with pytest.raises(ValueError, match="BITWISE"):
            run_case.verify_initial(bad, path, np)


@pytest.mark.parametrize("tamper", ["marker", "final", "quality", "effective_config", "run_params", "provenance"])
def test_resume_rejects_corrupt_complete_artifact(saved_study, tamper):
    path, manifest = saved_study
    case = manifest["cases"][0]
    runtime = Path(manifest["runtime_root"])
    run_child(runtime, path, [case["id"]])
    result_path = path.parent / "cases" / case["id"] / "result.json"
    before = json.loads(result_path.read_text())
    run = Path(before["run_dir"])
    if tamper == "marker":
        (run / "iter_11" / "complete.json").unlink()
    elif tamper == "final":
        np.save(run / "final_model.npy", np.full((20, 7, 7), np.nan))
    elif tamper == "quality":
        (run / "quality.jsonl").write_text('{"iter":0,"rms_m_s":1,"avg_abs_pct_dev":1}\n')
    else:
        meta = json.loads((run / "meta.json").read_text())
        if tamper == "run_params":
            meta["run_params"]["temperature"] = .25
        elif tamper == "effective_config":
            meta["run_params"]["effective_config"]["n_workers"] = 1
        else:
            meta["run_params"]["provenance"]["manifest_sha256"] = "0" * 64
        write_json(run / "meta.json", meta)
    code = RUN_CHILD[:RUN_CHILD.index('for case_id in sys.argv[3:]:')] + r'''
try:
    study.run_case(manifest_path, sys.argv[3])
except (ValueError, OSError):
    pass
else:
    raise AssertionError('corrupt completion was reused')
assert not calls
'''
    run_child(runtime, path, [case["id"]], code)
    failed = json.loads(result_path.read_text())
    assert failed["status"] == "failed" and failed["resume_rejected"]
    assert failed["run_dir"] == before["run_dir"]
    assert failed["attempts"][-1]["status"] == "complete"


def test_optional_pick_noise_is_resolved_in_production_metadata(saved_study):
    path, manifest = saved_study
    for key in ("weight_noise_relative_sigma", "weight_noise_absolute_sigma_s"):
        manifest["baseline_config"][key] = None
    write_json(path, manifest)
    write_json(Path(manifest["reference_run"]) / "meta.json", {"run_params": {
        **manifest["baseline_config"], "travel_times": run_case.SAVED_TRAVEL_TIMES}})
    run_child(Path(manifest["runtime_root"]), path, [manifest["cases"][0]["id"]])


@pytest.mark.parametrize("kind", ["input", "runtime"])
def test_hash_failure_is_persisted_before_production_import(saved_study, monkeypatch, kind):
    path, manifest = saved_study
    if kind == "input":
        input_dir = Path(manifest["experiments_root"]) / "input" / manifest["experiment_id"]
        with (input_dir / "generation.json").open("a") as stream:
            stream.write("\n")
    else:
        manifest["code_sha256"]["projects/EMTomo/eikonal.py"] = "0" * 64
        write_json(path, manifest)
    monkeypatch.setattr(run_case, "load_runtime", lambda _: pytest.fail("production imported before hash check"))
    case_id = manifest["cases"][0]["id"]
    with pytest.raises(ValueError, match="SHA256"):
        run_case.run_case(path, case_id)
    result = json.loads((path.parent / "cases" / case_id / "result.json").read_text())
    assert result["status"] == "failed" and result["run_dir"] is None


def test_runtime_rejects_cached_production_from_working_checkout(saved_study):
    path, manifest = saved_study
    code = f'''
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import run_case as study
sys.path.insert(0, {str(ROOT / 'projects/EMTomo')!r})
import config
manifest = json.loads(Path(sys.argv[2]).read_text())
try:
    study.load_runtime(manifest)
except ValueError as error:
    assert 'outside runtime snapshot' in str(error)
else:
    raise AssertionError('cached working-checkout config was accepted')
'''
    run_child(Path(manifest["runtime_root"]), path, [], code)


def test_failed_em_restores_method_and_preserves_retry_history(saved_study):
    path, manifest = saved_study
    case_id = manifest["cases"][0]["id"]
    code = RUN_CHILD[:RUN_CHILD.index('for case_id in sys.argv[3:]:')] + r'''
def fail_step(*args, **kwargs):
    raise RuntimeError('fixture numerical failure')
runtime.em.make_tomography_step = fail_step
try:
    study.run_case(manifest_path, sys.argv[3])
except RuntimeError as error:
    assert str(error) == 'fixture numerical failure'
else:
    raise AssertionError('failure was swallowed')
assert runtime.em.METHOD_DESCRIPTION is old_description
'''
    runtime = Path(manifest["runtime_root"])
    run_child(runtime, path, [case_id], code)
    result_path = path.parent / "cases" / case_id / "result.json"
    failed = json.loads(result_path.read_text())
    assert failed["status"] == "failed" and "fixture numerical failure" in failed["error"]
    run_child(runtime, path, [case_id])
    retried = json.loads(result_path.read_text())
    assert retried["status"] == "complete" and retried["run_dir"] != failed["run_dir"]
    assert retried["attempts"][-1]["run_dir"] == failed["run_dir"]
    assert Path(failed["run_dir"]).is_dir()


def test_saved_real_hard1000_baseline_and_initial_read_only(tmp_path):
    runs = list((ROOT / "projects/EMTomo/runs").glob("run_fl-hard-c1000-*20261013_200119"))
    if len(runs) != 1:
        pytest.skip("Saved hard1000 reference is not installed")
    reference = runs[0]
    meta = json.loads((reference / "meta.json").read_text())
    allowed = set(asdict(InversionConfig()))
    manifest = manifest_for(tmp_path)
    manifest["baseline_config"] = {k: v for k, v in meta["run_params"].items() if k in allowed}
    config = run_case.build_config(manifest, CASES[1], InversionConfig, meta)
    # Initializer is pure; do not prepare arrivals, warm JIT or launch inversion.
    sys.path.insert(0, str(ROOT / "projects/EMTomo"))
    try:
        from experiment_runner import initial_velocity
        run_case.verify_initial(initial_velocity(config, (20, 7, 7)), reference / "initial_model.npy", np)
    finally:
        sys.path.remove(str(ROOT / "projects/EMTomo"))


@pytest.mark.parametrize("module", ["run_case.py", "report.py"])
def test_direct_cli_help_outside_checkout(tmp_path, module):
    result = subprocess.run([sys.executable, str(Path(__file__).with_name(module)), "--help"],
                            cwd=tmp_path, text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "--manifest" in result.stdout
