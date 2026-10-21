"""Elevated pipeline contracts and real detached wrapper/queue smoke, no solves.

Regression assertions are deliberately not xfailed: unsafe publication is a
pipeline bug, not something these tests should normalize or patch around.
Only temporary snapshots/stages are fabricated; production sources are read-only.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock, create_autospec

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.EMTomo.studies.geysers2011 import pipeline as oldqueue
from projects.EMTomo.studies.geysers_elevated import pipeline, run_case
from projects.EMTomo.studies.geysers_elevated import test_run_case as saved

# Reuse the real saved native 3776 x 26 study and frozen legacy-sphere fixture.
runtime_snapshot = saved.runtime_snapshot
saved_study = saved.saved_study
LINUX = pytest.mark.skipif(sys.platform != "linux", reason="queue identity uses Linux /proc")


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")


def child(command, cwd, *, env=None, timeout=30):
    result = subprocess.run(command, cwd=cwd, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **(env or {})},
                            capture_output=True, text=True, timeout=timeout, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def wait_for(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.02)
    pytest.fail("Timed out waiting for detached queue; inspect temporary queue/stage logs")


def test_cases_exact_seven_order_resources_and_shared_queue():
    assert pipeline.core is oldqueue
    cases = pipeline.cases()
    assert [(case["candidate_mode"], case["n_candidates"]) for case in cases] == [
        ("hard", 5), ("soft", 5), ("hard", 10), ("soft", 10),
        ("hard", 15), ("soft", 15), ("hard", 1)]
    assert len({case["id"] for case in cases}) == len({case["run_name"] for case in cases}) == 7
    assert [case["id"] for case in cases if case["is_reference"]] == ["hard-k5"]
    for case in cases:
        soft = case["candidate_mode"] == "soft"
        assert case["weights_top_n"] == (case["n_candidates"] if soft else 1)
        assert case["temperature"] == (None if soft else 1.)
        assert (case["lambda_reg"], case["coverage_damping_power"]) == (1., 2.)
    assert run_case.shared.RESOURCE_VALUES == {
        "forward_workers": 32, "inversion_workers_per_run": 12,
        "blas_threads_per_run": 12, "max_parallel_runs": 2}
    pipeline.cases()[0]["n_candidates"] = 999
    assert pipeline.cases()[0]["n_candidates"] == 5


@pytest.fixture
def initialization(saved_study, monkeypatch):
    from projects.EMTomo.studies.geysers_elevated import prepare

    s = saved_study
    repo = s.state / "checkout"
    historical = s.state / "historical"
    shutil.copytree(s.manifest["runtime_root"], historical)
    # Ensure copying the current production source cannot accidentally pass.
    with (historical / "projects/EMTomo/eikonal.py").open("a") as stream:
        stream.write("\n# immutable historical source fixture\n")
    hashes = {path.relative_to(historical).as_posix(): oldqueue.sha(path)
              for path in historical.glob("projects/**/*.py")}
    runs = repo / "projects/EMTomo/runs"
    reference = runs / "run_gy11-hard-c1000-lam1-cov2-latest"
    shutil.copytree(s.manifest["reference_run"], reference)
    meta = oldqueue.read(reference / "meta.json")
    meta.update(started_at="2026-10-18T01:00:00Z")
    meta["run_params"]["provenance"] = {"runtime_root": str(historical), "runtime_hashes": hashes}
    put(reference / "meta.json", meta)
    for iteration in range(meta["run_params"]["n_cycles"]):
        put(reference / f"iter_{iteration}/complete.json", {"status": "complete"})
    (reference / "final_model.npy").write_bytes(b"completion marker, never loaded")
    experiments = repo / "projects/forward_modeling/experiments"
    shutil.copytree(s.manifest["experiments_root"], experiments)
    input_dir = experiments / "input" / s.manifest["experiment_id"]
    dataset = {"experiment_id": s.manifest["experiment_id"], "input_dir": str(input_dir),
               "event_count": 3776, "station_count": 26,
               "source_hashes": {"generation.json": "a" * 64}, "generation_sha256": "b" * 64}
    preparation = Mock(return_value=dataset)
    monkeypatch.setattr(prepare, "prepare_experiment", preparation)
    initial = np.load(reference / "initial_model.npy", allow_pickle=False)
    initial_velocity = Mock(return_value=initial.copy())
    monkeypatch.setitem(sys.modules, "config", SimpleNamespace(InversionConfig=saved.InversionConfig))
    monkeypatch.setitem(sys.modules, "experiment_runner", SimpleNamespace(initial_velocity=initial_velocity))
    monkeypatch.setattr(pipeline, "REPO", repo)
    state = s.state / "new-queue"
    state.mkdir()
    return SimpleNamespace(state=state, historical=historical, hashes=hashes, reference=reference,
                           runs=runs, preparation=preparation, initial_velocity=initial_velocity,
                           dataset=dataset, experiments=experiments)


def test_initialize_freezes_historical_production_and_current_wrapper_provenance(initialization):
    f = initialization
    before = saved.snapshot(f.historical), saved.snapshot(f.reference), saved.snapshot(f.experiments)
    manifest = pipeline.initialize(f.state)
    assert manifest == oldqueue.read(f.state / "manifest.json")
    run_case.verify_hashes(manifest)
    assert manifest["schema_version"] == 2 and manifest["cases"] == pipeline.cases()
    assert manifest["reference_run"] == str(f.reference)
    assert manifest["dataset"] == f.dataset
    assert manifest["resources"] == run_case.shared.RESOURCE_VALUES
    assert manifest["forward_spec"] == {"refinement": 2, "source_radius_cells": 2,
                                        "noise": {"relative_sigma": .01, "absolute_sigma_s": .01, "seed": 42}}
    assert manifest["temperature_policy"] == {
        "common_to_soft_cases": True, "bounds": [1., 16.], "target_median_ess_k15": 3.,
        "max_initial_model_observations": 128, "oracle_truth_used": False, "hard_temperature": 1.}
    frozen = Path(manifest["runtime_root"])
    for name, digest in f.hashes.items():
        assert manifest["code_sha256"][name] == digest
        assert (frozen / name).read_bytes() == (f.historical / name).read_bytes()
    assert (frozen / "projects/EMTomo/eikonal.py").read_bytes() != (ROOT / "projects/EMTomo/eikonal.py").read_bytes()
    helpers = {"queue_core.py": Path(oldqueue.__file__), "_shared_case.py": Path(run_case.shared.__file__),
               "_shared_report.py": pipeline.HERE.parent / "geysers2011/report.py"}
    helpers.update({name: pipeline.HERE / name for name in
                    ("pipeline.py", "run_case.py", "prepare.py", "calibrate.py", "report.py")})
    for name, source in helpers.items():
        assert (frozen / "runner" / name).read_bytes() == source.read_bytes()
        assert manifest["code_sha256"][f"runner/{name}"] == oldqueue.sha(source)
    assert {path.name for path in (f.state / "cases").iterdir()} == {case["id"] for case in manifest["cases"]}
    assert f.initial_velocity.call_count == 1
    config, shape = f.initial_velocity.call_args.args
    assert shape == (20, 7, 7) and config.subdivision == 8 and config.lambda_reg == 1
    f.preparation.assert_called_once_with()
    assert before == (saved.snapshot(f.historical), saved.snapshot(f.reference), saved.snapshot(f.experiments))
    assert not (f.state / "forward_complete.json").exists()


def test_initialize_existing_manifest_is_validated_read_only_reuse(initialization):
    f = initialization
    manifest = pipeline.initialize(f.state)
    before = saved.snapshot(f.state)
    f.preparation.side_effect = AssertionError("resume must use the published snapshot")
    f.initial_velocity.side_effect = AssertionError("resume must not regenerate initial model")
    assert pipeline.initialize(f.state) == manifest
    assert before == saved.snapshot(f.state)


@pytest.mark.parametrize("damage", ["runtime", "reference", "input"])
def test_initialize_resume_rejects_changed_hashes(initialization, damage):
    f = initialization
    manifest = pipeline.initialize(f.state)
    path = {"runtime": Path(manifest["runtime_root"]) / "runner/queue_core.py",
            "reference": f.reference / "meta.json",
            "input": f.experiments / "input" / manifest["experiment_id"] / "generation.json"}[damage]
    path.write_bytes(path.read_bytes() + b"\nchanged")
    before = (f.state / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        pipeline.initialize(f.state)
    assert (f.state / "manifest.json").read_bytes() == before


def test_initialize_changed_historical_source_never_publishes_manifest(initialization):
    f = initialization
    (f.historical / "projects/EMTomo/eikonal.py").write_text("changed historical method")
    with pytest.raises(ValueError, match="Historical source changed"):
        pipeline.initialize(f.state)
    assert not (f.state / "manifest.json").exists()


def test_initialize_refuses_unpublished_runtime_without_overwrite(initialization):
    f = initialization
    (f.state / "runtime").mkdir()
    marker = f.state / "runtime/manual.txt"
    marker.write_text("inspect me")
    with pytest.raises(ValueError, match="Unpublished runtime exists"):
        pipeline.initialize(f.state)
    assert marker.read_text() == "inspect me" and not (f.state / "manifest.json").exists()


def test_latest_reference_requires_all_iterations_and_chooses_latest(initialization):
    f = initialization
    for name, started, completed in (("older", "2026-10-17T00:00:00Z", True),
                                     ("partial", "2026-10-19T00:00:00Z", False)):
        run = f.runs / f"run_gy11-hard-c1000-{name}"
        shutil.copytree(f.reference, run)
        meta = oldqueue.read(run / "meta.json")
        meta["started_at"] = started
        put(run / "meta.json", meta)
        if not completed:
            (run / "iter_11/complete.json").unlink()
    assert pipeline.latest_reference(f.runs) == f.reference


@pytest.fixture
def forward(saved_study, monkeypatch):
    from projects.forward_modeling import experiments, solver
    from projects.forward_modeling.model import PointSet

    s = saved_study
    manifest = copy.deepcopy(s.manifest)
    private_runtime = s.state / "forward-runtime"
    shutil.copytree(manifest["runtime_root"], private_runtime)
    manifest.update(runtime_root=str(private_runtime), cases=pipeline.cases(), forward_spec={
        "refinement": 2, "source_radius_cells": 2,
        "noise": {"relative_sigma": .01, "absolute_sigma_s": .01, "seed": 42}})
    put(s.path, manifest)
    metadata = oldqueue.read(s.output / "metadata.json")
    metadata["config"] = {"refinement": 2, "source_radius_cells": 2}
    contents = {"metadata.json": json.dumps(metadata).encode(),
                "arrivals.csv": (s.output / "arrivals.csv").read_bytes()}
    model, stations, original_events = experiments.load_inputs(Path(manifest["experiments_root"]), manifest["experiment_id"])
    xyz = original_events.coordinates_m.copy()
    xyz[:, 2] = np.linspace(100., 6000., len(xyz))
    xyz[3, 2], xyz[1777, 2] = 1., 6999.
    events = PointSet(original_events.ids, xyz)
    shutil.rmtree(s.output)
    (s.state / "forward_complete.json").unlink()
    (s.state / "calibration.json").unlink()
    initial = np.load(Path(manifest["reference_run"]) / "initial_model.npy", allow_pickle=False)
    prepared = SimpleNamespace(initial_model=SimpleNamespace(velocity=initial), event_ids=events.ids,
                               station_ids=stations.ids, noise_sigmas=(.01, .01),
                               arrivals_table=np.zeros((3776, 26)), station_locs=stations.coordinates_m)
    runtime = SimpleNamespace(InversionConfig=saved.InversionConfig, np=np,
                              prepare_inversion=Mock(return_value=prepared))
    select_runtime = Mock(return_value=runtime)
    monkeypatch.setattr(run_case.shared, "load_runtime", select_runtime)
    monkeypatch.setitem(sys.modules, "run_case", run_case)
    convergence = create_autospec(solver.check_convergence, return_value={
        "max_abs_difference_s": .001, "max_absolute_time_difference_s": .002})
    load = create_autospec(experiments.load_inputs, return_value=(model, stations, events))

    def publish(*args, **kwargs):
        s.output.mkdir(parents=True)
        for name, data in contents.items():
            (s.output / name).write_bytes(data)
        return s.output

    run = create_autospec(experiments.run_experiment, side_effect=publish)
    monkeypatch.setattr(experiments, "load_inputs", load)
    monkeypatch.setattr(experiments, "run_experiment", run)
    monkeypatch.setattr(solver, "check_convergence", convergence)
    monkeypatch.setattr(solver, "compute_travel_times", Mock(side_effect=AssertionError("real solve forbidden")))
    calibration = {**s.calibration, "selection": {"reason": "fake lightweight calibration"}}
    staging_paths = []

    def calibrate(actual, config, selected_runtime, state_dir):
        assert actual is prepared and selected_runtime is runtime
        assert config.n_workers == 12 and config.candidate_mode == "hard" and config.n_candidates == 5
        staging = Path(state_dir)
        assert staging.parent == s.state and staging.name.startswith(".calibration-")
        assert not (s.state / "calibration.json").exists()
        assert not (s.state / "forward_complete.json").exists()
        staging_paths.append(staging)
        # The real calibrator also writes non-atomically. It must not see final state.
        (staging / "calibration.json").write_text('{"temperature":')
        put(staging / "calibration.json", calibration)
        return calibration

    calibrator = Mock(side_effect=calibrate)
    monkeypatch.setitem(sys.modules, "calibrate", SimpleNamespace(calibrate=calibrator))
    published = []
    atomic = oldqueue.atomic

    def observe_atomic(path, value):
        if path.name == "calibration.json":
            assert staging_paths and not staging_paths[-1].exists()
        if path.name == "forward_complete.json":
            assert oldqueue.read(s.state / "calibration.json") == calibration
        published.append(path.name)
        atomic(path, value)

    monkeypatch.setattr(pipeline.core, "atomic", observe_atomic)
    return SimpleNamespace(study=s, state=s.state, manifest=manifest, output=s.output, run=run,
                           load=load, convergence=convergence, runtime=runtime, prepared=prepared,
                           select_runtime=select_runtime, calibrator=calibrator, calibration=calibration,
                           staging_paths=staging_paths, published=published, metadata=metadata)


def test_worker_native_forward_resources_counts_and_staged_atomic_calibration(forward):
    f = forward
    pipeline.worker(f.state)
    f.load.assert_called_once_with(Path(f.manifest["experiments_root"]), f.manifest["experiment_id"])
    selected = f.convergence.call_args.args[2]
    expected = sorted(set(np.linspace(0, 3775, 16, dtype=int).tolist() + [3, 1777]))
    assert selected.ids == tuple(f"e{i:04}" for i in expected)
    assert f.convergence.call_args.kwargs["workers"] == 16
    assert vars(f.convergence.call_args.kwargs["config"]) == {"refinement": 2, "source_radius_cells": 2}
    assert f.run.call_args.args == (Path(f.manifest["experiments_root"]), f.manifest["experiment_id"])
    assert f.run.call_args.kwargs["workers"] == 32
    assert vars(f.run.call_args.kwargs["noise"]) == f.manifest["forward_spec"]["noise"]
    assert vars(f.run.call_args.kwargs["config"]) == {"refinement": 2, "source_radius_cells": 2}
    assert len(f.prepared.event_ids) == 3776 and len(f.prepared.station_ids) == 26
    assert f.prepared.arrivals_table.shape == (3776, 26)
    assert f.runtime.prepare_inversion.call_args.kwargs == {"experiments_root": Path(f.manifest["experiments_root"])}
    assert oldqueue.read(f.state / "forward_convergence.json")["event_ids"] == list(selected.ids)
    assert f.published == ["forward_convergence.json", "calibration.json", "forward_complete.json"]
    f.calibrator.assert_called_once()
    assert not f.staging_paths[0].exists()
    ledger = oldqueue.read(f.state / "forward_complete.json")
    assert ledger["output_sha256"] == {name: oldqueue.sha(f.output / name) for name in ("arrivals.csv", "metadata.json")}
    assert ledger["calibration_sha256"] == oldqueue.sha(f.state / "calibration.json")
    assert oldqueue.read(f.state / "calibration.json") == f.calibration
    before = saved.snapshot(f.output), saved.snapshot(f.state / "cases"), (f.state / "calibration.json").stat().st_mtime_ns
    publications = f.published.copy()
    pipeline.worker(f.state)
    assert f.run.call_count == f.convergence.call_count == f.load.call_count == f.calibrator.call_count == 1
    assert f.runtime.prepare_inversion.call_count == 2
    assert publications == f.published
    assert before == (saved.snapshot(f.output), saved.snapshot(f.state / "cases"), (f.state / "calibration.json").stat().st_mtime_ns)


def test_worker_forward_ledger_is_schema2_and_names_exact_experiment(forward):
    pipeline.worker(forward.state)
    ledger = oldqueue.read(forward.state / "forward_complete.json")
    assert {key: ledger.get(key) for key in ("schema_version", "experiment_id", "status")} == {
        "schema_version": 2, "experiment_id": forward.manifest["experiment_id"], "status": "complete"}


@pytest.mark.parametrize("damage", ["arrivals.csv", "metadata.json", "calibration.json"])
def test_worker_resume_output_hash_mismatch_fails_without_republication(forward, damage):
    f = forward
    pipeline.worker(f.state)
    path = (f.state if damage == "calibration.json" else f.output) / damage
    path.write_bytes(path.read_bytes() + b"\n")  # valid JSON/CSV, but wrong committed bytes
    before = (f.state / "forward_complete.json").read_bytes(), f.published.copy()
    with pytest.raises(ValueError, match="Forward output changed|Temperature calibration changed"):
        pipeline.worker(f.state)
    assert before == ((f.state / "forward_complete.json").read_bytes(), f.published)
    assert f.run.call_count == f.calibrator.call_count == 1


def test_worker_hash_failure_precedes_runtime_import(forward):
    f = forward
    (Path(f.manifest["runtime_root"]) / "runner/run_case.py").write_text("changed frozen code")
    with pytest.raises(ValueError, match="Runtime SHA256 mismatch"):
        pipeline.worker(f.state)
    f.select_runtime.assert_not_called()
    f.run.assert_not_called()
    assert not (f.state / "forward_complete.json").exists()


@pytest.mark.parametrize("attribute,value", [("event_ids", range(3780)), ("station_ids", range(25)),
                                             ("noise_sigmas", (.02, .01))])
def test_worker_rejects_wrong_geometry_counts_or_noise(forward, attribute, value):
    setattr(forward.prepared, attribute, value)
    with pytest.raises(ValueError, match="Unexpected geometry/counts/noise"):
        pipeline.worker(forward.state)
    forward.calibrator.assert_not_called()
    assert not (forward.state / "forward_complete.json").exists()


@pytest.mark.parametrize("key,value", [("max_abs_difference_s", float("nan")),
                                        ("max_absolute_time_difference_s", -.01)])
def test_worker_invalid_convergence_never_runs_full_forward(forward, key, value):
    forward.convergence.return_value[key] = value
    with pytest.raises(ValueError, match="Invalid convergence"):
        pipeline.worker(forward.state)
    forward.run.assert_not_called()
    forward.calibrator.assert_not_called()
    assert not (forward.state / "forward_complete.json").exists()


def test_worker_calibration_failure_removes_partial_staging_and_never_commits(forward):
    def fail(*args):
        staging = Path(args[-1])
        (staging / "calibration.json").write_text('{"temperature":')
        raise RuntimeError("fake calibration interrupted")

    forward.calibrator.side_effect = fail
    with pytest.raises(RuntimeError, match="calibration interrupted"):
        pipeline.worker(forward.state)
    assert not list(forward.state.glob(".calibration-*"))
    assert not (forward.state / "calibration.json").exists()
    assert not (forward.state / "forward_complete.json").exists()


@pytest.mark.parametrize("temperature", [0., 17.])
def test_worker_must_not_commit_out_of_policy_calibration(forward, temperature):
    forward.calibration["temperature"] = temperature
    with pytest.raises(ValueError, match="temperature|Temperature|calibration|Calibration"):
        pipeline.worker(forward.state)
    assert not (forward.state / "forward_complete.json").exists()


# Unlike the oldqueue smoke, pipeline.py below is the actual elevated wrapper,
# not a replacement CLI. Only run_case/calibrate/report stage boundaries are fake.
FAKE_STAGES = r'''
import argparse, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace

def stage(kind, state, identifier=None):
    manifest = json.loads((state / 'manifest.json').read_text())
    identifier = identifier or kind
    assert Path.cwd() == Path(manifest['runtime_root'])
    assert os.getsid(0) == os.getpid()
    assert os.read(0, 1) == b''
    assert os.environ['PYTHONPATH'] == manifest['runtime_root']
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        assert os.environ[name] == '1'
    for descriptor in Path('/proc/self/fd').iterdir():
        try:
            assert descriptor.resolve() != state / 'queue.lock'
        except FileNotFoundError:
            pass
    def record(event):
        fd = os.open(state / 'events.jsonl', os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        try:
            os.write(fd, (json.dumps(dict(id=identifier, event=event, pid=os.getpid(), sid=os.getsid(0))) + '\n').encode())
        finally:
            os.close(fd)
    record('start')
    if kind == 'forward':
        deadline = time.monotonic() + 12
        while not (state / 'release_forward').exists():
            if time.monotonic() > deadline:
                raise RuntimeError('finite fake forward deadline')
            time.sleep(.01)
    if kind == 'case':
        assert (state / 'forward_complete.json').exists()
        time.sleep(.06)
        marker = state / 'cases' / identifier / 'result.json'
        event = 'reused' if marker.exists() else 'computed'
        if not marker.exists():
            marker.write_text(json.dumps({'status': 'complete', 'case_id': identifier}))
        record(event)
    elif kind == 'report':
        assert all((state / 'cases' / case['id'] / 'result.json').is_file() for case in manifest['cases'])
        (state / 'report_summary.json').write_text(json.dumps({'case_count': len(manifest['cases'])}))
    record('end')

def make_runtime(manifest):
    import numpy as np
    from projects.EMTomo.config import InversionConfig
    initial = np.load(Path(manifest['reference_run']) / 'initial_model.npy', allow_pickle=False)
    def prepare(experiment, config, *, experiments_root):
        assert experiment == manifest['experiment_id'] and experiments_root == Path(manifest['experiments_root'])
        assert config.n_workers == 12 and config.weights_top_n == 1 and config.temperature == 1.
        return SimpleNamespace(initial_model=SimpleNamespace(velocity=initial), event_ids=range(3776),
                               station_ids=range(26), noise_sigmas=(.01, .01))
    return SimpleNamespace(InversionConfig=InversionConfig, np=np, prepare_inversion=prepare)

def cli(kind):
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--case-id')
    args = parser.parse_args()
    stage(kind, args.manifest.parent, args.case_id)
'''


@pytest.fixture
def detached(saved_study):
    s = saved_study
    runtime = s.state / "smoke-runtime"
    shutil.copytree(s.manifest["runtime_root"], runtime)
    runner = runtime / "runner"
    shutil.copyfile(runner / "run_case.py", runner / "case_contract.py")
    shutil.copyfile(pipeline.__file__, runner / "pipeline.py")
    shutil.copyfile(oldqueue.__file__, runner / "queue_core.py")
    (runner / "fake_stages.py").write_text(FAKE_STAGES)
    (runner / "run_case.py").write_text(
        "from case_contract import verify_hashes, build_config, verify_observations, shared\n"
        "from fake_stages import make_runtime, cli\n"
        "shared.load_runtime = make_runtime\n"
        "if __name__ == '__main__': cli('case')\n")
    (runner / "calibrate.py").write_text(
        "import json\nfrom pathlib import Path\nfrom fake_stages import stage\n"
        "def calibrate(prepared, config, runtime, state_dir):\n"
        "    stage('forward', Path(state_dir).parent)\n"
        "    result = {'temperature': 4.25, 'selection': {'fake': True}}\n"
        "    (Path(state_dir) / 'calibration.json').write_text(json.dumps(result))\n"
        "    return result\n")
    (runner / "report.py").write_text("from fake_stages import cli\ncli('report')\n")
    # Accelerate polling in this test's subprocesses, without changing the wrapper
    # or queue source bytes. Startup is stdlib-only, including for --help.
    (runtime / "sitecustomize.py").write_text(
        "import sys,time\nfrom pathlib import Path\nfrom types import SimpleNamespace\n"
        "runner = Path(__file__).parent / 'runner'\n"
        "if Path(sys.argv[0]).resolve().parent == runner:\n"
        "    sys.path.insert(0, str(runner))\n"
        "    import queue_core\n"
        "    queue_core.time = SimpleNamespace(monotonic=time.monotonic, sleep=lambda t: time.sleep(min(t, .01)))\n")
    manifest = copy.deepcopy(s.manifest)
    manifest.update(runtime_root=str(runtime), cases=pipeline.cases(),
                    forward_spec={"noise": {"relative_sigma": .01, "absolute_sigma_s": .01, "seed": 42}})
    manifest["code_sha256"] = {path.relative_to(runtime).as_posix(): oldqueue.sha(path)
                                for path in runtime.rglob("*.py")}
    metadata = oldqueue.read(s.output / "metadata.json")
    metadata["config"] = {"refinement": 2, "source_radius_cells": 2}
    put(s.output / "metadata.json", metadata)
    put(s.path, manifest)
    (s.state / "forward_complete.json").unlink()
    (s.state / "calibration.json").unlink()
    return SimpleNamespace(state=s.state, manifest=manifest, runtime=runtime, runner=runner,
                           env={"PYTHONPATH": str(runtime)})


def queue_status(f):
    path = f.state / "status.json"
    return oldqueue.read(path) if path.exists() else {}


def events(f):
    path = f.state / "events.jsonl"
    return [json.loads(row) for row in path.read_text().splitlines()] if path.exists() else []


def cleanup(f):
    status = queue_status(f)
    if oldqueue.alive(status):
        os.kill(status["pid"], signal.SIGTERM)
        wait_for(lambda: not oldqueue.alive(status))
    for process in status.get("active", []):
        if oldqueue.alive(process):
            os.killpg(process["pid"], signal.SIGKILL)
            wait_for(lambda: not oldqueue.alive(process))


@LINUX
@pytest.mark.parametrize("stop", [False, True], ids=["finish-and-resume", "stop"])
def test_real_detached_elevated_wrapper_queue_smoke(detached, stop):
    f = detached
    command = [sys.executable, "-B", str(pipeline.HERE / "pipeline.py")]
    try:
        child([*command, "launch", "--state", str(f.state)], f.state, env=f.env)
        status = queue_status(f)
        assert oldqueue.alive(status) and status["active"][0]["id"] == "forward"
        assert os.getsid(status["pid"]) == status["pid"]
        original_pid = status["pid"]
        with pytest.raises(RuntimeError, match="exclusive OS lock"):
            fd = oldqueue.acquire(f.state)
            os.close(fd)
        reported = child([*command, "status", "--state", str(f.state)], f.state, env=f.env)
        assert json.loads(reported.stdout)["scheduler_alive"] is True
        if stop:
            child([*command, "stop", "--state", str(f.state)], f.state, env=f.env)
        else:
            (f.state / "release_forward").touch()
        final = wait_for(lambda: (value if (value := queue_status(f)).get("status") not in (None, "running") else None))
        assert final["status"] == ("interrupted" if stop else "complete"), final
        assert final["active"] == []
        wait_for(lambda: not oldqueue.alive(status))
        if stop:
            assert not any(row["id"] != "forward" for row in events(f))
            return
        case_ids = {case["id"] for case in f.manifest["cases"]}
        assert len(final["finished_cases"]) == 9
        assert final["finished_cases"][0]["id"] == "forward"
        assert final["finished_cases"][-1]["id"] == "report"
        assert all(row["exit_code"] == 0 and not row["timed_out"] for row in final["finished_cases"])
        rows = events(f)
        starts = [row["id"] for row in rows if row["event"] == "start"]
        assert starts[0] == "forward" and starts[-1] == "report" and set(starts[1:-1]) == case_ids
        running, peak = set(), 0
        for row in rows:
            assert row["sid"] == row["pid"]
            if row["id"] in case_ids:
                if row["event"] == "start":
                    running.add(row["id"])
                    peak = max(peak, len(running))
                elif row["event"] == "end":
                    running.remove(row["id"])
        assert not running and 1 <= peak <= 2
        assert oldqueue.read(f.state / "report_summary.json") == {"case_count": 7}
        markers = {name: (f.state / "cases" / name / "result.json").read_bytes() for name in case_ids}
        child([*command, "launch", "--state", str(f.state)], f.state, env=f.env)
        resumed = wait_for(lambda: (value if (value := queue_status(f)).get("pid") != original_pid
                                   and value.get("status") == "complete" else None))
        wait_for(lambda: not oldqueue.alive(resumed))
        assert len([row for row in events(f) if row["id"] == "forward" and row["event"] == "start"]) == 1
        assert {row["id"] for row in events(f) if row["event"] == "reused"} == case_ids
        assert markers == {name: (f.state / "cases" / name / "result.json").read_bytes() for name in case_ids}
        fd = oldqueue.acquire(f.state)
        os.close(fd)
    finally:
        cleanup(f)


def test_frozen_wrapper_import_and_cli_help_need_no_production(initialization):
    manifest = pipeline.initialize(initialization.state)
    runner = Path(manifest["runtime_root"]) / "runner"
    script = (
        "import sys; sys.path.insert(0, sys.argv[1]); import pipeline, queue_core; "
        "assert pipeline.core is queue_core; "
        "assert not any(name in sys.modules for name in "
        "('numpy', 'config', 'experiment_runner', 'eikonal', 'tomography')); "
        "assert len(pipeline.cases()) == 7"
    )
    child([sys.executable, "-B", "-S", "-c", script, str(runner)], initialization.state)
    result = child([sys.executable, "-B", "-S", str(runner / "pipeline.py"), "--help"], initialization.state)
    assert "forward" in result.stdout and "supervise" in result.stdout and "--lock-fd" in result.stdout
