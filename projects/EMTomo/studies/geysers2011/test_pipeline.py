"""Fast queue contracts: Linux flock/processes, tiny snapshots, no numerical solves.

Dummy stages test scheduling/resume delegation, not numerical case artifacts (those
are covered by test_run_case.py).
"""
from __future__ import annotations

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

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.EMTomo.studies.geysers2011 import pipeline, run_case

LINUX = pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc process identity")


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pipeline.atomic(path, value)


def wait_until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.01)
    pytest.fail("Timed out waiting for finite test subprocess")


def close_process(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGKILL)
    process.communicate(timeout=5)


def test_case_grid9():
    cases = pipeline.case_grid()
    assert len(cases) == 9
    assert {(c["lambda_reg"], c["coverage_damping_power"]) for c in cases} == {
        (lam, power) for lam in (.1, .5, 1.) for power in (0, 1, 2)}
    assert len({c["id"] for c in cases}) == len({c["run_name"] for c in cases}) == 9
    assert [c for c in cases if c["is_reference"]] == [cases[0]]
    assert (cases[0]["lambda_reg"], cases[0]["coverage_damping_power"]) == (.5, 1)
    assert all(set(c) == run_case.CASE_FIELDS for c in cases)
    assert pipeline.case_grid() == cases
    cases[0]["id"] = "mutated"
    assert pipeline.case_grid()[0]["id"] == "lam0.5-cov1"
    assert Path(pipeline.PYTHON) == Path(sys.executable)


def test_lock_duplicate_and_reacquire(tmp_path):
    state = tmp_path / "queue"
    fd = pipeline.acquire(state)
    try:
        with pytest.raises(RuntimeError, match="exclusive OS lock"):
            pipeline.acquire(state)
    finally:
        os.close(fd)
    fd = pipeline.acquire(state)
    os.close(fd)


def test_lock_survives_parent_close_when_fd_is_inherited(tmp_path):
    fd = pipeline.acquire(tmp_path)
    child = None
    try:
        child = subprocess.Popen(
            [sys.executable, "-c",
             "import os,sys; fd=int(sys.argv[1]); os.fstat(fd); "
             "print('ready', flush=True); sys.stdin.buffer.read(1)", str(fd)],
            pass_fds=(fd,), close_fds=True, start_new_session=True,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        os.close(fd)
        fd = None
        # Waiting on a marker file would also work; this child writes before blocking.
        import select
        assert select.select([child.stdout], [], [], 5)[0], "child did not inherit fd"
        assert child.stdout.readline() == b"ready\n"
        with pytest.raises(RuntimeError, match="exclusive OS lock"):
            pipeline.acquire(tmp_path)
        out, err = child.communicate(b"x", timeout=5)
        assert child.returncode == 0, out + err
        released = pipeline.acquire(tmp_path)
        os.close(released)
    finally:
        if fd is not None:
            os.close(fd)
        if child is not None:
            close_process(child)


@LINUX
def test_lock_rejects_orphan_child_but_not_recycled_pid(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"],
                             start_new_session=True)
    try:
        birth = wait_until(lambda: pipeline.identity(child.pid))
        put(tmp_path / "status.json", {"active": [{"pid": child.pid, "birth": birth}]})
        with pytest.raises(RuntimeError, match="earlier queue child"):
            pipeline.acquire(tmp_path)
        put(tmp_path / "status.json", {"active": [{"pid": child.pid, "birth": birth + "-stale"}]})
        fd = pipeline.acquire(tmp_path)
        os.close(fd)
    finally:
        close_process(child)


@LINUX
def test_identity_current_missing_and_zombie():
    birth = pipeline.identity(os.getpid())
    assert birth and birth.isdecimal()
    assert pipeline.alive({"pid": os.getpid(), "birth": birth})
    assert not pipeline.alive({"pid": os.getpid(), "birth": birth + "-stale"})
    assert not pipeline.alive({"pid": os.getpid(), "birth": None})
    assert not pipeline.alive(None)
    assert pipeline.identity(-1) is None
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        # waitid(WNOWAIT) ensures a zombie without reaping it as Popen.poll would.
        os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT)
        assert pipeline.identity(child.pid) is None
    finally:
        child.wait(timeout=5)


def test_identity_parses_comm_with_spaces_and_parentheses(monkeypatch):
    # starttime is field 22: field 3 is state, so index 19 after the final ')'.
    monkeypatch.setattr(Path, "read_text", lambda self: "123 (worker (odd) name) "
                        + " ".join(["S"] + ["0"] * 18 + ["987654"] + ["0"] * 5))
    assert pipeline.identity(123) == "987654"


@pytest.mark.parametrize("same_birth", [True, False])
def test_stop_checks_identity_before_signaling(tmp_path, monkeypatch, capsys, same_birth):
    put(tmp_path / "status.json", {"pid": 123456, "birth": "original", "active": []})
    monkeypatch.setattr(pipeline, "identity", lambda pid: "original" if same_birth else "recycled")
    kill = Mock()
    monkeypatch.setattr(pipeline.os, "kill", kill)
    monkeypatch.setattr(sys, "argv", ["pipeline.py", "stop", "--state", str(tmp_path)])
    if same_birth:
        assert pipeline.main() == 0
        kill.assert_called_once_with(123456, signal.SIGTERM)
        assert "Stop requested" in capsys.readouterr().out
    else:
        with pytest.raises(RuntimeError, match="Supervisor not alive"):
            pipeline.main()
        kill.assert_not_called()


def test_status_reports_identity_without_changing_disk(tmp_path, monkeypatch, capsys):
    original = {"pid": 123456, "birth": "old", "status": "running", "active": []}
    put(tmp_path / "status.json", original)
    monkeypatch.setattr(pipeline, "identity", lambda pid: None)
    monkeypatch.setattr(sys, "argv", ["pipeline.py", "status", "--state", str(tmp_path)])
    assert pipeline.main() == 0
    assert json.loads(capsys.readouterr().out)["scheduler_alive"] is False
    assert pipeline.read(tmp_path / "status.json") == original


# Only the stage boundaries are dummy. The copied supervisor/launcher, CLI,
# flock inheritance, hash verification, signals and subprocesses are real.
DUMMY_TASKS = r'''
import json, os, sys, time
from pathlib import Path

def run(stage):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--case-id')
    args = parser.parse_args()
    state = args.state or args.manifest.parent
    manifest = json.loads((state / 'manifest.json').read_text())
    control = json.loads((state / 'control.json').read_text())
    identifier = args.case_id or stage
    assert Path.cwd() == Path(manifest['runtime_root'])
    assert os.getsid(0) == os.getpid(), 'worker must own a separate session'
    assert os.read(0, 1) == b'', 'stdin must be detached'
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        assert os.environ[key] == '1'
    if sys.platform == 'linux':
        for fd in Path('/proc/self/fd').iterdir():
            try:
                assert fd.resolve() != state / 'queue.lock', 'worker inherited queue lock'
            except FileNotFoundError:
                pass
    def event(kind):
        record = dict(id=identifier, kind=kind, pid=os.getpid(), ppid=os.getppid(),
                      sid=os.getsid(0), time=time.monotonic())
        fd = os.open(state / 'events.jsonl', os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, (json.dumps(record) + '\n').encode())
        finally:
            os.close(fd)
    event('start')
    if stage == 'forward' and control.get('hold_forward'):
        deadline = time.monotonic() + 8
        while not (state / 'release_forward').exists():
            if time.monotonic() >= deadline:
                raise RuntimeError('finite dummy forward deadline')
            time.sleep(.01)
    time.sleep(.04)
    failure = control.get('forward_exit', 0) if stage == 'forward' else (
        control.get('report_exit', 0) if stage == 'report' else
        (7 if identifier in control.get('fail_cases', []) else 0))
    if failure:
        event('fail')
        return failure
    marker = state / ('forward_complete.json' if stage == 'forward' else
                      'report_summary.json' if stage == 'report' else
                      'cases/' + identifier + '/result.json')
    if stage != 'report' and marker.exists():
        event('reused')
    else:
        marker.write_text(json.dumps({'status': 'complete', 'id': identifier}))
        event('computed')
    event('end')
    return 0
'''


@pytest.fixture
def snapshot(tmp_path):
    state = tmp_path / "queue"
    runner = state / "runtime" / "runner"
    runner.mkdir(parents=True)
    # HERE in the executing copy is deliberately not the live source directory.
    shutil.copyfile(pipeline.__file__, runner / "pipeline_impl.py")
    shutil.copyfile(run_case.__file__, runner / "case_contract.py")
    (runner / "dummy_tasks.py").write_text(DUMMY_TASKS)
    (runner / "pipeline.py").write_text(
        "import os,sys,time\nfrom types import SimpleNamespace\n"
        "import pipeline_impl as queue\n"
        "queue.STAGE_TIMEOUT_S = float(os.environ.get('TEST_STAGE_TIMEOUT_S', queue.STAGE_TIMEOUT_S))\n"
        "queue.time = SimpleNamespace(monotonic=time.monotonic, "
        "sleep=lambda delay: time.sleep(min(delay, .01)))\n"
        "if sys.argv[1] == 'forward':\n"
        "    from dummy_tasks import run\n"
        "    del sys.argv[1]\n"
        "    raise SystemExit(run('forward'))\n"
        "raise SystemExit(queue.main())\n")
    (runner / "run_case.py").write_text(
        "from case_contract import verify_hashes\n"
        "if __name__ == '__main__':\n"
        "    from dummy_tasks import run\n"
        "    raise SystemExit(run('inversion'))\n")
    (runner / "report.py").write_text(
        "from dummy_tasks import run\nraise SystemExit(run('report'))\n")
    reference = state / "reference"
    put(reference / "meta.json", {"run_params": {}})
    (reference / "initial_model.npy").write_bytes(b"dummy reference, never loaded")
    inputs = state / "experiments" / "input" / "dummy"
    inputs.mkdir(parents=True)
    for name in run_case.INPUT_FILES:
        (inputs / name).write_bytes(b"small immutable input")
    manifest = {
        "runtime_root": str(runner.parent), "experiments_root": str(state / "experiments"),
        "experiment_id": "dummy", "reference_run": str(reference),
        "reference_sha256": {p.name: pipeline.sha(p) for p in reference.iterdir()},
        "input_sha256": {p.name: pipeline.sha(p) for p in inputs.iterdir()},
        "code_sha256": {p.relative_to(runner.parent).as_posix(): pipeline.sha(p)
                        for p in runner.glob("*.py")},
        "cases": pipeline.case_grid(), "resources": {"max_parallel_runs": 2},
    }
    for case in manifest["cases"]:
        (state / "cases" / case["id"]).mkdir(parents=True)
    put(state / "manifest.json", manifest)
    put(state / "control.json", {})
    return SimpleNamespace(state=state, manifest=manifest, runner=runner)


def records(snapshot):
    path = snapshot.state / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def run_supervisor(snapshot, *, env=None):
    fd = pipeline.acquire(snapshot.state)
    try:
        process = subprocess.Popen(
            [sys.executable, str(snapshot.runner / "pipeline.py"), "supervise",
             "--state", str(snapshot.state), "--lock-fd", str(fd)],
            cwd=snapshot.runner.parent, pass_fds=(fd,), start_new_session=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
        )
    finally:
        os.close(fd)
    try:
        out, err = process.communicate(timeout=10)
        return process.returncode, out + err
    finally:
        close_process(process)


@LINUX
def test_supervise_finite_grid9_and_parallel_limit(snapshot):
    code, output = run_supervisor(snapshot)
    assert code == 0, output
    status = pipeline.read(snapshot.state / "status.json")
    ids = [c["id"] for c in snapshot.manifest["cases"]]
    assert status["status"] == "complete" and status["stage"] == "report"
    assert status["active"] == []
    assert [r["id"] for r in status["finished_cases"][:1]] == ["forward"]
    assert {r["id"] for r in status["finished_cases"][1:-1]} == set(ids)
    assert status["finished_cases"][-1]["id"] == "report"
    assert all(r["exit_code"] == 0 and not r["timed_out"] for r in status["finished_cases"])
    events = records(snapshot)
    starts = [r["id"] for r in events if r["kind"] == "start"]
    assert len(starts) == 11
    assert starts[0] == "forward" and starts[-1] == "report"
    assert set(starts[1:-1]) == set(ids)
    assert next(r for r in events if r["kind"] == "computed")["id"] == "forward"
    running, peak = set(), 0
    for row in events:
        if row["id"] not in ids:
            continue
        if row["kind"] == "start":
            running.add(row["id"])
            peak = max(peak, len(running))
        elif row["kind"] == "end":
            running.remove(row["id"])
    assert not running and 1 <= peak <= 2
    assert all(row["sid"] == row["pid"] for row in events)
    fd = pipeline.acquire(snapshot.state)
    os.close(fd)


def test_forward_failure_prevents_all_inversions_and_report(snapshot):
    put(snapshot.state / "control.json", {"forward_exit": 9})
    code, output = run_supervisor(snapshot)
    assert code == 1, output
    status = pipeline.read(snapshot.state / "status.json")
    assert status["status"] == "failed" and status["active"] == []
    assert "Forward failed" in status["error"]
    assert [(r["id"], r["exit_code"]) for r in status["finished_cases"]] == [("forward", 9)]
    assert {r["id"] for r in records(snapshot)} == {"forward"}


def test_forward_timeout_terminates_stage_and_prevents_inversions(snapshot):
    put(snapshot.state / "control.json", {"hold_forward": True})
    code, output = run_supervisor(snapshot, env={**os.environ, "TEST_STAGE_TIMEOUT_S": ".1"})
    assert code == 1, output
    status = pipeline.read(snapshot.state / "status.json")
    assert status["status"] == "failed" and status["active"] == []
    assert len(status["finished_cases"]) == 1
    row = status["finished_cases"][0]
    assert row["id"] == "forward" and row["timed_out"] is True
    assert row["exit_code"] != 0
    assert {r["id"] for r in records(snapshot)} <= {"forward"}


def test_inversion_failure_continues_reports_and_resume_delegates(snapshot):
    failed = snapshot.manifest["cases"][0]["id"]
    put(snapshot.state / "control.json", {"fail_cases": [failed]})
    code, output = run_supervisor(snapshot)
    assert code == 1, output
    status = pipeline.read(snapshot.state / "status.json")
    assert status["status"] == "complete_with_failures"
    assert len(status["finished_cases"]) == 11
    assert [(r["id"], r["exit_code"]) for r in status["finished_cases"] if r["exit_code"]] == [(failed, 7)]
    assert status["finished_cases"][-1]["id"] == "report"
    assert (snapshot.state / "report_summary.json").exists()
    before = len(records(snapshot))
    put(snapshot.state / "control.json", {})
    code, output = run_supervisor(snapshot)
    assert code == 0, output
    resumed = records(snapshot)[before:]
    assert {r["id"] for r in resumed if r["kind"] == "reused"} == {
        "forward", *(c["id"] for c in snapshot.manifest["cases"] if c["id"] != failed)}
    assert {r["id"] for r in resumed if r["kind"] == "computed"} == {failed, "report"}
    assert pipeline.read(snapshot.state / "status.json")["status"] == "complete"


def test_report_failure_is_not_success(snapshot):
    put(snapshot.state / "control.json", {"report_exit": 6})
    code, output = run_supervisor(snapshot)
    assert code == 1, output
    status = pipeline.read(snapshot.state / "status.json")
    assert status["status"] == "complete_with_failures"
    assert status["finished_cases"][-1]["exit_code"] == 6


def supervisor_record(snapshot):
    path = snapshot.state / "status.json"
    return pipeline.read(path) if path.exists() else None


def cleanup_detached(snapshot):
    status = supervisor_record(snapshot)
    if status and pipeline.alive(status):
        os.kill(status["pid"], signal.SIGTERM)
        wait_until(lambda: not pipeline.alive(status))
    # Popen is internal to launch, so reap the direct child explicitly in tests.
    if status:
        try:
            os.waitpid(status["pid"], 0)
        except ChildProcessError:
            pass


@LINUX
@pytest.mark.parametrize("stop_early", [False, True], ids=["finish", "stop"])
def test_detached_launch_heartbeat_duplicate_and_stop(snapshot, monkeypatch, stop_early):
    put(snapshot.state / "control.json", {"hold_forward": True})
    initialize = Mock(return_value=snapshot.manifest)
    monkeypatch.setattr(pipeline, "initialize", initialize)
    try:
        pipeline.launch(snapshot.state)
        initialize.assert_called_once_with(snapshot.state)
        status = supervisor_record(snapshot)
        assert pipeline.alive(status) and status["active"]
        assert os.getsid(status["pid"]) == status["pid"]
        assert status["active"][0]["id"] == "forward"
        child = status["active"][0]
        with pytest.raises(RuntimeError, match="exclusive OS lock"):
            pipeline.launch(snapshot.state)
        assert initialize.call_count == 1
        if stop_early:
            monkeypatch.setattr(sys, "argv", ["pipeline.py", "stop", "--state", str(snapshot.state)])
            assert pipeline.main() == 0
        else:
            (snapshot.state / "release_forward").touch()
        final = wait_until(lambda: (s if (s := supervisor_record(snapshot))["status"] != "running" else None))
        assert final["status"] == ("interrupted" if stop_early else "complete")
        assert final["active"] == []
        wait_until(lambda: not pipeline.alive(child))
        assert not pipeline.alive(child)
    finally:
        cleanup_detached(snapshot)
    fd = pipeline.acquire(snapshot.state)
    os.close(fd)


@pytest.fixture
def forward_study(snapshot, monkeypatch):
    import numpy as np
    from projects.forward_modeling import experiments, solver
    from projects.forward_modeling.model import PointSet, VelocityGrid

    manifest = snapshot.manifest
    manifest["forward_spec"] = {"noise": {"relative_sigma": .01, "absolute_sigma_s": .01, "seed": 42}}
    reference = Path(manifest["reference_run"])
    initial = np.full((20, 7, 7), 5000.)
    np.save(reference / "initial_model.npy", initial)
    manifest["reference_sha256"] = {p.name: pipeline.sha(p) for p in reference.iterdir()}
    put(snapshot.state / "manifest.json", manifest)
    output = Path(manifest["experiments_root"]) / "output" / manifest["experiment_id"]
    metadata = {"config": {"refinement": 2, "source_radius_cells": 2},
                "noise": {"config": manifest["forward_spec"]["noise"]}}

    def publish(*args, **kwargs):
        put(output / "metadata.json", metadata)
        (output / "arrivals.csv").write_text("station_id,event_id,arrival_time_s\ns,e,0\n")
        return output

    model = VelocityGrid(np.full((2, 2, 2), 5000.), 1000.)
    stations = PointSet(("s",), np.array([[100., 100., 0.]]))
    xyz = np.full((20, 3), 100.)
    xyz[:, 2] = np.arange(20) * 10. + 100.
    xyz[3, 2], xyz[17, 2] = 1., 1999.
    events = PointSet(tuple(f"e{i}" for i in range(20)), xyz)
    load = create_autospec(experiments.load_inputs, return_value=(model, stations, events))
    run = create_autospec(experiments.run_experiment, side_effect=publish)
    convergence = create_autospec(solver.check_convergence, return_value={
        "max_abs_difference_s": .001, "max_absolute_time_difference_s": .002})
    monkeypatch.setattr(experiments, "load_inputs", load)
    monkeypatch.setattr(experiments, "run_experiment", run)
    monkeypatch.setattr(solver, "check_convergence", convergence)
    monkeypatch.setattr(solver, "compute_travel_times", Mock(side_effect=AssertionError("real solve forbidden")))
    monkeypatch.setitem(sys.modules, "run_case", run_case)
    config = object()
    build = create_autospec(run_case.build_config, return_value=config)
    runtime = SimpleNamespace(InversionConfig=object, prepare_inversion=Mock(return_value=SimpleNamespace(
        initial_model=SimpleNamespace(velocity=initial), event_ids=range(3780), station_ids=range(26),
        noise_sigmas=(.01, .01))))
    monkeypatch.setattr(run_case, "load_runtime", Mock(return_value=runtime))
    monkeypatch.setattr(run_case, "build_config", build)
    return SimpleNamespace(snapshot=snapshot, output=output, metadata=metadata, run=run, load=load,
                           convergence=convergence, runtime=runtime, config=config, publish=publish)


def test_forward_worker_api_selection_validation_and_resume(forward_study):
    import numpy as np
    f = forward_study
    state, manifest = f.snapshot.state, f.snapshot.manifest
    pipeline.worker(state)
    call = f.convergence.call_args
    model, stations, selected = call.args
    assert call.kwargs["config"].refinement == 2 and call.kwargs["workers"] == 16
    expected = sorted(set(np.linspace(0, 19, 16, dtype=int).tolist() + [3, 17]))
    assert selected.ids == tuple(f"e{i}" for i in expected)
    f.load.assert_called_once_with(Path(manifest["experiments_root"]), manifest["experiment_id"])
    args, kwargs = f.run.call_args
    assert args == (Path(manifest["experiments_root"]), manifest["experiment_id"])
    assert kwargs["config"].refinement == 2 and kwargs["config"].source_radius_cells == 2
    assert kwargs["workers"] == 32
    assert vars(kwargs["noise"]) == manifest["forward_spec"]["noise"]
    f.runtime.prepare_inversion.assert_called_once_with(manifest["experiment_id"], f.config,
                                                       experiments_root=Path(manifest["experiments_root"]))
    ledger = pipeline.read(state / "forward_complete.json")
    assert ledger["status"] == "complete"
    assert ledger["output_sha256"] == {name: pipeline.sha(f.output / name)
                                       for name in ("arrivals.csv", "metadata.json")}
    assert pipeline.read(state / "forward_convergence.json")["event_ids"] == list(selected.ids)
    pipeline.worker(state)
    assert f.run.call_count == f.convergence.call_count == f.load.call_count == 1
    assert f.runtime.prepare_inversion.call_count == 2  # resume still validates initial model


def test_forward_hash_failure_precedes_runtime_import(forward_study, monkeypatch):
    f = forward_study
    load_runtime = Mock(side_effect=AssertionError("runtime selected before hash validation"))
    monkeypatch.setattr(run_case, "load_runtime", load_runtime)
    (f.snapshot.runner / "report.py").write_text("changed snapshot")
    with pytest.raises(ValueError, match="Runtime SHA256 mismatch"):
        pipeline.worker(f.snapshot.state)
    load_runtime.assert_not_called()
    f.run.assert_not_called()
    assert not (f.snapshot.state / "forward_complete.json").exists()


def test_forward_exception_does_not_publish_completion(forward_study):
    f = forward_study
    f.run.side_effect = RuntimeError("synthetic forward failure")
    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        pipeline.worker(f.snapshot.state)
    f.runtime.prepare_inversion.assert_not_called()
    assert not (f.snapshot.state / "forward_complete.json").exists()


@pytest.mark.parametrize("key,value", [("max_abs_difference_s", float("nan")),
                                        ("max_absolute_time_difference_s", -.01)])
def test_forward_bad_convergence_never_runs_full_stage(forward_study, key, value):
    f = forward_study
    f.convergence.return_value[key] = value
    with pytest.raises(ValueError, match="Invalid convergence diagnostic"):
        pipeline.worker(f.snapshot.state)
    f.run.assert_not_called()
    assert not (f.snapshot.state / "forward_complete.json").exists()


@pytest.mark.parametrize("damage", ["config", "noise", "arrivals", "initial", "counts"])
def test_forward_resume_rejects_changed_or_invalid_outputs(forward_study, damage):
    f = forward_study
    pipeline.worker(f.snapshot.state)
    if damage in {"config", "noise"}:
        meta = pipeline.read(f.output / "metadata.json")
        if damage == "config":
            meta["config"]["refinement"] = 4
        else:
            meta["noise"]["config"]["seed"] = 43
        put(f.output / "metadata.json", meta)
    elif damage == "arrivals":
        (f.output / "arrivals.csv").write_text("changed")
    elif damage == "initial":
        f.runtime.prepare_inversion.return_value.initial_model.velocity[0, 0, 0] += 1
    else:
        f.runtime.prepare_inversion.return_value.event_ids = range(3)
    before = (f.snapshot.state / "forward_complete.json").read_bytes()
    with pytest.raises(ValueError):
        pipeline.worker(f.snapshot.state)
    assert (f.snapshot.state / "forward_complete.json").read_bytes() == before
    assert f.run.call_count == 1



def test_launch_accepts_matching_already_completed_heartbeat(snapshot, monkeypatch):
    monkeypatch.setattr(pipeline, "initialize", Mock(return_value=snapshot.manifest))

    def completed(*args, **kwargs):
        put(snapshot.state / "status.json", {"pid": 654321, "status": "complete", "active": []})
        return SimpleNamespace(pid=654321, returncode=0, poll=lambda: 0)

    monkeypatch.setattr(pipeline.subprocess, "Popen", completed)
    pipeline.launch(snapshot.state)



def test_supervise_preflight_releases_lock_on_error(snapshot, monkeypatch):
    fd = pipeline.acquire(snapshot.state)
    with monkeypatch.context() as patch:
        patch.setattr(pipeline.signal, "signal", Mock())
        patch.setattr(sys, "path", list(sys.path))
        patch.setitem(sys.modules, "run_case", run_case)
        (snapshot.runner / "report.py").write_text("tampered")
        try:
            with pytest.raises(ValueError, match="Runtime SHA256 mismatch"):
                pipeline.supervise(snapshot.state, fd)
            closed = False
            try:
                os.fstat(fd)
            except OSError:
                closed = True
            assert closed, "supervise leaked its lock fd on preflight failure"
        finally:
            try:
                os.close(fd)
            except OSError:
                pass


@LINUX

def test_terminate_escalates_for_descendants_after_leader_exits(tmp_path):
    ready = tmp_path / "grandchild.json"
    grandchild_code = (
        "import os,signal,time,json,sys; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "Path(sys.argv[1]).write_text(json.dumps({'pid':os.getpid()})); time.sleep(8)"
    )
    leader = subprocess.Popen(
        [sys.executable, "-c", "import subprocess,sys,time; "
         "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]]); time.sleep(8)",
         grandchild_code, str(ready)], start_new_session=True,
    )
    try:
        wait_until(ready.exists)
        child = {**pipeline.read(ready), "birth": None}
        child["birth"] = pipeline.identity(child["pid"])
        pipeline.terminate(leader)
        assert not pipeline.alive(child), "SIGTERM-ignoring descendant survived terminate()"
    finally:
        try:
            os.killpg(leader.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        leader.wait(timeout=5)
