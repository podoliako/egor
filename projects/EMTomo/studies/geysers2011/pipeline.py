"""Finite detached Geysers study: launch / status / stop / resume via launch.

Stdlib only until a worker explicitly selects the immutable runtime snapshot.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
DEFAULT_STATE = REPO / "projects/EMTomo/runs/_geysers2011_queue"
PYTHON = Path(sys.executable)
STAGE_TIMEOUT_S = 72 * 3600


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(pid):
    """Linux process birth token prevents signaling a recycled PID."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if text[0] == "Z" else text[19]
    except (OSError, IndexError):
        return None


def alive(item):
    return bool(item and item.get("birth") and identity(item["pid"]) == item["birth"])


def acquire(state):
    state.mkdir(parents=True, exist_ok=True)
    fd = os.open(state / "queue.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise RuntimeError("Queue already running (exclusive OS lock is held)")
    previous = read(state / "status.json") if (state / "status.json").exists() else {}
    if any(alive(child) for child in previous.get("active", [])):
        os.close(fd)
        raise RuntimeError("An earlier queue child is still alive; do not start a duplicate")
    return fd


def case_grid():
    pairs = [(0.5, 1)] + [(lam, power) for lam in (0.1, 0.5, 1.0)
                           for power in (0, 1, 2) if (lam, power) != (0.5, 1)]
    return [{"id": f"lam{lam:g}-cov{power}", "lambda_reg": lam,
             "coverage_damping_power": power, "run_name": f"gy11-hard-c1000-lam{lam:g}-cov{power}",
             "is_reference": (lam, power) == (0.5, 1)} for lam, power in pairs]


def initialize(state):
    """Create inputs and a small source-only snapshot once; never rewrite a manifest."""
    sys.path.insert(0, str(REPO))
    from projects.EMTomo.studies.geysers2011.prepare import prepare_experiment
    from projects.EMTomo.studies.geysers2011.run_case import validate_manifest, verify_hashes
    manifest_path = state / "manifest.json"
    if manifest_path.exists():
        manifest = read(manifest_path)
        validate_manifest(manifest)
        verify_hashes(manifest)
        return manifest
    dataset = prepare_experiment()
    runtime = state / "runtime"
    if runtime.exists():
        raise RuntimeError(f"Unpublished runtime exists: inspect {runtime}; refusing overwrite")
    runtime.mkdir()
    sources = [REPO / "projects/__init__.py"]
    emtomo = REPO / "projects/EMTomo"
    sources += list(emtomo.glob("*.py"))
    for folder in (emtomo / "instruments", emtomo / "tomography", REPO / "projects/forward_modeling"):
        sources += list(folder.glob("*.py"))
    mapping = {}
    for source in sources:
        relative = source.relative_to(REPO)
        destination = runtime / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = (subprocess.check_output(["git", "--no-pager", "show", "c381cfd:projects/EMTomo/eikonal.py"], cwd=REPO)
                   if relative.as_posix() == "projects/EMTomo/eikonal.py" else source.read_bytes())
        destination.write_bytes(payload)
        mapping[relative.as_posix()] = sha(destination)
    # Separate runner directory avoids accidentally shadowing production modules.
    for name in ("pipeline.py", "run_case.py", "report.py", "prepare.py"):
        destination = runtime / "runner" / name
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes((HERE / name).read_bytes())
        mapping[f"runner/{name}"] = sha(destination)
    sys.path.insert(0, str(emtomo))
    from config import InversionConfig
    references = list((emtomo / "runs").glob("run_fl-hard-c1000-*20261013_200119"))
    if len(references) != 1:
        raise ValueError(f"Expected one reference hard1000, got {references}")
    reference = references[0]
    params = read(reference / "meta.json")["run_params"]
    allowed = {field.name for field in fields(InversionConfig)}
    baseline = {key: value for key, value in params.items() if key in allowed}
    from projects.EMTomo.experiment_runner import initial_velocity
    import numpy as np
    initial = initial_velocity(InversionConfig(**baseline), (20, 7, 7))
    expected = np.load(reference / "initial_model.npy", allow_pickle=False)
    if initial.dtype != expected.dtype or initial.tobytes() != expected.tobytes():
        raise ValueError("Baseline starting model differs from historical hard1000")
    manifest = {
        "schema_version": 1, "created_utc": now(), "experiment_id": dataset["experiment_id"],
        "experiments_root": str(REPO / "projects/forward_modeling/experiments"),
        "runs_dir": str(emtomo / "runs"), "runtime_root": str(runtime),
        "reference_run": str(reference), "baseline_config": baseline,
        "forward_implementation": "saved_hard1000_legacy_sphere",
        "resources": {"forward_workers": 32, "inversion_workers_per_run": 12,
                      "blas_threads_per_run": 12, "max_parallel_runs": 2},
        "dataset": dataset, "input_sha256": {name: sha(Path(dataset["input_dir"]) / name)
                        for name in ("model.npz", "stations.csv", "events.csv", "generation.json")},
        "code_sha256": mapping, "cases": case_grid(),
        "reference_sha256": {name: sha(reference / name) for name in ("meta.json", "initial_model.npy")},
        "forward_spec": {"refinement": 2, "source_radius_cells": 2,
                         "noise": {"relative_sigma": 0.01, "absolute_sigma_s": 0.01, "seed": 42},
                         "convergence": "16 evenly spaced events plus shallowest/deepest; r2 versus r4; diagnostic only"},
    }
    validate_manifest(manifest)
    verify_hashes(manifest)
    for case in manifest["cases"]:
        (state / "cases" / case["id"]).mkdir(parents=True, exist_ok=True)
    atomic(manifest_path, manifest)
    return manifest


def worker(state):
    """Numerical forward stage runs in a fresh interpreter from the snapshot."""
    from run_case import verify_hashes, load_runtime, build_config, verify_initial
    manifest = read(state / "manifest.json")
    verify_hashes(manifest)
    runtime = load_runtime(manifest)
    import numpy as np
    from projects.forward_modeling.experiments import load_inputs, run_experiment
    from projects.forward_modeling.model import ForwardConfig, PointSet
    from projects.forward_modeling.noise import NoiseConfig
    from projects.forward_modeling.solver import check_convergence
    root, experiment = Path(manifest["experiments_root"]), manifest["experiment_id"]
    output = root / "output" / experiment
    ledger = state / "forward_complete.json"
    if not output.exists():
        model, stations, events = load_inputs(root, experiment)
        indices = sorted(set(np.linspace(0, len(events.ids) - 1, 16, dtype=int).tolist()
                             + [int(np.argmin(events.coordinates_m[:, 2])), int(np.argmax(events.coordinates_m[:, 2]))]))
        selection = PointSet(tuple(events.ids[i] for i in indices), events.coordinates_m[indices])
        print(f"Forward convergence control: {len(indices)} catalog events, r=2 versus r=4", flush=True)
        control = check_convergence(model, stations, selection, config=ForwardConfig(refinement=2), workers=16)
        for key in ("max_abs_difference_s", "max_absolute_time_difference_s"):
            if not np.isfinite(control[key]) or control[key] < 0:
                raise ValueError(f"Invalid convergence diagnostic: {key}")
        atomic(state / "forward_convergence.json", {"event_ids": list(selection.ids), "result": control,
                                                   "policy": "diagnostic only; not a guaranteed error bound"})
        print(json.dumps(control), flush=True)
        print("Full synthetic arrivals: 3780 events x 26 stations, refinement=2, workers=32", flush=True)
        run_experiment(root, experiment, config=ForwardConfig(refinement=2), workers=32,
                       noise=NoiseConfig(relative_sigma=.01, absolute_sigma_s=.01, seed=42))
    meta = read(output / "metadata.json")
    if meta["config"] != {"refinement": 2, "source_radius_cells": 2}:
        raise ValueError("Existing forward configuration mismatch")
    if meta["noise"]["config"] != manifest["forward_spec"]["noise"]:
        raise ValueError("Existing forward noise configuration mismatch")
    hashes = {name: sha(output / name) for name in ("arrivals.csv", "metadata.json")}
    if ledger.exists() and read(ledger)["output_sha256"] != hashes:
        raise ValueError("Forward outputs changed since validation")
    reference = Path(manifest["reference_run"])
    config = build_config(manifest, manifest["cases"][0], runtime.InversionConfig, read(reference / "meta.json"))
    data = runtime.prepare_inversion(experiment, config, experiments_root=root)
    verify_initial(data.initial_model.velocity, reference / "initial_model.npy", np)
    if len(data.event_ids) != 3780 or len(data.station_ids) != 26 or data.noise_sigmas != (.01, .01):
        raise ValueError("Prepared experiment counts/noise differ")
    verify_hashes(manifest)
    atomic(ledger, {"status": "complete", "validated_utc": now(), "output_sha256": hashes})
    print("Forward complete and validated; initial model bitwise identical", flush=True)


def terminate(process):
    """Terminate a whole session, including FMM process-pool descendants."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    # Reaping the leader does not imply that its descendants have exited.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=15)
    time.sleep(.05)


def supervise(state, lock_fd):
    os.set_inheritable(lock_fd, False)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    def stop(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")
    signal.signal(signal.SIGTERM, stop)
    try:
        manifest = read(state / "manifest.json")
        sys.path.insert(0, str(Path(manifest["runtime_root"]) / "runner"))
        from run_case import verify_hashes
        verify_hashes(manifest)
        for name, digest in manifest["reference_sha256"].items():
            if sha(Path(manifest["reference_run"]) / name) != digest:
                raise ValueError(f"Reference changed: {name}")
        environment = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": manifest["runtime_root"]}
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[name] = "1"
        runtime = Path(manifest["runtime_root"])
    except BaseException as error:
        try:
            atomic(state / "status.json", {"status": "failed", "stage": "preflight", "pid": os.getpid(),
                   "birth": identity(os.getpid()), "active": [], "error": str(error), "updated_utc": now()})
        finally:
            os.close(lock_fd)
        raise
    active = []
    status = {"status": "running", "stage": "forward", "pid": os.getpid(), "birth": identity(os.getpid()),
              "started_utc": now(), "active": [], "finished_cases": [], "manifest": str(state / "manifest.json")}
    def publish():
        status["updated_utc"] = now()
        status["active"] = [{"id": item["id"], "pid": item["process"].pid,
                              "birth": item["birth"], "log": str(item["log"])} for item in active]
        atomic(state / "status.json", status)
    def spawn(identifier, command, log):
        handle = log.open("a")
        handle.write(f"\n--- {now()} {identifier} ---\n")
        handle.flush()
        process = subprocess.Popen(command, cwd=runtime, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=handle, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        active.append({"id": identifier, "process": process, "birth": identity(process.pid),
                       "started": time.monotonic(), "handle": handle, "log": log})
        publish()
    def wait_one():
        while True:
            for item in list(active):
                process = item["process"]
                timeout = time.monotonic() - item["started"] > STAGE_TIMEOUT_S
                if timeout:
                    terminate(process)
                code = process.poll()
                if code is not None:
                    terminate(process)
                    active.remove(item)
                    item["handle"].close()
                    result = {"id": item["id"], "exit_code": code, "timed_out": timeout, "finished_utc": now()}
                    print(json.dumps(result), flush=True)
                    status["finished_cases"].append(result)
                    publish()
                    return result
            publish()
            time.sleep(5)
    try:
        publish()
        spawn("forward", [str(PYTHON), str(runtime / "runner/pipeline.py"), "forward", "--state", str(state)], state / "forward.log")
        if wait_one()["exit_code"] != 0:
            raise RuntimeError("Forward failed; inversions not started. See forward.log")
        status["stage"] = "inversions"
        pending = list(manifest["cases"])
        while pending or active:
            while pending and len(active) < manifest["resources"]["max_parallel_runs"]:
                case = pending.pop(0)
                spawn(case["id"], [str(PYTHON), str(runtime / "runner/run_case.py"), "--manifest", str(state / "manifest.json"),
                                   "--case-id", case["id"]], state / "cases" / case["id"] / "run.log")
            wait_one()
        status["stage"] = "report"
        spawn("report", [str(PYTHON), str(runtime / "runner/report.py"), "--manifest", str(state / "manifest.json")], state / "report.log")
        wait_one()
        status["status"] = "complete" if all(row["exit_code"] == 0 for row in status["finished_cases"]) else "complete_with_failures"
        status["finished_utc"] = now()
    except BaseException as error:
        status.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error), finished_utc=now())
        traceback.print_exc()
    finally:
        for item in active:
            terminate(item["process"])
            item["handle"].close()
        active.clear()
        publish()
        os.close(lock_fd)
    return 0 if status["status"] == "complete" else 1


def launch(state):
    fd = acquire(state)
    try:
        manifest = initialize(state)
        runtime = Path(manifest["runtime_root"])
        with (state / "queue.log").open("a") as stream:
            process = subprocess.Popen([str(PYTHON), str(runtime / "runner/pipeline.py"), "supervise", "--state", str(state),
                                        "--lock-fd", str(fd)], cwd=runtime, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                                       close_fds=True, pass_fds=(fd,))
        # Wait for a real heartbeat, not merely successful Popen.
        for _ in range(100):
            if (state / "status.json").exists():
                status = read(state / "status.json")
                if status.get("pid") == process.pid:
                    if status.get("active") or status.get("status") in {"complete", "complete_with_failures"}:
                        print(json.dumps(status, indent=2))
                        return
            if process.poll() is not None:
                raise RuntimeError(f"Queue exited at startup ({process.returncode}); inspect queue.log")
            time.sleep(.1)
        raise RuntimeError(f"Queue PID {process.pid} started but no child heartbeat yet; inspect status, do not relaunch blindly")
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "status", "stop", "supervise", "forward"))
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--lock-fd", type=int)
    args = parser.parse_args()
    state = args.state.resolve()
    if args.action == "launch":
        launch(state)
    elif args.action == "status":
        status = read(state / "status.json")
        status["scheduler_alive"] = alive(status)
        print(json.dumps(status, indent=2))
    elif args.action == "stop":
        status = read(state / "status.json")
        if alive(status):
            os.kill(status["pid"], signal.SIGTERM)
            print("Stop requested; supervisor will terminate its child process groups")
        else:
            raise RuntimeError("Supervisor not alive; inspect active child records before recovery")
    elif args.action == "forward":
        worker(state)
    else:
        if args.lock_fd is None:
            parser.error("supervise requires inherited --lock-fd; use launch")
        return supervise(state, args.lock_fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
