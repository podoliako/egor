"""Sequential frozen runner: python -B STATE/runner/pipeline.py --state STATE.

Manifest case.config is a dict with n_cycles; relative result.run_dir is resolved
against project_root. No numerical modules are imported. Exit codes: 0 success,
1 case failures, 2 lock contention, 130 interruption/supervisor error.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


HEARTBEAT_S = 5
GRACE_S = 30


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def stop_worker(process, heartbeat=lambda: None):
    """Reap the worker and kill surviving members of its private process group."""
    def send(sig):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
    send(signal.SIGTERM)
    deadline = time.monotonic() + GRACE_S
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                process.wait(timeout=min(HEARTBEAT_S, remaining))
                break
            except subprocess.TimeoutExpired:
                heartbeat()
    finally:
        # Also handles descendants surviving an already-exited group leader.
        send(signal.SIGKILL)
        process.wait()


def result_for(case, path, root):
    result = read(path)
    if result.get("status") == "failed":
        return result
    if result.get("status") != "completed":
        raise ValueError("Unfinished or unrecognized worker result; refusing rerun")
    run = root / result["run_dir"]
    artifacts = case.get("completion_artifacts")
    if artifacts is not None:
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError("completion_artifacts must be a nonempty list")
        for name in artifacts:
            if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Unsafe completion artifact")
            artifact = run / name
            if (not artifact.is_file() or artifact.stat().st_size == 0
                    or hashlib.sha256(artifact.read_bytes()).hexdigest()
                    != result.get("artifact_sha256", {}).get(name)):
                raise ValueError(f"Missing or changed completion artifact: {name}")
        return result
    cycles = case["config"]["n_cycles"]
    if not isinstance(cycles, int) or isinstance(cycles, bool) or cycles < 1:
        raise ValueError("config.n_cycles must be a positive integer")
    if not (run / "final_model.npy").is_file() or not all(
            (run / f"iter_{i}" / "complete.json").is_file() for i in range(cycles)):
        raise ValueError("Incomplete completed result: missing model/checkpoints")
    return result


def fail(path, error):
    try:
        result = read(path)
        if not isinstance(result, dict):
            result = {}
    except (OSError, ValueError):
        result = {}
    result.setdefault("error", str(error))
    result.update(status="failed", supervisor_error=str(error), finished_at=time.time())
    write(path, result)
    return result


def summary(state, cases, results):
    path = state / "summary.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "case_id", "status", "run_dir", "final_rms_m_s", "hard_control"))
        writer.writeheader()
        for case in cases:
            result = results.get(case["id"], {})
            writer.writerow(dict(case_id=case["id"], status=result.get("status", "pending"),
                                 run_dir=result.get("run_dir", ""),
                                 final_rms_m_s=result.get("final_rms_m_s", ""),
                                 hard_control=result.get("hard_control", case.get(
                                     "hard_control", case["config"].get("candidate_mode") == "hard"))))
    temporary.replace(path)


def run(state):
    state = Path(state).resolve()
    with (state / "queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Queue already locked", file=sys.stderr)
            return 2
        cases, results, handlers = [], {}, {}
        process = current = error = None
        interrupted = False
        started = time.time()

        def request_stop(signum, frame):
            nonlocal interrupted
            interrupted = True

        def publish(status="running"):
            completed = sum(r["status"] == "completed" for r in results.values())
            failed = sum(r["status"] == "failed" for r in results.values())
            write(state / "status.json", dict(
                status=status, current_case=current, completed=completed, failed=failed,
                pending=len(cases) - completed - failed - int(current is not None),
                total=len(cases), worker_pid=process.pid if process else None,
                supervisor_pid=os.getpid(), started_at=started, heartbeat=time.time(), error=error))

        try:
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                handlers[sig] = signal.signal(
                    sig, signal.SIG_IGN if sig == signal.SIGHUP else request_stop)
            manifest = read(state / "manifest.json")
            root = Path(manifest["project_root"])
            timeout = float(manifest.get("case_timeout_s", 14400))
            if not root.is_absolute() or not root.is_dir() or not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("Invalid project_root or case_timeout_s")
            candidate_cases = manifest["cases"]
            ids = [c["id"] for c in candidate_cases]
            if any(not isinstance(i, str) or i in ("", ".", "..") or Path(i).name != i for i in ids) or len(set(ids)) != len(ids):
                raise ValueError("Case ids must be unique directory names")
            cases = candidate_cases
            # Classify all old results up front so heartbeat counts include skipped cases.
            for case in cases:
                path = state / "cases" / case["id"] / "result.json"
                if path.exists() or path.with_name("run.log").exists():
                    try:
                        results[case["id"]] = result_for(case, path, root)
                    except Exception as exc:
                        results[case["id"]] = fail(path, exc)
            publish()
            for case in cases:
                if interrupted:
                    break
                if case["id"] in results:
                    continue
                current = case["id"]
                print(f"Starting {current} ({len(results) + 1}/{len(cases)})", flush=True)
                directory = state / "cases" / current
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / "result.json"
                try:
                    write(path, dict(status="running", started_at=time.time()))
                    env = dict(os.environ, **{name: "1" for name in (
                        "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")})
                    with (directory / "run.log").open("a") as log:
                        process = subprocess.Popen(
                            [sys.executable, "-B", str(Path(__file__).resolve().with_name("worker.py")),
                             "--state", str(state), "--case", current],
                            cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
                        deadline = time.monotonic() + timeout
                        publish()
                        while True:
                            if interrupted:
                                raise InterruptedError("Supervisor received TERM/INT")
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                raise TimeoutError(f"Worker exceeded {timeout:g} seconds")
                            try:
                                returncode = process.wait(timeout=min(HEARTBEAT_S, remaining))
                                break
                            except subprocess.TimeoutExpired:
                                publish()
                        if interrupted:
                            raise InterruptedError("Supervisor received TERM/INT")
                        if returncode:
                            raise RuntimeError(f"Worker exited with code {returncode}")
                        results[current] = result_for(case, path, root)
                except Exception as exc:
                    if process is not None:
                        stop_worker(process, publish)
                        process = None
                    results[current] = fail(path, exc)
                finally:
                    if process is not None:
                        stop_worker(process, publish)
                    process = current = None
                print(f"Finished {case['id']}: {results[case['id']]['status']}", flush=True)
                summary(state, cases, results)
                publish()
        except Exception as exc:
            interrupted, error = True, f"{type(exc).__name__}: {exc}"
        finally:
            try:
                if process is not None:
                    stop_worker(process, publish)
                current = process = None
                status = ("interrupted" if interrupted else "completed_with_failures"
                          if any(r["status"] == "failed" for r in results.values()) else "completed")
                publish(status)
                summary(state, cases, results)
            finally:
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)
        return 130 if interrupted else 1 if status == "completed_with_failures" else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path)
    return run(parser.parse_args().state)


if __name__ == "__main__":
    sys.exit(main())
