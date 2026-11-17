"""Validate by default. Only explicit --launch starts the four detached workers.

A launch reservation is permanent, including on failure: inspect status/logs before
any manual recovery. Never automatically retry partially launched inversions.
"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from run import CASE_IDS, THREAD_ENV, read, require, sha, verify, write

STATE = Path(__file__).resolve().parent
PYTHON = "/mnt/disk01/egor/venv/bin/python"


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_all(manifest, env):
    reports = {}
    for case_id in CASE_IDS:
        completed = subprocess.run(
            [PYTHON, manifest["runner"], "--manifest", str(STATE / "manifest.json"),
             "--case-id", case_id, "--validate-only"],
            cwd=STATE, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=180)
        require(completed.returncode == 0,
                f"Validation failed: {case_id}\n{completed.stdout}\n{completed.stderr}")
        import json
        reports[case_id] = json.loads(completed.stdout)
        print(completed.stdout, end="", flush=True)
    report = dict(validated_at=now(), manifest_sha256=sha(STATE / "manifest.json"),
                  python=PYTHON, fresh_interpreters=True, cases=reports)
    write(STATE / "validation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--launch", action="store_true")
    group.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    manifest = read(STATE / "manifest.json")
    require(Path(manifest["state_dir"]).resolve() == STATE, "Wrong study location")
    for name, checksum in manifest["study_code_sha256"].items():
        verify(STATE / name, checksum)
    verify(manifest["runner"], manifest["runner_sha256"])
    env = dict(os.environ, **THREAD_ENV, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    # Do not inherit caller-controlled import paths or Python startup customizations.
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    if not args.launch:
        validate_all(manifest, env)
        print("Validated all four cases; no workers launched.")
        return

    # Atomic, exclusive study-wide reservation; it remains on ALL failure paths.
    write(STATE / ".launch.lock", dict(pid=os.getpid(), reserved_at=now()), exclusive=True)
    status = dict(status="preflight", launcher_pid=os.getpid(), started_at=now(),
                  manifest_sha256=sha(STATE / "manifest.json"), workers={})
    processes, logs = {}, {}
    try:
        write(STATE / "launch_status.json", status, exclusive=True)
        for case_id in CASE_IDS:
            case = manifest["cases"][case_id]
            directory = STATE / "cases" / case_id
            require(not Path(case["result_file"]).exists(), f"Case already reserved: {case_id}")
            require(not list(Path(case["config"]["runs_dir"]).glob(
                f"run_{case['config']['run_name']}_*")), f"Run name already exists: {case_id}")
            # Reserve every log before any worker starts, without truncating anything.
            logs[case_id] = (directory / "worker.log").open("xb", buffering=0)
        validate_all(manifest, env)
        require(sha(STATE / "manifest.json") == status["manifest_sha256"],
                "Manifest changed during preflight")
        status.update(status="launching")
        write(STATE / "launch_status.json", status)
        for case_id in CASE_IDS:
            command = [PYTHON, manifest["runner"], "--manifest", str(STATE / "manifest.json"),
                       "--case-id", case_id]
            process = subprocess.Popen(command, cwd=STATE, env=env, stdin=subprocess.DEVNULL,
                                       stdout=logs[case_id], stderr=subprocess.STDOUT,
                                       start_new_session=True, close_fds=True)
            processes[case_id] = process
            status["workers"][case_id] = dict(pid=process.pid, launched_at=now(), command=command,
                                              log=str(STATE / "cases" / case_id / "worker.log"))
            write(STATE / "launch_status.json", status)
            time.sleep(0.2)
            require(process.poll() is None, f"Early worker exit: {case_id}: {process.returncode}")
        # Detect immediate validation/import/startup failures across all four workers.
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            for case_id, process in processes.items():
                require(process.poll() is None,
                        f"Early worker exit: {case_id}: {process.returncode}")
                result = Path(manifest["cases"][case_id]["result_file"])
                if result.exists():
                    require(read(result)["status"] != "failed", f"Worker failed: {case_id}")
            time.sleep(0.2)
        status.update(status="launched", launch_finished_at=now())
        write(STATE / "launch_status.json", status)
        print("Detached workers launched; inspect case result.json and worker.log for progress.")
    except BaseException as error:
        # Fail closed: stop any siblings already spawned, record all known PIDs.
        status.update(status="failed", error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc(), failed_at=now())
        for case_id, process in processes.items():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    status["workers"][case_id]["cleanup"] = "SIGTERM sent to process group"
                except ProcessLookupError:
                    pass
            status["workers"][case_id]["returncode_at_failure"] = process.poll()
        write(STATE / "launch_status.json", status)
        raise
    finally:
        for stream in logs.values():
            stream.close()


if __name__ == "__main__":
    main()
