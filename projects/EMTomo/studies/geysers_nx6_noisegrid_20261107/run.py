"""Fixed-likelihood soft/hard noise grid: plan, check, then finite background queue.

Actual observation sigmas vary. All ten inversions explicitly retain the same
likelihood sigmas, paired seed, numerical settings and homogeneous start.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback


BASE = Path(__file__).resolve().parents[2]
STATE = Path(__file__).resolve().parent
REPO = BASE.parents[1]
ASSUMED = dict(relative_sigma=0.01, absolute_sigma_s=0.05, model_sigma_s=0.02)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verify(root, hashes):
    for name, checksum in hashes.items():
        if sha(Path(root) / name) != checksum:
            raise ValueError(f"Frozen source/input changed: {Path(root) / name}")


def case_config(reference, level_id, mode):
    sys.path.insert(0, str(BASE))
    from config import InversionConfig

    original = InversionConfig(**{f.name: reference[f.name] for f in fields(InversionConfig)
                                  if f.name in reference})
    config = replace(original, n_cycles=12, candidate_mode=mode,
                     weight_noise_relative_sigma=ASSUMED["relative_sigma"],
                     weight_noise_absolute_sigma_s=ASSUMED["absolute_sigma_s"],
                     weight_model_sigma_s=ASSUMED["model_sigma_s"],
                     runs_dir=str(BASE / "runs"), save_runs=True,
                     run_name=f"geysers-nx6-noise-{level_id}-{mode}")
    if (config.n_candidates, config.weights_top_n, config.weights_min_distance) != (50, 50, 7):
        raise ValueError("Expected the previous K50/top50/dmin7 protocol")
    return asdict(config)


def plan():
    if (STATE / "manifest.json").exists():
        raise FileExistsError("Queue already planned; refusing overwrite")
    data = read(STATE / "data_manifest.json")
    references = {}
    for mode in ("soft", "hard"):
        paths = list((BASE / "runs").glob(f"run_nx6-bottom-blobs-{mode}50-dmin7-12iter-*/meta.json"))
        path = max(paths, key=lambda p: p.stat().st_mtime)
        references[mode] = {"path": str(path), "sha256": sha(path), "params": read(path)["run_params"]}
    cases = []
    by_id = {level["id"]: level for level in data["levels"]}
    for level_id in ("x1", "x0p5", "x2", "x0p75", "x1p5"):
        level = by_id[level_id]
        pair = []
        for mode in ("soft", "hard"):
            config = case_config(references[mode]["params"], level_id, mode)
            identifier = f"{level_id}-{mode}"
            (STATE / "cases" / identifier).mkdir(parents=True, exist_ok=False)
            case = dict(id=identifier, level_id=level_id, mode=mode,
                        multiplier=level["multiplier"], experiment_id=level["experiment_id"],
                        actual_noise=level["actual_noise"], config=config,
                        input_sha256=level["input_sha256"], output_sha256=level["output_sha256"])
            cases.append(case)
            pair.append(config)
        differences = {k for k in pair[0] if pair[0][k] != pair[1][k]}
        if differences != {"candidate_mode", "run_name"}:
            raise ValueError(f"Unmatched hard/soft settings: {differences}")
    snapshot = STATE / "snapshot"
    snapshot.mkdir(exist_ok=False)
    sources = [BASE / name for name in ("config.py", "eikonal.py", "experiment_data.py",
                                        "experiment_runner.py", "interpolation.py", "raytracing.py",
                                        "velocity_model.py")]
    for directory in (BASE / "instruments", BASE / "tomography"):
        sources.extend(directory.rglob("*.py"))
    sources.extend((BASE.parent / "forward_modeling").glob("*.py"))
    if (REPO / "projects/__init__.py").exists():
        sources.append(REPO / "projects/__init__.py")
    runtime_hashes = {}
    for source in sorted(sources):
        relative = source.relative_to(REPO)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        runtime_hashes[str(relative)] = sha(target)
    runner = STATE / "runner"
    runner.mkdir(exist_ok=False)
    shutil.copyfile(Path(__file__), runner / "run.py")
    manifest = dict(schema_version=1, created_at=now(), state_dir=str(STATE),
                    experiments_root=data["experiments_root"],
                    data_manifest_sha256=sha(STATE / "data_manifest.json"),
                    snapshot=str(snapshot), runtime_sha256=runtime_hashes,
                    runner=str(runner / "run.py"), runner_sha256=sha(runner / "run.py"),
                    references=references, assumed_noise=ASSUMED, cases=cases,
                    max_parallel_runs=4, case_timeout_s=86400, queue_timeout_s=259200,
                    versions={name: version(name) for name in
                              ("numpy", "scipy", "numba", "scikit-fmm", "pykonal")},
                    policy={"actual_noise_only_varies": True, "paired_noise_seed": 42,
                            "likelihood_noise_fixed": True, "temperature_recalibrated": False,
                            "checkpoint": 12, "event_locations": "E-step before each M-step"})
    with (STATE / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"cases": len(cases), "max_parallel": 4, "assumed_noise": ASSUMED}, indent=2))


def worker(manifest_path, case_id, validate_only=False):
    manifest = read(manifest_path)
    state = Path(manifest["state_dir"])
    case = next(c for c in manifest["cases"] if c["id"] == case_id)
    if sha(state / "data_manifest.json") != manifest["data_manifest_sha256"]:
        raise ValueError("Noise-data manifest changed")
    if sha(manifest["runner"]) != manifest["runner_sha256"]:
        raise ValueError("Frozen runner changed")
    verify(manifest["snapshot"], manifest["runtime_sha256"])
    root = Path(manifest["experiments_root"])
    verify(root / "input" / case["experiment_id"], case["input_sha256"])
    verify(root / "output" / case["experiment_id"], case["output_sha256"])
    sys.path[:0] = [str(Path(manifest["snapshot"]) / "projects/EMTomo"), manifest["snapshot"]]
    import numpy as np
    import config as config_module
    import experiment_runner
    from tomography.tomography_em import run_em, warm_up_jit
    from tomography.tomography_logging import TomographyLogger

    for module in (config_module, experiment_runner):
        if not Path(module.__file__).resolve().is_relative_to(Path(manifest["snapshot"])):
            raise ValueError("Worker imported a non-frozen numerical module")
    config = config_module.InversionConfig(**case["config"])
    prepared = experiment_runner.prepare_inversion(case["experiment_id"], config, root)
    noise = experiment_runner.resolve_pick_noise(config, prepared.noise_sigmas)
    if asdict(noise) != manifest["assumed_noise"]:
        raise ValueError("Likelihood sigmas changed with the actual noise level")
    if prepared.arrivals_table.shape != (3471, 31):
        raise ValueError("Unexpected Geysers catalog")
    if validate_only:
        print(f"Validated {case_id}: actual={case['actual_noise']}, assumed={asdict(noise)}", flush=True)
        return
    result_path = state / "cases" / case_id / "result.json"
    if result_path.exists():
        raise FileExistsError("Case already attempted; refusing rerun")
    result = dict(status="running", case_id=case_id, pid=os.getpid(), started_at=now(),
                  actual_noise=case["actual_noise"], assumed_noise=asdict(noise))
    atomic(result_path, result)
    try:
        warm_up_jit()
        logger = TomographyLogger(base_dir=config.runs_dir, run_name=config.run_name,
                                  run_version=config.run_version,
                                  run_tags=dict(level=case["level_id"], mode=case["mode"],
                                                ncand=50, topn=50, dmin=7))
        result.update(run_id=logger.run_id, run_dir=str(logger.run_dir))
        atomic(result_path, result)
        run_em(config, prepared.initial_model, prepared.arrivals_table, prepared.station_locs,
               noise, reference_model=prepared.reference_model, logger=logger,
               source_experiment={"id": case["experiment_id"],
                                  "model_sha256": case["input_sha256"]["model.npz"],
                                  "noise_grid": {"manifest": str(manifest_path),
                                                 "multiplier": case["multiplier"],
                                                 "assumed_noise": asdict(noise)}})
        final = np.load(logger.run_dir / "final_model.npy", allow_pickle=False)
        if (final.shape != prepared.initial_model.shape or not np.isfinite(final).all()
                or np.any(final <= 0)
                or len(list(logger.run_dir.glob("iter_*/complete.json"))) != 12):
            raise ValueError("Incomplete or invalid final inversion")
        result.update(status="completed", final_model_sha256=sha(logger.run_dir / "final_model.npy"))
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
        raise
    finally:
        result["finished_at"] = now()
        atomic(result_path, result)


def kill_group(item):
    try:
        os.killpg(item["process"].pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        item["process"].wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(item["process"].pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    item["process"].wait(timeout=5)


def supervise():
    manifest_path = STATE / "manifest.json"
    manifest = read(manifest_path)
    with (STATE / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (STATE / "status.json").exists():
            raise FileExistsError("Queue already attempted; no implicit resume")
        status = dict(state="running", pid=os.getpid(), started_at=now(),
                      cases={c["id"]: {"state": "queued"} for c in manifest["cases"]})
        pending = list(manifest["cases"])
        active = []
        started = time.monotonic()
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
        env.update({name: "1" for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                          "MKL_NUM_THREADS", "NUMBA_NUM_THREADS",
                                          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")})

        def publish():
            status["updated_at"] = now()
            for case_id, record in status["cases"].items():
                path = STATE / "cases" / case_id / "result.json"
                if path.exists():
                    result = read(path)
                    record.update({k: result[k] for k in ("run_id", "run_dir", "error") if k in result})
                    if result.get("run_dir"):
                        record["completed_iterations"] = len(list(Path(result["run_dir"]).glob("iter_*/complete.json")))
            atomic(STATE / "status.json", status)

        def interrupt(signum, frame):
            raise KeyboardInterrupt(f"Queue received signal {signum}")

        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        try:
            publish()
            while pending or active:
                if time.monotonic() - started > manifest["queue_timeout_s"]:
                    raise TimeoutError("Overall queue deadline exceeded")
                for item in list(active):
                    process = item["process"]
                    timed_out = time.monotonic() - item["started"] > manifest["case_timeout_s"]
                    if timed_out:
                        kill_group(item)
                    if process.poll() is not None:
                        result_path = STATE / "cases" / item["id"] / "result.json"
                        result = read(result_path) if result_path.exists() else {}
                        completed = process.returncode == 0 and result.get("status") == "completed"
                        record = status["cases"][item["id"]]
                        record.update(state="completed" if completed else "failed", finished_at=now(),
                                      returncode=process.returncode, timed_out=timed_out)
                        if not completed:
                            kill_group(item)
                        item["log"].close()
                        active.remove(item)
                        print(item["id"], record["state"], flush=True)
                while pending and len(active) < manifest["max_parallel_runs"]:
                    case = pending.pop(0)
                    log = (STATE / "cases" / case["id"] / "run.log").open("x")
                    process = subprocess.Popen(
                        [sys.executable, "-u", manifest["runner"], "worker", "--manifest", str(manifest_path),
                         "--case-id", case["id"]], cwd=BASE, env=env, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    active.append(dict(id=case["id"], process=process, log=log, started=time.monotonic()))
                    status["cases"][case["id"]].update(state="running", pid=process.pid, started_at=now())
                    print("Started", case["id"], "PID", process.pid, flush=True)
                publish()
                if active:
                    time.sleep(5)
            status["state"] = ("completed" if all(r["state"] == "completed" for r in status["cases"].values())
                               else "completed_with_failures")
        except BaseException as error:
            status.update(state="failed", error=f"{type(error).__name__}: {error}")
            for item in active:
                kill_group(item)
                item["log"].close()
                status["cases"][item["id"]].update(state="cancelled")
            for case in pending:
                status["cases"][case["id"]]["state"] = "cancelled"
            traceback.print_exc()
            raise
        finally:
            status["finished_at"] = now()
            publish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "check", "supervise", "worker"))
    parser.add_argument("--manifest", type=Path, default=STATE / "manifest.json")
    parser.add_argument("--case-id")
    args = parser.parse_args()
    if args.command == "plan":
        plan()
    elif args.command == "check":
        for case in read(args.manifest)["cases"]:
            worker(args.manifest, case["id"], validate_only=True)
    elif args.command == "worker":
        worker(args.manifest, args.case_id)
    else:
        supervise()


if __name__ == "__main__":
    main()
