"""Freeze CURRENT live runtime and the previous p12 inputs; never launch inversions."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import shutil

from run import (CASE_IDS, EXPERIMENT_ID, INITIAL_SHA256, THREAD_ENV,
                 check_config, read, require, run_name, sha, verify, write)

STATE = Path(__file__).resolve().parent
REPOSITORY = STATE.parents[3]
SOURCE = STATE.parent / "geysers_realistic_previous_cov0_auto_parallel12"


def plan():
    previous = read(SOURCE / "manifest.json")
    require(previous["experiment_id"] == EXPERIMENT_ID, "Wrong previous experiment")
    for category in ("input", "output"):
        for name, checksum in previous[f"{category}_sha256"].items():
            verify(Path(previous["experiments_root"]) / category / EXPERIMENT_ID / name, checksum)
    verify(SOURCE / "initial_model.npz", INITIAL_SHA256)
    # Include every runtime module, package initializer and forward-model dependency;
    # intentionally exclude unrelated viewer/server, tests, studies and generated files.
    runtime = set(previous["runtime_sha256"])
    runtime.update(str(p.relative_to(REPOSITORY)) for p in
                   (REPOSITORY / "projects/EMTomo").glob("*.py")
                   if p.name not in {"server.py", "main.py", "benchmark_performance.py",
                                     "prepare_geysers_geometry.py"})
    runtime.add("projects/__init__.py")
    for package in ("projects/EMTomo/tomography", "projects/EMTomo/instruments",
                    "projects/forward_modeling"):
        runtime.update(str(p.relative_to(REPOSITORY)) for p in
                       (REPOSITORY / package).rglob("*.py") if "__pycache__" not in p.parts)
    hashes = {name: sha(REPOSITORY / name) for name in sorted(runtime)}
    manifest = {key: deepcopy(previous[key]) for key in
                ("experiment_id", "experiments_root", "input_sha256", "output_sha256",
                 "assumed_noise", "control_split")}
    manifest.update(schema_version=2, state_dir=str(STATE),
                    created_at=datetime.now(timezone.utc).isoformat(),
                    snapshot=str(STATE / "snapshot"), runtime_sha256=hashes,
                    runtime_policy="Full current live runtime, NOT a validator-only overlay",
                    previous_runtime_sha256=previous["runtime_sha256"],
                    runtime_changes={name: {"previous": previous["runtime_sha256"].get(name),
                                            "current": checksum}
                                     for name, checksum in hashes.items()
                                     if checksum != previous["runtime_sha256"].get(name)},
                    source_manifest={"path": str(SOURCE / "manifest.json"),
                                     "sha256": sha(SOURCE / "manifest.json")},
                    runner=str(STATE / "runner/run.py"), runner_sha256=sha(STATE / "run.py"),
                    study_code_sha256={name: sha(STATE / name)
                                       for name in ("run.py", "prepare.py", "launch.py")},
                    initial_model_file=str(STATE / "initial_model.npz"),
                    initial_model_sha256=INITIAL_SHA256,
                    hardware_policy=dict(n_workers_per_run=12, concurrent_runs=4, total_cpus=48,
                                         thread_environment=THREAD_ENV),
                    auto_stopping=dict(max_updates=20, min_updates=3, patience=2,
                                       min_relative_improvement=0.01, retain_stopping_update=True),
                    stopping_policy="At least 3 updates; patience 2; 1% relative control RMSE; "
                                    "retain stopping update; maximum 20 updates",
                    reuse_control_localization=True, event_ray_container="rays.npz schema 1",
                    python="/mnt/disk01/egor/venv/bin/python",
                    versions={name: version(name) for name in
                              ("numpy", "scipy", "numba", "scikit-fmm", "pykonal")}, cases={})
    for case_id in CASE_IDS:
        old = previous["cases"][case_id]
        result_path = SOURCE / "cases" / case_id / "result.json"
        result = read(result_path)
        require(result["status"] == "completed" and result["config"] == old["config"],
                f"Previous p12 case not completed/mismatched: {case_id}")
        config = dict(old["config"], n_cycles=20, run_name=run_name(case_id))
        check_config(case_id, config, old["config"])
        initial = deepcopy(old["initial_model_source"])
        if case_id.startswith("1d-"):
            require(initial["sha256"] == INITIAL_SHA256, "Wrong previous initial model")
            verify(initial["snapshot_file"], INITIAL_SHA256)
            initial["previous_snapshot_file"] = initial["snapshot_file"]
            initial["snapshot_file"] = str(STATE / "initial_model.npz")
        manifest["cases"][case_id] = dict(
            family=old["family"], mode=old["mode"], config=config,
            source_config=deepcopy(old["config"]), initial_model_source=initial,
            previous_run=dict(result_file=str(result_path), result_sha256=sha(result_path),
                              run_id=result["run_id"], run_dir=result["run_dir"], status="completed"),
            result_file=str(STATE / "cases" / case_id / "result.json"))
    return manifest


def prepare(manifest):
    for name in ("manifest.json", "snapshot", "runner", "cases", "initial_model.npz", ".prepare.lock"):
        require(not (STATE / name).exists(), f"Refusing duplicate/partial preparation: {name}")
    with (STATE / ".prepare.lock").open("x") as stream:
        stream.write(manifest["created_at"] + "\n")
    for name, checksum in manifest["runtime_sha256"].items():
        source = REPOSITORY / name
        verify(source, checksum)
        target = Path(manifest["snapshot"]) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as incoming, target.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
        verify(target, checksum)
    for source, target, checksum in (
        (SOURCE / "initial_model.npz", STATE / "initial_model.npz", INITIAL_SHA256),
        (STATE / "run.py", Path(manifest["runner"]), manifest["runner_sha256"]),
    ):
        target.parent.mkdir(parents=True, exist_ok=True)
        verify(source, checksum)
        with source.open("rb") as incoming, target.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
        verify(target, checksum)
    for case_id in CASE_IDS:
        (STATE / "cases" / case_id).mkdir(parents=True)
    write(STATE / "runner/manifest_path.json", {"manifest": str(STATE / "manifest.json")}, exclusive=True)
    write(STATE / "manifest.json", manifest, exclusive=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    manifest = plan()
    if not args.check_only:
        prepare(manifest)
    print(f"{'Checked' if args.check_only else 'Prepared'} {STATE / 'manifest.json'}")
    print(f"{len(manifest['runtime_sha256'])} current source files; "
          f"{len(manifest['runtime_changes'])} changed/added versus previous p12; no launches")


if __name__ == "__main__":
    main()
