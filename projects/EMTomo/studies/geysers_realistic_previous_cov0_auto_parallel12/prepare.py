"""Freeze four matched p12 cases once, without launching any workers.

--check-only checks the source plan in memory, without creating study artifacts.
The parent owns control_parallel_equivalence.json; this script never touches it.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil

from run import (CASE_IDS, CONTROL, EXPERIMENT_ID, INITIAL_SHA256, NOISE, THREAD_ENV,
                 VALIDATOR, check_config, read, require, run_name, sha, verify, write)

STATE = Path(__file__).resolve().parent
REPOSITORY = STATE.parents[3]
SOURCES = {
    "homogeneous": STATE.parent / "geysers_realistic_previous_cov0_auto" / "manifest.json",
    "1d": STATE.parent / "geysers_realistic_previous_1d_cov0_auto" / "manifest.json",
}


def plan():
    """Verify all original bytes before copying anything; return a standalone manifest."""
    originals = {family: read(path) for family, path in SOURCES.items()}
    baseline = originals["homogeneous"]
    common = ("experiment_id", "experiments_root", "runtime_sha256", "input_sha256",
              "output_sha256", "assumed_noise", "control_split", "stopping_policy")
    require(baseline["experiment_id"] == EXPERIMENT_ID, "Wrong source experiment")
    require(baseline["assumed_noise"] == NOISE and baseline["control_split"] == CONTROL,
            "Wrong original noise/control descriptors")
    for family, original in originals.items():
        require(all(original[key] == baseline[key] for key in common),
                f"Original manifests disagree: {family}")
        verify(original["runner"], original["runner_sha256"])
        for name, checksum in original["runtime_sha256"].items():
            require(Path(name).suffix == ".py" and not Path(name).is_absolute()
                    and ".." not in Path(name).parts, f"Not a source path: {name}")
            verify(Path(original["snapshot"]) / name, checksum)
    for category in ("input", "output"):
        for name, checksum in baseline[f"{category}_sha256"].items():
            verify(Path(baseline["experiments_root"]) / category / EXPERIMENT_ID / name, checksum)
    initial = originals["1d"]["initial_model_source"]
    require(initial["sha256"] == INITIAL_SHA256, "Unexpected original 1D model")
    verify(initial["snapshot_file"], INITIAL_SHA256)
    # The historical original path is provenance only; the already-frozen file is authoritative.
    overlay = REPOSITORY / VALIDATOR
    overlay_sha = sha(overlay)
    require(overlay_sha != baseline["runtime_sha256"][VALIDATOR],
            "Live validator is still the old serial source")
    manifest = {key: deepcopy(baseline[key]) for key in common}
    manifest.update(
        schema_version=1, state_dir=str(STATE), created_at=datetime.now(timezone.utc).isoformat(),
        snapshot=str(STATE / "snapshot"), original_runtime_sha256=deepcopy(baseline["runtime_sha256"]),
        runner=str(STATE / "runner" / "run.py"), runner_sha256=sha(STATE / "run.py"),
        runner_manifest_pointer=str(STATE / "runner" / "manifest_path.json"),
        validator_source=dict(path=str(overlay), sha256=overlay_sha,
                              original_sha256=baseline["runtime_sha256"][VALIDATOR],
                              policy="Only live source overlay; fork pool with ordered reduction and progress"),
        hardware_policy=dict(n_workers_per_run=12, concurrent_runs=4, total_cpus=48,
                             thread_environment=THREAD_ENV,
                             launch_policy="Parent launches four detached Popen workers; no queue"),
        auto_stopping=dict(max_updates=7, min_updates=3, patience=2,
                           min_relative_improvement=0.01, retain_stopping_update=True),
        source_manifests={family: dict(path=str(path), sha256=sha(path))
                          for family, path in SOURCES.items()},
        versions=deepcopy(baseline.get("versions", {})), cases={})
    manifest["runtime_sha256"][VALIDATOR] = overlay_sha
    for case_id in CASE_IDS:
        family, mode = case_id.rsplit("-", 1)
        original = originals[family]
        old_status_path = SOURCES[family].parent / f"{mode}_status.json"
        old_status = read(old_status_path)
        require(old_status.get("status", old_status.get("state")) == "stopped",
                f"Old session not stopped: {case_id}")
        require(old_status["config"] == original["configs"][mode]
                and old_status["experiment_id"] == EXPERIMENT_ID, "Old status/config mismatch")
        config = dict(original["configs"][mode], n_workers=12, run_name=run_name(case_id))
        check_config(case_id, config, original["configs"][mode])
        require(config["runs_dir"] == baseline["configs"]["soft"]["runs_dir"],
                "Runs must share the original viewer directory")
        descriptor = deepcopy(initial) if family == "1d" else dict(kind="homogeneous", velocity_m_s=5000.0)
        if family == "1d":
            descriptor["previous_snapshot_file"] = descriptor["snapshot_file"]
            descriptor["snapshot_file"] = str(STATE / "initial_model.npz")
        manifest["cases"][case_id] = dict(
            family=family, mode=mode, config=config, source_config=deepcopy(original["configs"][mode]),
            source_manifest=manifest["source_manifests"][family], initial_model_source=descriptor,
            previous_run=dict(run_id=old_status["run_id"], run_dir=old_status["run_dir"],
                              status="stopped", status_file=str(old_status_path),
                              status_sha256=sha(old_status_path)),
            result_file=str(STATE / "cases" / case_id / "result.json"))
    return manifest


def prepare(manifest):
    # Exclusive reservations intentionally remain after failure: never overwrite a partial freeze.
    reserved = ("manifest.json", "snapshot", "runner", "cases", "initial_model.npz", ".prepare.lock")
    for name in reserved:
        if (STATE / name).exists():
            raise FileExistsError(f"Refusing duplicate/partial preparation: {STATE / name}")
    with (STATE / ".prepare.lock").open("x") as stream:
        stream.write(manifest["created_at"] + "\n")
    snapshot = Path(manifest["snapshot"])
    snapshot.mkdir()
    baseline = read(SOURCES["homogeneous"])
    for name, checksum in manifest["runtime_sha256"].items():
        source = (Path(manifest["validator_source"]["path"]) if name == VALIDATOR
                  else Path(baseline["snapshot"]) / name)
        verify(source, checksum)
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as incoming, target.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
        verify(target, checksum)
    initial = manifest["cases"]["1d-soft"]["initial_model_source"]
    verify(initial["previous_snapshot_file"], INITIAL_SHA256)
    with Path(initial["previous_snapshot_file"]).open("rb") as incoming, Path(initial["snapshot_file"]).open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing)
    verify(initial["snapshot_file"], INITIAL_SHA256)
    runner = Path(manifest["runner"])
    runner.parent.mkdir()
    verify(STATE / "run.py", manifest["runner_sha256"])
    with (STATE / "run.py").open("rb") as incoming, runner.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing)
    verify(runner, manifest["runner_sha256"])
    write(Path(manifest["runner_manifest_pointer"]), dict(manifest=str(STATE / "manifest.json")), exclusive=True)
    for case_id in CASE_IDS:
        (STATE / "cases" / case_id).mkdir(parents=True)
    write(STATE / "manifest.json", manifest, exclusive=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Read-only source/hash/config checks")
    args = parser.parse_args()
    manifest = plan()
    if not args.check_only:
        prepare(manifest)
    print("Verified source plan (no writes)" if args.check_only else f"Prepared {STATE / 'manifest.json'}")
    for case_id, case in manifest["cases"].items():
        print(f"{case_id}: {case['config']['run_name']}; 12 workers")
    print(f"4 cases; 48 CPUs total; {len(manifest['runtime_sha256'])} source files; "
          "9009 control / 81081 training picks; 3244 scored events; BLAS/OMP/Numba=1")
    print(f"Frozen worker: {manifest['runner']} --case-id ID --manifest {STATE / 'manifest.json'} [--validate-only]")


if __name__ == "__main__":
    main()
