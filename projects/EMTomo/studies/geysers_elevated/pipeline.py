"""Detached elevated-station series. Reuse the tested finite queue, not its dataset."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
DEFAULT_STATE = REPO / "projects/EMTomo/runs/_geysers_elevated_queue"
try:
    import queue_core as core
except ModuleNotFoundError:
    sys.path.insert(0, str(REPO))
    from projects.EMTomo.studies.geysers2011 import pipeline as core


def cases():
    sequence = [("hard", 5), ("soft", 5), ("hard", 10), ("soft", 10),
                ("hard", 15), ("soft", 15), ("hard", 1)]
    return [{"id": f"{mode}-k{k}", "run_name": f"gyelev-{mode}-c1000-k{k}",
             "candidate_mode": mode, "n_candidates": k, "weights_top_n": k if mode == "soft" else 1,
             "temperature": None if mode == "soft" else 1., "lambda_reg": 1.,
             "coverage_damping_power": 2., "is_reference": (mode, k) == ("hard", 5)} for mode, k in sequence]


def latest_reference(runs):
    eligible = []
    for run in runs.glob("run_gy11-hard-c1000-*"):
        if not (run / "meta.json").is_file() or not (run / "final_model.npy").is_file():
            continue
        meta = core.read(run / "meta.json")
        n = meta["run_params"]["n_cycles"]
        if all((run / f"iter_{i}/complete.json").is_file() for i in range(n)):
            eligible.append((meta["started_at"], run))
    if not eligible:
        raise ValueError("No completed prior Geysers reference")
    return max(eligible)[1]


def initialize(state):
    from projects.EMTomo.studies.geysers_elevated.prepare import prepare_experiment
    from projects.EMTomo.studies.geysers_elevated.run_case import verify_hashes, shared
    manifest_path = state / "manifest.json"
    if manifest_path.exists():
        manifest = core.read(manifest_path)
        verify_hashes(manifest)
        return manifest
    reference = latest_reference(REPO / "projects/EMTomo/runs")
    reference_meta = core.read(reference / "meta.json")
    allowed_runtime = Path(reference_meta["run_params"]["provenance"]["runtime_root"])
    reference_code = reference_meta["run_params"]["provenance"]["runtime_hashes"]
    dataset = prepare_experiment()
    runtime = state / "runtime"
    if runtime.exists():
        raise ValueError("Unpublished runtime exists; inspect instead of overwriting")
    runtime.mkdir()
    code_hashes = {}
    for name, digest in reference_code.items():
        if not name.startswith("projects/") or not name.endswith(".py"):
            continue
        source = allowed_runtime / name
        if core.sha(source) != digest:
            raise ValueError(f"Historical source changed: {name}")
        target = runtime / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        code_hashes[name] = core.sha(target)
    runner = runtime / "runner"
    runner.mkdir()
    helpers = {"queue_core.py": HERE.parent / "geysers2011/pipeline.py",
               "_shared_case.py": HERE.parent / "geysers2011/run_case.py",
               "_shared_report.py": HERE.parent / "geysers2011/report.py"}
    helpers.update({name: HERE / name for name in ("pipeline.py", "run_case.py", "prepare.py", "calibrate.py", "report.py")})
    for name, source in helpers.items():
        target = runner / name
        target.write_bytes(source.read_bytes())
        code_hashes[f"runner/{name}"] = core.sha(target)
    sys.path.insert(0, str(REPO / "projects/EMTomo"))
    from config import InversionConfig
    from experiment_runner import initial_velocity
    import numpy as np
    allowed = {field.name for field in fields(InversionConfig)}
    baseline = {key: value for key, value in reference_meta["run_params"].items() if key in allowed}
    initial = initial_velocity(InversionConfig(**baseline), (20, 7, 7))
    shared.verify_initial(initial, reference / "initial_model.npy", np)
    manifest = {
        "schema_version": 2, "created_utc": core.now(), "experiment_id": dataset["experiment_id"],
        "experiments_root": str(REPO / "projects/forward_modeling/experiments"),
        "runs_dir": str(REPO / "projects/EMTomo/runs"), "runtime_root": str(runtime),
        "reference_run": str(reference), "baseline_config": baseline,
        "reference_sha256": {name: core.sha(reference / name) for name in ("meta.json", "initial_model.npy")},
        "input_sha256": {name: core.sha(Path(dataset["input_dir"]) / name) for name in shared.INPUT_FILES},
        "code_sha256": code_hashes, "resources": dict(shared.RESOURCE_VALUES),
        "cases": cases(), "dataset": dataset, "forward_implementation": "saved_hard1000_legacy_sphere",
        "method_policy": "exact production source snapshot of latest completed prior Geysers run; no v2 change",
        "forward_spec": {"refinement": 2, "source_radius_cells": 2,
                         "noise": {"relative_sigma": .01, "absolute_sigma_s": .01, "seed": 42}},
        "temperature_policy": {"common_to_soft_cases": True, "bounds": [1., 16.],
                               "target_median_ess_k15": 3., "max_initial_model_observations": 128,
                               "oracle_truth_used": False, "hard_temperature": 1.},
    }
    verify_hashes(manifest)
    for case in manifest["cases"]:
        (state / "cases" / case["id"]).mkdir(parents=True)
    core.atomic(manifest_path, manifest)
    return manifest


def worker(state):
    from run_case import verify_hashes, build_config, shared, verify_observations
    from calibrate import calibrate
    manifest = core.read(state / "manifest.json")
    verify_hashes(manifest)
    runtime = shared.load_runtime(manifest)
    import numpy as np
    from projects.forward_modeling.experiments import load_inputs, run_experiment
    from projects.forward_modeling.model import ForwardConfig, PointSet
    from projects.forward_modeling.noise import NoiseConfig
    from projects.forward_modeling.solver import check_convergence
    root, experiment = Path(manifest["experiments_root"]), manifest["experiment_id"]
    output = root / "output" / experiment
    ledger_path = state / "forward_complete.json"
    config = build_config(manifest, manifest["cases"][0], runtime)
    if not output.exists():
        model, stations, events = load_inputs(root, experiment)
        indices = sorted(set(np.linspace(0, len(events.ids)-1, 16, dtype=int).tolist()
                             + [int(np.argmin(events.coordinates_m[:, 2])), int(np.argmax(events.coordinates_m[:, 2]))]))
        selection = PointSet(tuple(events.ids[i] for i in indices), events.coordinates_m[indices])
        print(f"Convergence diagnostic r2/r4: {len(indices)} events, workers16", flush=True)
        control = check_convergence(model, stations, selection, config=ForwardConfig(refinement=2), workers=16)
        for key in ("max_abs_difference_s", "max_absolute_time_difference_s"):
            if not np.isfinite(control[key]) or control[key] < 0:
                raise ValueError("Invalid convergence result")
        core.atomic(state / "forward_convergence.json", {"event_ids": list(selection.ids), "result": control,
                    "policy": "subset diagnostic, no guaranteed error bound or new stopping threshold"})
        print(json.dumps(control), flush=True)
        print("Full synthetic observations: 3776 events x26 stations, workers32", flush=True)
        run_experiment(root, experiment, config=ForwardConfig(refinement=2), workers=32,
                       noise=NoiseConfig(relative_sigma=.01, absolute_sigma_s=.01, seed=42))
    metadata = core.read(output / "metadata.json")
    if metadata["config"] != {"refinement": 2, "source_radius_cells": 2} or metadata["noise"]["config"] != manifest["forward_spec"]["noise"]:
        raise ValueError("Existing forward configuration/noise mismatch")
    prepared = runtime.prepare_inversion(experiment, config, experiments_root=root)
    shared.verify_initial(prepared.initial_model.velocity, Path(manifest["reference_run"]) / "initial_model.npy", np)
    if len(prepared.event_ids) != 3776 or len(prepared.station_ids) != 26 or tuple(prepared.noise_sigmas) != (.01, .01):
        raise ValueError("Unexpected geometry/counts/noise")
    if ledger_path.exists():
        verify_observations(state, manifest)
        print("Validated existing observations and temperature calibration", flush=True)
        return
    print("Calibrating common soft T on initial model, <=128 observation rows, no truth", flush=True)
    with tempfile.TemporaryDirectory(prefix=".calibration-", dir=state) as staging:
        calibration = calibrate(prepared, config, runtime, staging)
    temperature = calibration.get("temperature")
    if type(temperature) not in (int, float) or not np.isfinite(temperature) or not 1. <= temperature <= 16.:
        raise ValueError("Calibrated temperature must be finite in [1,16]")
    core.atomic(state / "calibration.json", calibration)
    output_hashes = {name: core.sha(output / name) for name in ("arrivals.csv", "metadata.json")}
    verify_hashes(manifest)
    core.atomic(ledger_path, {"schema_version": 2, "experiment_id": experiment,
        "status": "complete", "validated_utc": core.now(),
        "output_sha256": output_hashes, "calibration_sha256": core.sha(state / "calibration.json")})
    print(f"Ready: soft T={calibration['temperature']:g}; initial model bitwise identical to reference", flush=True)
    print(json.dumps(calibration["selection"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "status", "stop", "supervise", "forward"))
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--lock-fd", type=int)
    args = parser.parse_args()
    state = args.state.resolve()
    if args.action == "launch":
        core.initialize = initialize
        core.launch(state)
    elif args.action == "status":
        status = core.read(state / "status.json")
        status["scheduler_alive"] = core.alive(status)
        print(json.dumps(status, indent=2))
    elif args.action == "stop":
        status = core.read(state / "status.json")
        if core.alive(status):
            os.kill(status["pid"], core.signal.SIGTERM)
            print("Stop requested; child process groups will be terminated")
        else:
            print(f"No live scheduler; status={status['status']}, active={status.get('active', [])}")
    elif args.action == "forward":
        worker(state)
    else:
        if args.lock_fd is None:
            parser.error("Use launch, not direct supervise")
        return core.supervise(state, args.lock_fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
