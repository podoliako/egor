"""One frozen p12 case; --validate-only never warms JIT or calls a solver."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
import traceback

EXPERIMENT_ID = "geysers_realistic_previous_20x8x8_26st_3465ev_noise01_pick005_seed42"
VALIDATOR = "projects/EMTomo/tomography/tomography_validation.py"
MASK_SHA256 = "f5da0c11e6245130c14cae34432ba5c7be6651cdbee864f966f6592f7919b1bf"
INITIAL_SHA256 = "23e1119f2d8250d6a4c6cf10cbfb4eaa74abf3fbd167122f7ad0c97aaab08dce"
CASE_IDS = ("homogeneous-soft", "homogeneous-hard", "1d-soft", "1d-hard")
THREAD_ENV = dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
                           "NUMBA_NUM_THREADS"), "1")
NOISE = dict(relative_sigma=0.01, absolute_sigma_s=0.05, model_sigma_s=0.02)
CONTROL = dict(requested_fraction=0.1, seed=42, n_control=9009, n_training=81081,
               scored_events=3244, mask_sha256=MASK_SHA256)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(path, checksum):
    require(sha(path) == checksum, f"SHA256 mismatch: {path}")


def write(path, value, exclusive=False):
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with Path(path).open("x") as stream:
            stream.write(text)
    else:
        temporary = Path(path).with_suffix(f".json.{os.getpid()}.tmp")
        temporary.write_text(text)
        temporary.replace(path)


def run_name(case_id):
    family, mode = case_id.rsplit("-", 1)
    return f"geysers-realistic-prev-{'hom' if family == 'homogeneous' else '1d'}-cov0-auto-p12-{mode}"


def check_config(case_id, config, original):
    require(case_id in CASE_IDS, "Unknown case")
    expected = dict(original, n_workers=12, run_name=run_name(case_id))
    require(config == expected and original["n_workers"] == 4,
            "Only n_workers 4->12 and run_name may change")
    fixed = dict(n_workers=12, n_cycles=7, n_candidates=50, weights_top_n=50,
                 weights_min_distance=7, coverage_damping_power=0.0, lambda_reg=1.0,
                 max_velocity_step_fraction=0.02, cell_size=800.0, subdivision=8,
                 control_fraction=0.1, control_seed=42, control_min_training_picks=5,
                 early_stopping_patience=2, early_stopping_min_relative_improvement=0.01,
                 weight_noise_relative_sigma=0.01, weight_noise_absolute_sigma_s=0.05,
                 weight_model_sigma_s=0.02, candidate_mode=case_id.rsplit("-", 1)[1],
                 save_runs=True)
    require(all(config.get(key) == value for key, value in fixed.items()),
            "Scientific parameters or auto stopping changed")
    if case_id.startswith("1d-"):
        require(len(config["initial_layer_velocities_m_s"] or []) == 10
                and len(config["initial_layer_boundaries_km"] or []) == 9,
                "1D case requires original ten-layer initialization")
    else:
        require(config["initial_velocity_m_s"] == 5000.0
                and config["initial_gradient_m_s"] is None
                and config["initial_layer_boundaries_km"] is None
                and config["initial_layer_velocities_m_s"] is None,
                "Homogeneous initialization changed")


def validate(manifest, case_id):
    """Return prepared inputs and the mask, with no forward/scoring/JIT calls."""
    require(manifest["experiment_id"] == EXPERIMENT_ID, "Wrong experiment")
    require(manifest["assumed_noise"] == NOISE and manifest["control_split"] == CONTROL,
            "Noise or common split descriptor changed")
    require(set(manifest["cases"]) == set(CASE_IDS), "Expected exactly four cases")
    require(manifest["auto_stopping"] == dict(max_updates=7, min_updates=3, patience=2,
                min_relative_improvement=0.01, retain_stopping_update=True),
            "Auto stopping descriptor changed")
    policy = manifest["hardware_policy"]
    require(policy["n_workers_per_run"] == 12 and policy["concurrent_runs"] == 4
            and policy["total_cpus"] == 48 and policy["thread_environment"] == THREAD_ENV,
            "Hardware policy changed")
    snapshot = Path(manifest["snapshot"]).resolve()
    for name, checksum in manifest["runtime_sha256"].items():
        require(Path(name).suffix == ".py" and not Path(name).is_absolute()
                and ".." not in Path(name).parts, "Invalid runtime source path")
        verify(snapshot / name, checksum)
    require(manifest["runtime_sha256"][VALIDATOR] == manifest["validator_source"]["sha256"],
            "Validator overlay descriptor mismatch")
    original_runtime = manifest["original_runtime_sha256"]
    require(set(original_runtime) == set(manifest["runtime_sha256"]), "Runtime file set changed")
    require(all(checksum == original_runtime[name] for name, checksum in
                manifest["runtime_sha256"].items() if name != VALIDATOR),
            "Non-validator numerical source changed")
    verify(manifest["runner"], manifest["runner_sha256"])
    verify(__file__, manifest["runner_sha256"])
    root = Path(manifest["experiments_root"])
    for category in ("input", "output"):
        for name, checksum in manifest[f"{category}_sha256"].items():
            verify(root / category / EXPERIMENT_ID / name, checksum)
    for other_id, other in manifest["cases"].items():
        check_config(other_id, other["config"], other["source_config"])
    case = manifest["cases"][case_id]

    # Refuse contaminated interpreters, rather than silently reusing live modules.
    require(not any(name in sys.modules for name in
                    ("config", "experiment_runner", "tomography", "velocity_model", "projects")),
            "Worker needs a fresh interpreter before activating snapshot")
    os.environ.update(THREAD_ENV)
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(snapshot / "projects/EMTomo"), str(snapshot)]
    import numpy as np
    import config as config_module
    import experiment_runner
    from tomography.tomography_validation import split_control_arrivals

    config = config_module.InversionConfig(**case["config"])
    require(asdict(config) == case["config"], "Config normalization changed parameters")
    prepared = experiment_runner.prepare_inversion(EXPERIMENT_ID, config, root)
    noise = experiment_runner.resolve_pick_noise(config, prepared.noise_sigmas)
    training, mask = split_control_arrivals(
        prepared.arrivals_table, fraction=config.control_fraction, seed=config.control_seed,
        min_training_picks=config.control_min_training_picks)
    actual = dict(requested_fraction=config.control_fraction, seed=config.control_seed,
                  n_control=int(mask.sum()), n_training=int(np.isfinite(training).sum()),
                  scored_events=int(mask.any(axis=1).sum()),
                  mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())
    require(actual == manifest["control_split"], "Common control mask/counts changed")
    require(prepared.arrivals_table.shape == (3465, 26)
            and prepared.station_locs.shape == (26, 3)
            and prepared.initial_model.shape == (25, 10, 10)
            and prepared.reference_model.shape == (25, 10, 10)
            and len(prepared.event_ids) == 3465 and len(prepared.station_ids) == 26
            and asdict(noise) == NOISE, "Unexpected dimensions or assumed noise")
    require(np.isnan(training[mask]).all() and np.isfinite(prepared.arrivals_table[mask]).all(),
            "Held-out picks leaked into training")
    np.testing.assert_array_equal(training[~mask], prepared.arrivals_table[~mask])
    require(np.all(np.isfinite(training).sum(axis=1) >= config.control_min_training_picks),
            "Insufficient training picks")
    initial = case["initial_model_source"]
    if case_id.startswith("1d-"):
        require(initial["sha256"] == INITIAL_SHA256, "Wrong supplied 1D model")
        verify(initial["snapshot_file"], INITIAL_SHA256)
        with np.load(initial["snapshot_file"], allow_pickle=False) as saved:
            require(float(saved["cell_size_m"]) == config.cell_size
                    and np.array_equal(saved["origin_m"], [0, 0, 0]), "1D grid mismatch")
            np.testing.assert_array_equal(prepared.initial_model.velocity, saved["velocity"])
    else:
        require(initial == dict(kind="homogeneous", velocity_m_s=5000.0), "Wrong initial descriptor")
        require(np.all(prepared.initial_model.velocity == 5000.0), "Wrong homogeneous model")
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if filename and (name in ("config", "experiment_runner", "experiment_data", "velocity_model",
                                 "eikonal", "interpolation", "raytracing")
                         or name.split(".")[0] in ("tomography", "instruments", "projects")):
            require(Path(filename).resolve().is_relative_to(snapshot), f"Non-frozen import: {name}")
    return config, prepared, noise, mask, actual


def check_finished(logger, config, prepared, mask):
    import numpy as np
    directory = logger.run_dir
    stopping = read(directory / "stopping.json")
    completed = stopping["completed_updates"]
    require(type(completed) is int and 3 <= completed <= 7, "Invalid completed update count")
    require({p.parent.name for p in directory.glob("iter_*/complete.json")}
            == {f"iter_{i}" for i in range(completed)}, "Completion markers disagree with stopping")
    require({p.name for p in directory.glob("iter_*") if p.is_dir()}
            == {f"iter_{i}" for i in range(completed)}, "Unexpected iteration directories")
    previous = None
    insufficient = 0
    for i in range(completed):
        checkpoint = directory / f"iter_{i}"
        require(read(checkpoint / "complete.json")["iter"] == i, "Invalid complete marker")
        score = read(checkpoint / "control.json")
        rmse = score["rmse_s"]
        require(score["iter"] == i and score["n_control"] == CONTROL["n_control"]
                and np.isfinite(rmse) and rmse >= 0, "Invalid saved control score")
        if previous is not None:
            insufficient = 0 if rmse < previous * 0.99 else insufficient + 1
        stop = i + 1 >= 3 and insufficient >= config.early_stopping_patience
        require(score["insufficient_steps"] == insufficient and score["stop"] == stop
                and (not stop or i == completed - 1), "Saved auto stopping inconsistent")
        previous = rmse
    require(stopping["control_rmse_s"] == previous and stopping["insufficient_steps"] == insufficient
            and stopping["reason"] == ("control_rmse_plateau" if stop else "max_cycles")
            and (stop or completed == config.n_cycles), "Stopping summary inconsistent")
    final = np.load(directory / "final_model.npy", allow_pickle=False)
    require(final.shape == prepared.initial_model.shape and np.isfinite(final).all()
            and np.all(final > 0), "Invalid final model")
    np.testing.assert_array_equal(np.load(directory / "control_mask.npy", allow_pickle=False), mask)
    np.testing.assert_array_equal(np.load(directory / "initial_model.npy", allow_pickle=False),
                                  prepared.initial_model.velocity)
    split = read(directory / "control_split.json")
    require(all(split[key] == CONTROL[key] for key in
                ("seed", "requested_fraction", "n_control", "n_training")), "Saved split changed")
    meta = read(directory / "meta.json")
    require(meta["source_experiment"]["id"] == EXPERIMENT_ID and meta["event_locs"] == [],
            "Viewer source ID or event truth isolation changed")
    return stopping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", choices=CASE_IDS, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = read(manifest_path)
    path = Path(manifest["state_dir"]) / "cases" / args.case_id / "result.json"
    started = perf_counter()
    status = dict(status="running", state="running", pid=os.getpid(), case_id=args.case_id,
                  started_at=datetime.now(timezone.utc).isoformat(), manifest=str(manifest_path),
                  manifest_sha256=sha(manifest_path), experiment_id=manifest["experiment_id"],
                  config=manifest["cases"][args.case_id]["config"],
                  control_split=manifest["control_split"], hardware_policy=manifest["hardware_policy"],
                  initial_model_source=manifest["cases"][args.case_id]["initial_model_source"])
    if not args.validate_only:
        write(path, status, exclusive=True)
    try:
        config, prepared, noise, mask, actual = validate(manifest, args.case_id)
        if args.validate_only:
            print(json.dumps(dict(case_id=args.case_id, validated=True, n_workers=config.n_workers,
                                  control_split=actual, updates="3..7", solver_calls=0), indent=2), flush=True)
            return
        from tomography.tomography_em import run_em, warm_up_jit
        from tomography.tomography_logging import TomographyLogger

        class CaseLogger(TomographyLogger):
            def __init__(self):
                super().__init__(base_dir=config.runs_dir, run_name=config.run_name,
                                 run_version=config.run_version)
                status.update(run_id=self.run_id, run_dir=str(self.run_dir))
                write(path, status)

        warm_up_jit()
        logger = CaseLogger()
        run_em(config, prepared.initial_model, prepared.arrivals_table, prepared.station_locs,
               noise, reference_model=prepared.reference_model, logger=logger,
               source_experiment=dict(id=EXPERIMENT_ID,
                                      model_sha256=manifest["input_sha256"]["model.npz"],
                                      study_manifest=str(manifest_path), case_id=args.case_id,
                                      initial_model_source=status["initial_model_source"]))
        stopping = check_finished(logger, config, prepared, mask)
        status.update(status="completed", state="completed", stopping=stopping,
                      completed_updates=stopping["completed_updates"],
                      final_model_sha256=sha(logger.run_dir / "final_model.npy"))
    except BaseException as error:
        status.update(status="failed", state="failed", error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc())
        traceback.print_exc()
        raise
    finally:
        if not args.validate_only:
            status.update(finished_at=datetime.now(timezone.utc).isoformat(), elapsed_s=perf_counter() - started)
            write(path, status)


if __name__ == "__main__":
    main()
