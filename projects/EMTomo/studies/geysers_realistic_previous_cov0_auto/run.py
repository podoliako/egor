"""Matched realistic-model soft/hard runs using the current held-out stopping rule.

The manifest freezes the updated runtime and a common split; max cycles is only
an upper bound. Never import the preceding pre-early-stopping runtime.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("soft", "hard"), required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parent / "manifest.json")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    manifest = read(args.manifest)
    state = Path(manifest["state_dir"])
    root = Path(manifest["experiments_root"])
    for name, checksum in manifest["runtime_sha256"].items():
        if sha(Path(manifest["snapshot"]) / name) != checksum:
            raise ValueError(f"Frozen updated runtime changed: {name}")
    if sha(manifest["runner"]) != manifest["runner_sha256"]:
        raise ValueError("Frozen runner changed")
    for category in ("input", "output"):
        for name, checksum in manifest[f"{category}_sha256"].items():
            if sha(root / category / manifest["experiment_id"] / name) != checksum:
                raise ValueError(f"Experiment changed: {category}/{name}")
    sys.path[:0] = [str(Path(manifest["snapshot"]) / "projects/EMTomo"), manifest["snapshot"]]
    import numpy as np
    import config as config_module
    import experiment_runner
    from tomography.tomography_em import run_em, warm_up_jit
    from tomography.tomography_logging import TomographyLogger
    from tomography.tomography_validation import split_control_arrivals

    for module in (config_module, experiment_runner):
        if not Path(module.__file__).resolve().is_relative_to(Path(manifest["snapshot"])):
            raise ValueError("Non-frozen numerical module imported")
    config = config_module.InversionConfig(**manifest["configs"][args.mode])
    prepared = experiment_runner.prepare_inversion(manifest["experiment_id"], config, root)
    noise = experiment_runner.resolve_pick_noise(config, prepared.noise_sigmas)
    training, mask = split_control_arrivals(
        prepared.arrivals_table, fraction=config.control_fraction, seed=config.control_seed,
        min_training_picks=config.control_min_training_picks)
    if (prepared.arrivals_table.shape != (3465, 26) or prepared.initial_model.shape != (25, 10, 10)
            or asdict(noise) != manifest["assumed_noise"]
            or hashlib.sha256(mask.tobytes()).hexdigest() != manifest["control_split"]["mask_sha256"]
            or int(mask.sum()) != manifest["control_split"]["n_control"]):
        raise ValueError("Unexpected data, noise, grid or control split")
    if not np.all(np.isnan(training[mask])):
        raise ValueError("Control picks leaked into training")
    if args.validate_only:
        print(f"Validated {args.mode}: shared {int(mask.sum())} control picks, "
              f"auto stop patience={config.early_stopping_patience}, "
              f"min improvement={config.early_stopping_min_relative_improvement}, "
              f"max updates={config.n_cycles}", flush=True)
        return
    path = state / f"{args.mode}_status.json"
    if path.exists():
        raise FileExistsError("Run already attempted; refusing duplicate launch")
    status = dict(state="running", pid=os.getpid(), mode=args.mode,
                  started_at=datetime.now(timezone.utc).isoformat(), config=asdict(config),
                  experiment_id=manifest["experiment_id"], manifest_sha256=sha(args.manifest))
    write(path, status)
    print(json.dumps(status, indent=2), flush=True)
    try:
        warm_up_jit()
        logger = TomographyLogger(base_dir=config.runs_dir, run_name=config.run_name,
                                  run_version=config.run_version,
                                  run_tags=dict(mode=args.mode, cov=0, ncand=50, topn=50, dmin=7,
                                                stop="control", cap=config.n_cycles))
        status.update(run_id=logger.run_id, run_dir=str(logger.run_dir))
        write(path, status)
        run_em(config, prepared.initial_model, prepared.arrivals_table, prepared.station_locs,
               noise, reference_model=prepared.reference_model, logger=logger,
               source_experiment={"id": manifest["experiment_id"],
                                  "model_sha256": manifest["input_sha256"]["model.npz"],
                                  "study_manifest": str(args.manifest)})
        stopping = read(logger.run_dir / "stopping.json")
        completed = len(list(logger.run_dir.glob("iter_*/complete.json")))
        final = np.load(logger.run_dir / "final_model.npy", allow_pickle=False)
        if (completed != stopping["completed_updates"] or not 3 <= completed <= config.n_cycles
                or final.shape != prepared.initial_model.shape or not np.isfinite(final).all()
                or np.any(final <= 0)):
            raise ValueError("Invalid completed run or stopping checkpoint")
        np.testing.assert_array_equal(np.load(logger.run_dir / "control_mask.npy"), mask)
        status.update(state="completed", stopping=stopping, completed_updates=completed,
                      final_model_sha256=sha(logger.run_dir / "final_model.npy"))
    except BaseException as error:
        status.update(state="failed", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
        raise
    finally:
        status["finished_at"] = datetime.now(timezone.utc).isoformat()
        write(path, status)


if __name__ == "__main__":
    main()
