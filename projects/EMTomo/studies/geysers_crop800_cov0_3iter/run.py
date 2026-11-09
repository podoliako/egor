"""Three-step soft/hard coverage-damping ablation on unchanged standard-noise data.

Uses the preceding series' frozen numerical runtime. Only cycle count, run name
and coverage_damping_power change. Floor/percentile remain diagnostic settings:
with exponent zero neither can affect the solved velocity increment.
"""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback


BASE = Path(__file__).resolve().parents[2]
STATE = Path(__file__).resolve().parent
SOURCE = BASE / "studies/geysers_cb5x2x2_noisegrid_20261107/manifest.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("soft", "hard"), required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    manifest = read(SOURCE)
    case = next(c for c in manifest["cases"] if c["id"] == f"x1-{args.mode}")
    root = Path(manifest["experiments_root"])
    for relative, checksum in manifest["runtime_sha256"].items():
        if sha(Path(manifest["snapshot"]) / relative) != checksum:
            raise ValueError(f"Original frozen runtime changed: {relative}")
    for category, hashes in (("input", case["input_sha256"]), ("output", case["output_sha256"])):
        for name, checksum in hashes.items():
            if sha(root / category / case["experiment_id"] / name) != checksum:
                raise ValueError(f"Original experiment changed: {category}/{name}")
    sys.path[:0] = [str(Path(manifest["snapshot"]) / "projects/EMTomo"), manifest["snapshot"]]
    import numpy as np
    from config import InversionConfig
    from experiment_runner import prepare_inversion, resolve_pick_noise
    from tomography.tomography_em import run_em, warm_up_jit
    from tomography.tomography_logging import TomographyLogger
    from tomography.tomography_math import solve_slowness_update

    config = replace(InversionConfig(**case["config"]), n_cycles=3, coverage_damping_power=0.0,
                     run_name=f"geysers-crop800-cov0-{args.mode}-3iter")
    prepared = prepare_inversion(case["experiment_id"], config, root)
    noise = resolve_pick_noise(config, prepared.noise_sigmas)
    if asdict(noise) != manifest["assumed_noise"] or prepared.arrivals_table.shape != (3465, 26):
        raise ValueError("Unexpected standard-noise setup")
    if args.validate_only:
        hessian = np.diag([0.0, 1.0, 2.0, 4.0])
        rhs = np.array([0.0, 1.0, 1.0, 1.0])
        expected = rhs / (np.diag(hessian) + np.diag(hessian).mean())
        for floor in (0.001, 0.05, 1.0):
            for percentile in (0.0, 75.0, 100.0):
                delta, _, _ = solve_slowness_update(
                    hessian, rhs, (4, 1, 1), lambda_reg=1.0,
                    coverage_damping_power=0.0, coverage_floor=floor,
                    coverage_reference_percentile=percentile)
                np.testing.assert_allclose(delta.ravel(), expected, rtol=1e-12, atol=1e-12)
        print(f"Validated {args.mode}: standard noise, 3 cycles, coverage power 0; "
              "all nine floor/percentile combinations produce the identical uniform-damped update")
        return
    path = STATE / f"{args.mode}_status.json"
    if path.exists():
        raise FileExistsError("Run already attempted; inspect its status rather than restarting")
    status = dict(state="running", pid=os.getpid(), mode=args.mode,
                  started_at=datetime.now(timezone.utc).isoformat(),
                  reference_manifest=str(SOURCE), reference_manifest_sha256=sha(SOURCE),
                  reference_case=case["id"], experiment_id=case["experiment_id"],
                  coverage_floor_and_percentile="diagnostic only; inactive in damping with power zero",
                  config=asdict(config))
    write(path, status)
    print(json.dumps(status, indent=2), flush=True)
    try:
        warm_up_jit()
        logger = TomographyLogger(base_dir=config.runs_dir, run_name=config.run_name,
                                  run_version=config.run_version,
                                  run_tags=dict(mode=args.mode, ncand=50, topn=50, dmin=7, cov=0, cycles=3))
        status.update(run_id=logger.run_id, run_dir=str(logger.run_dir))
        write(path, status)
        run_em(config, prepared.initial_model, prepared.arrivals_table, prepared.station_locs,
               noise, reference_model=prepared.reference_model, logger=logger,
               source_experiment={"id": case["experiment_id"],
                                  "model_sha256": case["input_sha256"]["model.npz"],
                                  "coverage_ablation": {"reference_manifest": str(SOURCE),
                                                        "reference_case": case["id"], "power": 0.0}})
        final = np.load(logger.run_dir / "final_model.npy", allow_pickle=False)
        if (final.shape != prepared.initial_model.shape or not np.isfinite(final).all()
                or np.any(final <= 0)
                or len(list(logger.run_dir.glob("iter_*/complete.json"))) != 3):
            raise ValueError("Incomplete or invalid three-step result")
        status.update(state="completed", final_model_sha256=sha(logger.run_dir / "final_model.npy"))
    except BaseException as error:
        status.update(state="failed", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
        raise
    finally:
        status["finished_at"] = datetime.now(timezone.utc).isoformat()
        write(path, status)


if __name__ == "__main__":
    main()
