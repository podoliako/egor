"""Hard-500 started from the model after eight updates of saved hard-1000.

From egor:
    OPENBLAS_NUM_THREADS=12 OMP_NUM_THREADS=12 venv/bin/python \
        projects/EMTomo/studies/flatlayers/run_multiscale.py --workers 12

Only the starting velocity changes. The saved baseline forward is replayed in
this process, so this is not also a test of the later homogeneous-sphere fix.
No production files or original runs are modified. Hypocentres are searched
normally; only velocities, not locations, transfer between grids.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from config import InversionConfig  # noqa: E402
import eikonal  # noqa: E402
from experiment_data import DEFAULT_EXPERIMENTS_ROOT, _sha256  # noqa: E402
from experiment_runner import prepare_inversion, resolve_pick_noise  # noqa: E402
from studies.station_seed.flat4_controls import baseline_field, model_metrics  # noqa: E402
import tomography.tomography_em as em  # noqa: E402
from tomography.tomography_logging import TomographyLogger  # noqa: E402
from velocity_model import VelocityModel  # noqa: E402
import json  # noqa: E402

RESULT = HERE / "multiscale_result.json"
SOURCE_PATTERN = "run_fl-hard-c1000-*20261013_200119"
BASELINE_PATTERN = "run_fl-hard-c500-*20261013_200119"
SAVED_FORWARD = "skfmm_order2_exact_station_homogeneous_sphere"


def unique_run(pattern):
    paths = list((ROOT / "runs").glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected one run matching {pattern}, got {len(paths)}")
    return paths[0]


def load_setup(workers):
    source, baseline = unique_run(SOURCE_PATTERN), unique_run(BASELINE_PATTERN)
    source_meta = json.loads((source / "meta.json").read_text())
    meta = json.loads((baseline / "meta.json").read_text())
    if source_meta["source_experiment"] != meta["source_experiment"]:
        raise ValueError("Source and baseline experiments differ")
    if any(m["run_params"]["travel_times"] != SAVED_FORWARD for m in (source_meta, meta)):
        raise ValueError("Saved runs do not use the expected replay forward")
    allowed = {field.name for field in fields(InversionConfig)}
    params = {key: value for key, value in meta["run_params"].items() if key in allowed}
    config = replace(InversionConfig(**params), n_workers=workers,
                     run_name="fl-hard-c500-from-c1000-c8", runs_dir=str(ROOT / "runs"),
                     initial_gradient_m_s=None, initial_layer_boundaries_km=None,
                     initial_layer_velocities_m_s=None)
    experiment_id = meta["source_experiment"]["id"]
    prepared = prepare_inversion(experiment_id, config)
    # iter_i/model is PRE-update: iter_8 is the state AFTER eight full updates.
    checkpoint = source / "iter_8/model.npy"
    if not (source / "iter_7/complete.json").exists():
        raise ValueError("The eighth parent update is not complete")
    coarse = VelocityModel(np.load(checkpoint), source_meta["grid_info"]["coarse_cell_size"])
    ratio = coarse.cell_size / config.cell_size
    if ratio != 2 or coarse.shape != (20, 7, 7) or config.subdivision != 4:
        raise ValueError("Expected the saved 1000 -> 500 m flat4 grid pair")
    initial = coarse.refined(2, "nearest")
    if initial.shape != prepared.initial_model.shape:
        raise ValueError("Transferred grid does not cover the experiment domain")
    source_sub = source_meta["run_params"]["subdivision"]
    fine_identical = np.array_equal(coarse.refined(source_sub).velocity,
                                    initial.refined(config.subdivision).velocity)
    if not fine_identical:
        raise ValueError("Transfer changed the fine-grid velocity field")
    origin = {"run_id": source.name, "completed_updates": 8,
              "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": _sha256(checkpoint),
              "source_cell_size_m": coarse.cell_size, "target_cell_size_m": initial.cell_size,
              "transfer": "nearest; split each parent into 2x2x2 identical cells",
              "fine_grid_velocity_bitwise_identical": fine_identical,
              "hypocentres_transferred": False}
    return config, experiment_id, prepared, initial, origin, baseline


class CheckpointLogger(TomographyLogger):
    def __init__(self, *args, origin, **kwargs):
        super().__init__(*args, **kwargs)
        self.origin = origin

    def save_meta(self, run_params, *args, **kwargs):
        params = {**run_params, "initialization": "saved_checkpoint_nearest",
                  "initial_model_source": self.origin}
        return super().save_meta(params, *args, **kwargs)


def summarize(run, baseline, data, origin):
    def rows(path):
        meta = json.loads((path / "meta.json").read_text())
        n = meta["run_params"]["n_cycles"]
        output = []
        for cycle in range(n + 1):
            model = np.load(path / ("final_model.npy" if cycle == n else f"iter_{cycle}/model.npy"))
            metrics = model_metrics(model, 500.0, data)
            output.append({"cycle": cycle, "model_metrics": metrics})
        for iteration, row in enumerate(output[:-1]):
            distances, residuals = [], []
            for event, xyz in enumerate(data.reference_event_coordinates_m):
                directory = path / f"iter_{iteration}/event_{event}"
                with np.load(directory / "weights.npz") as weights:
                    position = weights["positions"][int(np.argmax(weights["weight_values"]))]
                distances.append(float(np.linalg.norm((position + 0.5) * 125.0 - xyz)))
                # Each pairwise row mean equals the arithmetically centred residual.
                residuals.append(np.load(directory / "residuals.npy").mean(axis=1))
            row["hypocentre_error_m"] = {"median": float(np.median(distances)),
                                         "p90": float(np.percentile(distances, 90))}
            row["centered_residual_rms_s"] = float(np.mean(np.concatenate(residuals) ** 2) ** 0.5)
            row["rays"] = json.loads((path / f"iter_{iteration}/ray_status.json").read_text())
        return output

    result = {"run_dir": str(run.relative_to(ROOT)), "baseline_run_dir": str(baseline.relative_to(ROOT)),
              "initial_model_source": origin, "forward": SAVED_FORWARD,
              "residual_and_hypocentre_metrics_are": "PRE-update; final post-update arrivals are not logged",
              "warm_start": rows(run), "original_hard_c500": rows(baseline)}
    RESULT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Summary: {RESULT}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    config, experiment_id, prepared, initial, origin, baseline = load_setup(args.workers)
    print(json.dumps({"initial_model_source": origin, "shape": initial.shape,
                      "fine_cell_size_m": initial.cell_size / config.subdivision,
                      "cycles": config.n_cycles, "mode": config.candidate_mode,
                      "forward": SAVED_FORWARD}, indent=2), flush=True)
    if args.validate_only:
        return
    from experiment_data import load_tomography_experiment
    data = load_tomography_experiment(experiment_id)
    name = f"{config.run_name}_{experiment_id}"
    logger = CheckpointLogger(base_dir=config.runs_dir, run_name=name, run_version=config.run_version,
                              run_tags={"mode": "hard", "cell": 500, "sub": 4, "init": "c1000-c8"}, origin=origin)
    print(f"Run directory: {logger.run_dir}", flush=True)
    old_forward, old_description = eikonal.travel_time_field, em.METHOD_DESCRIPTION.copy()
    try:
        eikonal.travel_time_field = baseline_field
        em.METHOD_DESCRIPTION = {**old_description, "travel_times": SAVED_FORWARD}
        em.warm_up_jit()
        em.run_em(config, initial, prepared.arrivals_table, prepared.station_locs,
                  resolve_pick_noise(config, prepared.noise_sigmas), reference_model=prepared.reference_model,
                  source_experiment={"id": experiment_id, "model_sha256": _sha256(
                      DEFAULT_EXPERIMENTS_ROOT / "input" / experiment_id / "model.npz")},
                  run_name=name, logger=logger)
    finally:
        eikonal.travel_time_field, em.METHOD_DESCRIPTION = old_forward, old_description
    summarize(logger.run_dir, baseline, data, origin)


if __name__ == "__main__":
    main()
