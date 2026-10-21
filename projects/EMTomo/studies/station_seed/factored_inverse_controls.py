"""Compare corrected skfmm and experimental factored forward inside unchanged EMTomo.

No model prior, spatial penalty, ray-deposition change or inversion stopping
rule is added. Each successful run performs all 12 cycles. JSON checkpoints
are saved after each full update; an incomplete resource-limited run is not
reported as a complete comparison.

From egor:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 venv/bin/python \
        projects/EMTomo/studies/station_seed/factored_inverse_controls.py homogeneous
    OPENBLAS_NUM_THREADS=12 OMP_NUM_THREADS=12 venv/bin/python \
        projects/EMTomo/studies/station_seed/factored_inverse_controls.py flat4 --workers 12
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from factored_candidate import travel_time_field as factored_field  # noqa: E402
from flat4_controls import homogenized_field, model_metrics, residual_metrics, sample_events  # noqa: E402
from config import InversionConfig  # noqa: E402
from experiment_data import load_tomography_experiment  # noqa: E402
from instruments.instruments_coords import metric_to_cell_coord  # noqa: E402
from instruments.likelihood import PickNoise  # noqa: E402
import tomography.tomography_em as em  # noqa: E402
import tomography.tomography_events as events  # noqa: E402
from tomography.tomography_logging import summarize_ray_status  # noqa: E402
from velocity_model import VelocityModel  # noqa: E402

OUTPUT = HERE / "factored_inverse_results.json"
_STATE = {}


def rms(a):
    return float(np.mean(np.asarray(a) ** 2) ** 0.5)


def save(result):
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")


def _init_field_worker(velocity, h, mode):
    _STATE.update(velocity=velocity, h=h, mode=mode, limits=threadpool_limits(limits=1))


def _field_worker(station):
    solver = factored_field if _STATE["mode"] == "factored" else homogenized_field
    return solver(_STATE["velocity"], _STATE["h"], station)


def station_fields(model, stations, workers, mode):
    result = np.empty((len(stations),) + model.shape, dtype=np.float32)
    with mp.get_context("fork").Pool(min(workers, len(stations)), initializer=_init_field_worker,
                                   initargs=(model.velocity, model.cell_size, mode)) as pool:
        for index, field in enumerate(pool.imap(_field_worker, stations)):
            result[index] = field
    return result


def predictions(times, xyz, h):
    coords = metric_to_cell_coord(xyz, h)
    return np.stack([sample_events(field, coords) for field in times], axis=1)


@contextmanager
def fixed_forward(arrivals, xyz, stations, config, mode, current):
    previous = (events.negative_log_likelihood, events.select_candidate_cells,
                events.refine_hypocentre_in_cell, em.station_travel_time_fields, em.run_events)
    original_run_events = em.run_events
    lookup = {row.tobytes(): point for row, point in zip(arrivals, xyz)}
    if len(lookup) != len(arrivals):
        raise ValueError("Event arrival rows must be unique")
    location = {}
    h = config.cell_size / config.subdivision

    def locate(times, observed, noise, normalization=None):
        location["point"] = metric_to_cell_coord(lookup[observed.tobytes()], h)
        return np.zeros(times.shape[1:])

    def select(cost, n, min_distance=1):
        return np.rint(location["point"]).astype(int)[None]

    def refine(times, observed, cell, noise):
        return location["point"].copy(), 0.0

    def forward(model, sources, n_workers=1):
        tick = time.perf_counter()
        times = station_fields(model, sources, n_workers, mode)
        current["forward_seconds"] = time.perf_counter() - tick
        current["pre_update_residuals"] = residual_metrics(predictions(times, xyz, h), arrivals)
        return times

    def instrumented_events(*args, **kwargs):
        hessian, rhs, logs = original_run_events(*args, **kwargs)
        status = summarize_ray_status(logs, len(stations))
        if status is None or status["expected"] != len(stations) * len(arrivals):
            raise RuntimeError("Incomplete ray-status logging")
        current["rays"] = status
        return hessian, rhs, logs

    events.negative_log_likelihood, events.select_candidate_cells = locate, select
    events.refine_hypocentre_in_cell, em.station_travel_time_fields = refine, forward
    em.run_events = instrumented_events
    try:
        yield
    finally:
        (events.negative_log_likelihood, events.select_candidate_cells,
         events.refine_hypocentre_in_cell, em.station_travel_time_fields, em.run_events) = previous


def run(initial, arrivals, xyz, stations, config, noise, mode, evaluate, section, result):
    model, current = initial, {}
    section.update(status="running", config=asdict(config), forward=mode, cycles=[])
    started = time.perf_counter()
    with fixed_forward(arrivals, xyz, stations, config, mode, current):
        for cycle in range(config.n_cycles):
            current.clear()
            tick = time.perf_counter()
            delta, sensitivity, confidence = em.make_tomography_step(model, arrivals, stations, noise, config)
            model = VelocityModel(em.update_velocity(model.velocity, delta, config.max_velocity_step_fraction), model.cell_size)
            row = {"cycle": cycle + 1, "wall_seconds": time.perf_counter() - tick,
                   "metrics": evaluate(model.velocity), **current}
            section["cycles"].append(row)
            section["last_model_m_s"] = model.velocity.tolist()
            section["wall_seconds"] = time.perf_counter() - started
            save(result)
            print(mode, row["cycle"], row["metrics"], row["rays"], flush=True)
    final_times = station_fields(model.refined(config.subdivision, config.slowness_interpolation), stations, config.n_workers, mode)
    section["final_residuals"] = residual_metrics(predictions(final_times, xyz, config.cell_size / config.subdivision), arrivals)
    section.update(status="complete", wall_seconds=time.perf_counter() - started)
    save(result)
    return model


def homogeneous(result, workers):
    shape, cell, velocity = (12, 8, 8), 500.0, 4000.0
    stations = np.array([(x, y, 0.0) for x in (750.0, 2250.0, 3750.0, 5250.0)
                         for y in (2000.0 / 3, 2000.0, 10000.0 / 3)])
    xyz = np.array([(x, y, z) for x in np.linspace(500, 5500, 8)
                    for y in np.linspace(500, 3500, 4) for z in np.linspace(750, 3250, 5)])
    initial = VelocityModel(np.full(shape, velocity), cell)
    exact = np.linalg.norm(xyz[:, None] - stations[None], axis=2) / velocity
    config = InversionConfig(cell_size=cell, subdivision=4, n_cycles=12, n_candidates=1, weights_top_n=1,
                             candidate_mode="hard", lambda_reg=0.5, coverage_damping_power=1,
                             max_velocity_step_fraction=0.02, n_workers=workers, save_runs=False)
    section = result.setdefault("homogeneous", {})
    section.update(initial_is_truth=True, noise_added=False, stations_m=stations.tolist(), events_m=xyz.tolist())

    def evaluate(v):
        delta = v - velocity
        return {"rms_error_m_s": rms(delta), "top_rms_error_m_s": rms(delta[:, :, 0]),
                "top_min_error_m_s": float(delta[:, :, 0].min()), "top_max_error_m_s": float(delta[:, :, 0].max())}

    for mode in ("skfmm", "factored"):
        times = station_fields(initial.refined(4), stations, workers, mode)
        own = predictions(times, xyz, 125.0)
        for observation_name, observed in (("own_null", own), ("analytic", exact)):
            arrivals = observed - observed.min(axis=1, keepdims=True)
            key = f"{mode}_{observation_name}"
            section[key] = {"initial_residuals": residual_metrics(own, observed)}
            run(initial, arrivals, xyz, stations, config, PickNoise(0, 0.01, 0.02), mode,
                evaluate, section[key], result)


def flat4(result, workers):
    saved_run = next((ROOT / "runs").glob("run_fl-fixed-c500-*20261013_200119"))
    meta = json.loads((saved_run / "meta.json").read_text())
    data = load_tomography_experiment(meta["source_experiment"]["id"])
    allowed = {f.name for f in fields(InversionConfig)}
    params = {k: v for k, v in meta["run_params"].items() if k in allowed}
    config = replace(InversionConfig(**params), n_workers=workers, save_runs=False)
    initial = VelocityModel(np.load(saved_run / "initial_model.npy"), config.cell_size)
    section = result["flat4_factored"] = {"saved_run": saved_run.name, "experiment": meta["source_experiment"],
                                            "initial_model_prior": False, "fixed_true_hypocentres": True}
    evaluate = lambda v: model_metrics(v, config.cell_size, data)
    model = run(initial, data.arrival_times_s, data.reference_event_coordinates_m, data.station_coordinates_m,
                config, PickNoise(0.01, 0.01, 0.02), "factored", evaluate, section, result)
    baseline = json.loads((HERE / "flat4_results.json").read_text())["patched_fixed_c500"]
    section["corrected_skfmm_reference"] = {"cycles": [{"cycle": row["cycle"], "metrics": row["model_metrics"]}
                                                        for row in baseline["cycles"]],
                                             "status": baseline["status"]}
    delta = model.velocity - np.asarray(baseline["models_m_s"][-1])
    section["difference_to_corrected_skfmm_m_s"] = {"rms": rms(delta), "max_abs": float(abs(delta).max())}
    save(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("homogeneous", "flat4"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    result = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}
    result["provenance"] = {"created_utc": datetime.now(timezone.utc).isoformat(),
                            "initial_model_prior": False, "production_solver_unchanged": True,
                            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            "candidate_sha256": hashlib.sha256((HERE / "factored_candidate.py").read_bytes()).hexdigest()}
    # Compile once in the parent before fork workers share the kernels.
    factored_field(np.full((6, 5, 4), 4000.0), 125.0, (250.0, 250.0, 0.0))
    em.warm_up_jit()
    if args.stage == "homogeneous":
        homogeneous(result, args.workers)
    else:
        flat4(result, args.workers)
    save(result)


if __name__ == "__main__":
    main()
