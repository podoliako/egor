"""Reproducible, no-output EM-step benchmark; no saved runs are created or modified.

Example: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
python benchmark_performance.py --repeats 3 --workers 1
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from statistics import median
from time import perf_counter

import numpy as np

from config import InversionConfig
from eikonal import station_travel_time_fields
from instruments.instruments_coords import metric_to_cell_coord, sample_cell_centered_trilinear_batch
from instruments.likelihood import PickNoise
from tomography.tomography_em import make_tomography_step, update_velocity, warm_up_jit
from velocity_model import VelocityModel


def benchmark(*, shape=(8, 6, 6), subdivision=3, n_events=32, n_stations=12,
              n_candidates=5, workers=1, repeats=3, interpolation="nearest"):
    if (len(shape) != 3 or min(shape) < 2 or subdivision < 1 or n_events < 1
            or n_stations < 2 or n_candidates < 1 or workers < 1 or repeats < 1):
        raise ValueError("Positive sizes/workers/repeats, at least two stations and two cells per axis required")
    rng = np.random.default_rng(42)
    initial = VelocityModel(np.full(shape, 5000.), 1000.)
    truth = VelocityModel(rng.uniform(4600., 5400., shape), 1000.)
    extent = np.asarray(shape) * initial.cell_size
    stations = rng.uniform(.1 * extent, .9 * extent, size=(n_stations, 3))
    stations[:, 2] = 0.0
    positions = rng.uniform(.15 * extent, .85 * extent, size=(n_events, 3))
    fine = truth.refined(subdivision, interpolation)
    times = station_travel_time_fields(fine, stations, workers)
    arrivals = np.stack([sample_cell_centered_trilinear_batch(
        times, metric_to_cell_coord(position, fine.cell_size)) for position in positions])
    arrivals -= arrivals.min(axis=1, keepdims=True)
    del fine, times, truth
    noise = PickNoise(.01, .01, .1)
    base = InversionConfig(n_cycles=1, cell_size=1000., subdivision=subdivision,
                            slowness_interpolation=interpolation, n_candidates=n_candidates,
                            weights_top_n=min(3, n_candidates), n_workers=workers,
                            lambda_reg=.1, save_runs=False)
    configs = {
        "reference": replace(base, likelihood_batch_size=1, compact_rays=False,
                             fast_refinement=False, linear_solver="legacy"),
        "optimized_dense": replace(base, linear_solver="dense"),
        "matrix_free": replace(base, linear_solver="matrix_free"),
    }
    warm_up_jit()
    measurements = {name: [] for name in configs}
    expected = None
    errors = {}
    # One untimed complete step per path also warms layout-dependent JIT specializations.
    for repeat in range(repeats + 1):
        names = list(configs)
        if repeat > 0:
            shift = (repeat - 1) % len(names)
            names = names[shift:] + names[:shift]
        for name in names:
            diagnostics = {}
            started = perf_counter()
            with redirect_stdout(io.StringIO()):
                result = make_tomography_step(initial, arrivals, stations, noise, configs[name],
                                               diagnostics=diagnostics)
            elapsed = perf_counter() - started
            if expected is None:
                expected = result
            for actual, reference in zip(result, expected):
                np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-10)
            velocity = update_velocity(initial.velocity, result[0], base.max_velocity_step_fraction)
            reference_velocity = update_velocity(initial.velocity, expected[0], base.max_velocity_step_fraction)
            errors[name] = {
                "max_abs_delta_s": float(np.max(np.abs(result[0] - expected[0]))),
                "max_abs_velocity_m_s": float(np.max(np.abs(velocity - reference_velocity))),
            }
            if repeat:
                measurements[name].append((elapsed, diagnostics))
    medians = {name: median(row[0] for row in rows) for name, rows in measurements.items()}
    modes = {}
    for name, rows in measurements.items():
        modes[name] = {
            "median_step_s": medians[name], "speedup_vs_reference": medians["reference"] / medians[name],
            "median_stage_wall_s": {key: median(row[1]["wall_s"][key] for row in rows)
                                    for key in rows[0][1]["wall_s"]},
            "median_worker_time_sum_s": {key: median(row[1]["events"]["worker_time_s"][key] for row in rows)
                                         for key in rows[0][1]["events"]["worker_time_s"]},
            "solver": rows[-1][1]["solver"], **errors[name],
        }
    return {"shape": shape, "subdivision": subdivision, "events": n_events, "stations": n_stations,
            "n_candidates": n_candidates, "workers": workers, "repeats": repeats,
            "output_enabled": False, "modes": modes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", nargs=3, type=int, default=(8, 6, 6))
    parser.add_argument("--subdivision", type=int, default=3)
    parser.add_argument("--events", type=int, default=32)
    parser.add_argument("--stations", type=int, default=12)
    parser.add_argument("--n-candidates", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--interpolation", choices=("nearest", "trilinear"), default="nearest")
    args = vars(parser.parse_args())
    args["shape"] = tuple(args["shape"])
    args["n_events"] = args.pop("events")
    args["n_stations"] = args.pop("stations")
    print(json.dumps(benchmark(**args), indent=2))


if __name__ == "__main__":
    main()
