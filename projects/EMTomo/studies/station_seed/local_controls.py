"""Local source-sphere and homogeneous negative controls, without model priors.

From the repository root:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 venv/bin/python \
        projects/EMTomo/studies/station_seed/local_controls.py

Both forward formulas are frozen in flat4_controls.py. The homogeneous control
uses exact Euclidean arrivals, true fixed event positions, no pick noise, and
12 cycles with the same increment damping as flat4. It is deliberately not a
reconstruction of a heterogeneous target or a tuning of stopping criteria.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from flat4_controls import (  # noqa: E402
    baseline_field, homogenized_field, local_station_fields, sample_events as sample_field,
)
from config import InversionConfig  # noqa: E402
from experiment_data import load_tomography_experiment  # noqa: E402
from instruments.instruments_coords import metric_to_cell_coord, sample_cell_centered_trilinear_batch  # noqa: E402
from instruments.instruments_ops import restriction_tables  # noqa: E402
from instruments.likelihood import PickNoise  # noqa: E402
from raytracing import compute_G_all_stations_serial  # noqa: E402
import tomography.tomography_em as em  # noqa: E402
import tomography.tomography_events as events  # noqa: E402
from velocity_model import VelocityModel  # noqa: E402


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def centered(a):
    return a - a.mean(axis=1, keepdims=True)


def summary(a):
    return {"rms": rms(a), "mean": float(np.mean(a)), "min": float(np.min(a)), "max": float(np.max(a))}


def sample_events(fields, xyz, h):
    coords = metric_to_cell_coord(xyz, h)
    return np.stack([sample_field(field, coords) for field in fields], axis=1)


def solver_fields(model, stations, solver, n_workers):
    mode = "baseline" if solver is baseline_field else "homogenized"
    return local_station_fields(model, stations, n_workers, mode=mode)


def sphere_invariance():
    shape, h, source = (32, 24, 24), 125.0, (1900.0, 1450.0, 0.0)
    axes = [(np.arange(n) + 0.5) * h - s for n, s in zip(shape, source)]
    distance = np.sqrt(axes[0][:, None, None] ** 2 + axes[1][None, :, None] ** 2 + axes[2][None, None, :] ** 2)
    base = np.full(shape, 4000.0)
    changed = base.copy()
    changed[(distance > 250.0) & (distance <= 375.0)] = 2000.0
    result = {}
    for name, solver in (("baseline", baseline_field), ("homogenized", homogenized_field)):
        difference = solver(changed, h, source) - solver(base, h, source)
        result[name] = {"outside_difference_ms": summary(1000 * difference[distance > 750]),
                        "inside_max_difference_ms": float(np.max(np.abs(difference[distance <= 375]))) * 1000}
    return result


def finite_differences(data, run):
    """Compare ray G with central differences at all events for three stations.

    These are unprofiled travel-time derivatives, a necessary local check, not
    a derivative of the complete weighted likelihood or a resolution estimate.
    """
    meta = json.loads((run / "meta.json").read_text())
    velocity = np.load(run / "initial_model.npy")
    cell = meta["grid_info"]["coarse_cell_size"]
    sub = meta["run_params"]["subdivision"]
    fine = VelocityModel(velocity, cell).refined(sub)
    xyz = data.reference_event_coordinates_m
    results = []
    for si in (0, 24, 51):
        station = data.station_coordinates_m[si]
        fields = baseline_field(fine.velocity, fine.cell_size, station)[None]
        gradients = [np.ascontiguousarray(g[None], dtype=np.float32) for g in np.gradient(fields[0])]
        source_cell = np.minimum((station // cell).astype(int), np.asarray(velocity.shape) - 1)
        cells = [tuple(source_cell)]
        for axis in (0, 1):
            for direction in (-1, 1):
                index = source_cell.copy()
                index[axis] += direction
                if np.all(index >= 0) and np.all(index < velocity.shape):
                    cells.append(tuple(index))
        cells.append((int(source_cell[0]), int(source_cell[1]), 1))
        g = np.empty((len(xyz), len(cells)))
        reached_all = True
        tables = restriction_tables(velocity.shape, sub)
        station_coord = metric_to_cell_coord(station[None], fine.cell_size)
        for ei, point in enumerate(xyz):
            sensitivity, reached = compute_G_all_stations_serial(
                *gradients, station_coord, metric_to_cell_coord(point, fine.cell_size),
                fine.cell_size, 0.1, 0.1, 50000, np.zeros(3), np.asarray(fine.shape) - 1.0, 3.0,
                *tables, velocity.shape,
            )
            reached_all &= bool(reached[0])
            g[ei] = [sensitivity[0][index] for index in cells]
        assert reached_all
        print(f"FD station {si}: {len(cells)} cells, {len(xyz)} events", flush=True)
        for ci, index in enumerate(cells):
            for fraction in (0.001, 0.003):
                ds = fraction / velocity[index]
                for name, solver in (("baseline", baseline_field), ("homogenized", homogenized_field)):
                    predicted = []
                    for sign in (-1, 1):
                        perturbed = velocity.copy()
                        perturbed[index] = 1 / (1 / velocity[index] + sign * ds)
                        medium = VelocityModel(perturbed, cell).refined(sub)
                        field = solver(medium.velocity, fine.cell_size, station)[None]
                        predicted.append(sample_events(field, xyz, fine.cell_size)[:, 0])
                    fd = (predicted[1].astype(float) - predicted[0].astype(float)) / (2 * ds)
                    difference = g[:, ci] - fd
                    results.append({"station_index": si, "station_m": station.tolist(), "cell": [int(i) for i in index],
                                    "slowness_fraction": fraction, "forward": name, "g_m": summary(g[:, ci]),
                                    "fd_m": summary(fd), "difference_m": summary(difference),
                                    "relative_l2_error": rms(difference) / max(rms(fd), 1e-12)})
    return results


@contextmanager
def fixed_positions(arrivals, xyz, solver):
    original = (events.negative_log_likelihood, events.select_candidate_cells,
                events.refine_hypocentre_in_cell, em.station_travel_time_fields)
    lookup = {row.tobytes(): point for row, point in zip(arrivals, xyz)}
    current = {}

    def locate(times, observed, noise, normalization=None):
        current["point"] = metric_to_cell_coord(lookup[observed.tobytes()], 125.0)
        return np.zeros(times.shape[1:])

    def select(cost, n, min_distance=1):
        return np.rint(current["point"]).astype(int)[None]

    def refine(times, observed, cell, noise):
        return current["point"].copy(), 0.0

    def fields(model, stations, n_workers=1):
        return solver_fields(model, stations, solver, n_workers)

    events.negative_log_likelihood, events.select_candidate_cells = locate, select
    events.refine_hypocentre_in_cell, em.station_travel_time_fields = refine, fields
    try:
        yield
    finally:
        (events.negative_log_likelihood, events.select_candidate_cells,
         events.refine_hypocentre_in_cell, em.station_travel_time_fields) = original


def homogeneous_controls():
    shape, cell, sub, velocity = (12, 8, 8), 500.0, 4, 4000.0
    stations = np.array([(x, y, 0.0) for x in (750.0, 2250.0, 3750.0, 5250.0)
                         for y in (2000.0 / 3, 2000.0, 10000.0 / 3)])
    xyz = np.array([(x, y, z) for x in np.linspace(500, 5500, 8)
                    for y in np.linspace(500, 3500, 4) for z in np.linspace(750, 3250, 5)])
    initial = VelocityModel(np.full(shape, velocity), cell)
    exact = np.linalg.norm(xyz[:, None, :] - stations[None, :, :], axis=2) / velocity
    fields = solver_fields(initial.refined(sub), stations, baseline_field, 4)
    numerical = sample_events(fields, xyz, cell / sub)
    config = InversionConfig(cell_size=cell, subdivision=sub, n_cycles=12, n_candidates=1,
                             weights_top_n=1, candidate_mode="hard", lambda_reg=0.5,
                             coverage_damping_power=1.0, max_velocity_step_fraction=0.02,
                             n_workers=4, save_runs=False, weight_noise_relative_sigma=0.0,
                             weight_noise_absolute_sigma_s=0.01, weight_model_sigma_s=0.02)
    result = {"geometry": {"shape": shape, "stations_m": stations.tolist(), "events_m": xyz.tolist()},
              "config": asdict(config), "pick_noise_added": False, "initial_model_is_truth": True,
              "initial_centered_numerical_error_ms": summary(centered(numerical - exact) * 1000), "runs": {}}
    em.warm_up_jit()
    for observation_name, observations, cycles in (("self_consistent", numerical, 12), ("analytic", exact, 12)):
        arrivals = observations - observations.min(axis=1, keepdims=True)
        for solver_name, solver in (("baseline", baseline_field), ("homogenized", homogenized_field)):
            model = initial
            curve = []
            with fixed_positions(arrivals, xyz, solver):
                for cycle in range(cycles):
                    ds, _, _ = em.make_tomography_step(model, arrivals, stations, PickNoise(0, 0.01, 0.02), config)
                    model = VelocityModel(em.update_velocity(model.velocity, ds, 0.02), cell)
                    difference = model.velocity - velocity
                    curve.append({"cycle": cycle + 1, "rms_m_s": rms(difference),
                                  "top_rms_m_s": rms(difference[:, :, 0]),
                                  "top_min_m_s": float(difference[:, :, 0].min()),
                                  "top_max_m_s": float(difference[:, :, 0].max())})
                    print(observation_name, solver_name, curve[-1], flush=True)
            final_fields = solver_fields(model.refined(sub), stations, solver, 4)
            final_times = sample_events(final_fields, xyz, cell / sub)
            result["runs"][f"{observation_name}_{solver_name}"] = {
                "cycles": curve, "final_centered_residual_ms": summary(centered(final_times - observations) * 1000)}
            if observation_name == "self_consistent":
                result["runs"][f"{observation_name}_{solver_name}"]["stable_below_1e_7_m_s"] = curve[-1]["rms_m_s"] < 1e-7
    return result


def main():
    runs_root = HERE.parents[1] / "runs"
    run = next(runs_root.glob("run_fl-fixed-c500-*20261013_200119"))
    meta = json.loads((run / "meta.json").read_text())
    data = load_tomography_experiment(meta["source_experiment"]["id"])
    result = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "flat4_run": run.name, "derivative_units": "metres (dT/dslowness)",
              "sphere_invariance": sphere_invariance(),
              "finite_differences": finite_differences(data, run)}
    output = HERE / "local_results.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    result["homogeneous"] = homogeneous_controls()
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Results: {output}", flush=True)


if __name__ == "__main__":
    main()
