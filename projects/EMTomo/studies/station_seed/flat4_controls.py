"""Paired flat4 station-seed controls; writes only flat4_results.json beside this file.

Run from any directory (with the repository venv):
    PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=12 OMP_NUM_THREADS=12 \
        python flat4_controls.py fields --workers 12
    ... python flat4_controls.py reproduce --workers 12
    ... python flat4_controls.py invert --workers 12 --budget-seconds 900
    ... python flat4_controls.py summarize

The two forward formulas below are frozen locally, independent of production
changes to eikonal.py. Rays and updates use production code (station-cell-only
inside-sphere deposition, c381cfd). This is a fixed-TRUE-hypocentre oracle, not
an EM location run. There is no prior to the initial model and no stopping-rule
change: a successful patched run always performs exactly the saved 12 cycles.
All model checkpoints are embedded in JSON to respect the two-file write scope.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields as dataclass_fields, replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import platform
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
from scipy.ndimage import map_coordinates
import skfmm
from threadpoolctl import threadpool_limits

from config import InversionConfig
from experiment_data import load_tomography_experiment
from instruments.instruments_coords import metric_to_cell_coord
from instruments.likelihood import PickNoise
from velocity_model import VelocityModel, block_average_slowness

RESULTS = Path(__file__).with_name("flat4_results.json")
EXPERIMENT_ID = "flat4_20x7x7km_52st_1920ev_grid24x8x10_z0.5-5.5_noise01_pick001"
RADIUS_CELLS = 3.0
_WORKER = {}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def baseline_field(velocity, cell_size, source_m):
    """Original order-2 FMM, ORIGINAL velocity in marching, then inside overwrite."""
    return _frozen_field(velocity, cell_size, source_m, homogenize=False)


def homogenized_field(velocity, cell_size, source_m):
    """Only difference: march with a COPY, velocity[d <= 3h] = v_source."""
    return _frozen_field(velocity, cell_size, source_m, homogenize=True)


def _frozen_field(velocity, cell_size, source_m, *, homogenize):
    velocity = np.asarray(velocity, dtype=np.float64)
    source = np.asarray(source_m, dtype=np.float64)
    extent = np.asarray(velocity.shape) * cell_size
    if source.shape != (3,) or np.any(source < 0) or np.any(source > extent):
        raise ValueError(f"source {source_m} is outside the model domain")
    offsets = [(np.arange(n) + 0.5) * cell_size - s for n, s in zip(velocity.shape, source)]
    distance = np.sqrt(offsets[0][:, None, None] ** 2
                       + offsets[1][None, :, None] ** 2
                       + offsets[2][None, None, :] ** 2)
    source_cell = tuple(min(int(s // cell_size), n - 1) for s, n in zip(source, velocity.shape))
    source_velocity = velocity[source_cell]
    radius = RADIUS_CELLS * cell_size
    phi = distance - radius
    inside = phi <= 0.0
    if inside.all():
        return (distance / source_velocity).astype(np.float32)
    if homogenize:
        velocity = velocity.copy()
        velocity[inside] = source_velocity
    times = np.asarray(skfmm.travel_time(phi, velocity, dx=cell_size, order=2), dtype=np.float64)
    times += radius / source_velocity
    times[inside] = distance[inside] / source_velocity
    return times.astype(np.float32)


def sample_events(field, event_coords):
    # float64 output avoids quantizing interpolated values back to float32.
    return map_coordinates(field, event_coords.T, order=1, mode="nearest",
                           prefilter=False, output=np.float64)


def _init_field_worker(velocity, cell_size, mode, event_coords=None):
    _WORKER.update(velocity=velocity, cell_size=cell_size, mode=mode, event_coords=event_coords)
    _WORKER["thread_limits"] = threadpool_limits(limits=1)


def _field_worker(station):
    fn = baseline_field if _WORKER["mode"] == "baseline" else homogenized_field
    return fn(_WORKER["velocity"], _WORKER["cell_size"], station)


def _paired_field_worker(station):
    velocity, h = _WORKER["velocity"], _WORKER["cell_size"]
    before = baseline_field(velocity, h, station)
    after = homogenized_field(velocity, h, station)
    return (sample_events(before, _WORKER["event_coords"]),
            sample_events(after, _WORKER["event_coords"]),
            float(np.max(np.abs(after - before))))


def local_station_fields(model, stations_m, n_workers, *, mode):
    stations = np.asarray(stations_m, dtype=np.float64)
    result = np.empty((len(stations),) + model.shape, dtype=np.float32)
    with mp.get_context("fork").Pool(min(n_workers, len(stations)),
            initializer=_init_field_worker,
            initargs=(model.velocity, model.cell_size, mode)) as pool:
        for i, field in enumerate(pool.imap(_field_worker, stations)):
            result[i] = field
    return result


def discover_runs():
    runs = {}
    for label in ("fixed-c500", "hard-c500", "fixed-c1000", "hard-c1000"):
        candidates = sorted((ROOT / "runs").glob(f"run_fl-{label}-flat4*"))
        if not candidates:
            raise FileNotFoundError(f"No saved {label} flat4 run")
        run = candidates[-1]
        meta = json.loads((run / "meta.json").read_text())
        if meta["source_experiment"]["id"] != EXPERIMENT_ID:
            raise ValueError("Unexpected experiment")
        if not (run / "final_model.npy").exists():
            raise ValueError(f"Incomplete run: {run}")
        runs[label] = (run, meta)
    return runs


def distribution(values):
    v = np.asarray(values, dtype=np.float64)
    return {"mean": float(v.mean()), "rms": float(np.sqrt(np.mean(v ** 2))),
            "min": float(v.min()), "p50": float(np.percentile(v, 50)),
            "p95": float(np.percentile(v, 95)), "max": float(v.max())}


def centered(values):
    values = np.asarray(values, dtype=np.float64)
    return values - values.mean(axis=1, keepdims=True)


def residual_metrics(predicted, observed):
    residual = centered(predicted - observed)
    station_mean = residual.mean(axis=0)
    return {"centered_rms_s": float(np.sqrt(np.mean(residual ** 2))),
            "centered_max_abs_s": float(np.max(np.abs(residual))),
            "station_centered_mean_s": station_mean.tolist(),
            "station_systematic_rms_s": float(np.sqrt(np.mean(station_mean ** 2))),
            "event_centered_rms_distribution_s": distribution(np.sqrt(np.mean(residual ** 2, axis=1)))}


def model_metrics(velocity, cell_size, data):
    reference = block_average_slowness(data.source_model.velocity, data.source_model.cell_size_m,
                                       velocity.shape, cell_size)
    error = velocity - reference
    lateral = velocity - velocity.mean(axis=(0, 1), keepdims=True)
    station_cells = np.floor(data.station_coordinates_m / cell_size).astype(int)
    station_mask = np.zeros(velocity.shape, dtype=bool)
    station_mask[tuple(station_cells.T)] = True
    xy = np.stack(np.meshgrid((np.arange(velocity.shape[0]) + 0.5) * cell_size,
                              (np.arange(velocity.shape[1]) + 0.5) * cell_size,
                              indexing="ij"), axis=-1)
    horizontal_distance = np.sqrt(np.min(np.sum(
        (xy[:, :, None, :] - data.station_coordinates_m[None, None, :, :2]) ** 2, axis=-1), axis=-1))
    result = {"rms_error_m_s": float(np.sqrt(np.mean(error ** 2))),
              "mean_abs_percentage_error": float(np.mean(np.abs(error) / reference) * 100),
              "velocity_min_m_s": float(velocity.min()), "velocity_max_m_s": float(velocity.max()),
              "horizontal_std_by_depth_m_s": velocity.std(axis=(0, 1)).tolist(),
              "horizontal_mean_by_depth_m_s": velocity.mean(axis=(0, 1)).tolist(),
              "station_cell_error_m_s": error[station_mask].tolist(),
              "station_cell_lateral_anomaly_m_s": lateral[station_mask].tolist(),
              "station_cell_rms_error_m_s": float(np.sqrt(np.mean(error[station_mask] ** 2))),
              "station_cell_lateral_rms_m_s": float(np.sqrt(np.mean(lateral[station_mask] ** 2)))}
    for depth in (500, 1000, 2000):
        mask = (np.arange(velocity.shape[2]) + 0.5) * cell_size < depth
        if not mask.any():
            continue
        result[f"top_{depth}m"] = {
            "rms_error_m_s": float(np.sqrt(np.mean(error[:, :, mask] ** 2))),
            "lateral_rms_m_s": float(np.sqrt(np.mean(lateral[:, :, mask] ** 2))),
            "lateral_max_abs_m_s": float(np.max(np.abs(lateral[:, :, mask])))}
    for name, mask in (("near_station_xy_le750m", horizontal_distance <= 750),
                       ("away_station_xy_gt750m", horizontal_distance > 750)):
        if mask.any():
            result[name] = {"top_cell_error_rms_m_s": float(np.sqrt(np.mean(error[:, :, 0][mask] ** 2))),
                            "top_cell_lateral_rms_m_s": float(np.sqrt(np.mean(lateral[:, :, 0][mask] ** 2))),
                            "n_columns": int(mask.sum())}
    return result


def save_results(results):
    # No temporary files: the permitted write scope contains exactly two paths.
    RESULTS.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


def load_results():
    if not RESULTS.exists():
        raise FileNotFoundError("Run the fields stage first")
    return json.loads(RESULTS.read_text())


def provenance(runs, data, workers):
    git = subprocess.run(["git", "--no-pager", "rev-parse", "HEAD"], cwd=ROOT,
                         check=True, capture_output=True, text=True).stdout.strip()
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "experiment_id": EXPERIMENT_ID,
            "git_head_at_fields": git, "python": sys.version, "platform": platform.platform(),
            "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "skfmm": skfmm.__version__},
            "workers": workers, "radius_cells": RADIUS_CELLS, "fine_cell_size_m": 125.0,
            "difference_sign": "homogenized minus baseline; arithmetic station mean removed per event",
            "residual_sign": "predicted minus observed; arithmetic station mean removed per event",
            "ray_baseline": "production c381cfd station-cell-only inside-sphere deposition, unchanged",
            "baseline_formula": "skfmm.travel_time(d-3h, ORIGINAL velocity, dx=h, order=2)+3h/v_source; inside=d/v_source; float32",
            "patched_formula": "same; before marching COPY velocity and set velocity[d<=3h]=v_source",
            "scope": "Only flat4_controls.py and flat4_results.json; all checkpoints embedded in JSON",
            "initial_model_prior": False, "n_events": len(data.event_ids), "n_stations": len(data.station_ids),
            "event_ids": list(data.event_ids), "station_ids": list(data.station_ids),
            "station_coordinates_m": data.station_coordinates_m.tolist(),
            "observation_noise_sigmas": list(data.noise_sigmas),
            "production_sha256_at_fields": {name: sha256(ROOT / name) for name in (
                "eikonal.py", "raytracing.py", "tomography/tomography_em.py", "tomography/tomography_math.py")},
            "saved_runs": {label: {"path": str(run.relative_to(ROOT)),
                "initial_sha256": sha256(run / "initial_model.npy"),
                "final_sha256": sha256(run / "final_model.npy"), "run_params": meta["run_params"]}
                for label, (run, meta) in runs.items()}}


def compare_fields(workers):
    started = time.perf_counter()
    runs = discover_runs()
    data = load_tomography_experiment(EXPERIMENT_ID)
    assert len(data.event_ids) == 1920 and len(data.station_ids) == 52
    results = {"provenance": provenance(runs, data, workers), "field_comparisons": {}}
    cases = []
    for size in (500, 1000):
        run, meta = runs[f"fixed-c{size}"]
        hard_run, _ = runs[f"hard-c{size}"]
        assert np.array_equal(np.load(run / "initial_model.npy"), np.load(hard_run / "initial_model.npy"))
        cases.append((f"initial-c{size}", run / "initial_model.npy", meta))
    cases.extend((f"final-{label}", run / "final_model.npy", meta) for label, (run, meta) in runs.items())
    for label, path, meta in cases:
        tick = time.perf_counter()
        params = meta["run_params"]
        coarse = VelocityModel(np.load(path), params["cell_size"])
        fine = coarse.refined(params["subdivision"], params["slowness_interpolation"])
        assert fine.cell_size == 125.0
        coords = metric_to_cell_coord(data.reference_event_coordinates_m, fine.cell_size)
        with mp.get_context("fork").Pool(min(workers, 52), initializer=_init_field_worker,
                initargs=(fine.velocity, fine.cell_size, "paired", coords)) as pool:
            pairs = list(pool.imap(_paired_field_worker, data.station_coordinates_m))
        before = np.stack([p[0] for p in pairs], axis=1)
        after = np.stack([p[1] for p in pairs], axis=1)
        delta = after - before
        dc = centered(delta)
        event_rms = np.sqrt(np.mean(dc ** 2, axis=1))
        event_max = np.max(np.abs(dc), axis=1)
        station_mean = dc.mean(axis=0)
        row = {"model_path": str(path.relative_to(ROOT)), "model_sha256": sha256(path),
               "coarse_cell_size_m": coarse.cell_size,
               "n_event_station_pairs": int(delta.size), "wall_seconds": time.perf_counter() - tick,
               "raw_delta_s": distribution(delta), "centered_rms_s": float(np.sqrt(np.mean(dc ** 2))),
               "centered_max_abs_s": float(np.max(np.abs(dc))),
               "event_centered_rms_s": event_rms.tolist(), "event_centered_max_abs_s": event_max.tolist(),
               "event_centered_rms_distribution_s": distribution(event_rms),
               "event_centered_max_abs_distribution_s": distribution(event_max),
               "station_raw_mean_delta_s": delta.mean(axis=0).tolist(),
               "station_centered_mean_delta_s": station_mean.tolist(),
               "station_centered_std_across_events_s": dc.std(axis=0).tolist(),
               "station_centered_max_abs_delta_s": np.max(np.abs(dc), axis=0).tolist(),
               "station_systematic_rms_s": float(np.sqrt(np.mean(station_mean ** 2))),
               "station_systematic_max_abs_s": float(np.max(np.abs(station_mean))),
               "systematic_fraction_of_centered_delta_energy": (
                   float(np.mean(station_mean ** 2) / np.mean(dc ** 2)) if np.any(dc) else 0.0),
               "grid_max_abs_delta_by_station_s": [p[2] for p in pairs],
               "baseline_residuals": residual_metrics(before, data.arrival_times_s),
               "homogenized_residuals": residual_metrics(after, data.arrival_times_s),
               "saved_model_metrics": model_metrics(coarse.velocity, coarse.cell_size, data)}
        results["field_comparisons"][label] = row
        save_results(results)
        print(f"FIELDS {label}: centered RMS {row['centered_rms_s'] * 1000:.6f} ms; "
              f"max {row['centered_max_abs_s'] * 1000:.6f} ms; station systematic RMS "
              f"{row['station_systematic_rms_s'] * 1000:.6f} ms; {row['wall_seconds']:.2f}s", flush=True)
    results["fields_wall_seconds"] = time.perf_counter() - started
    save_results(results)


def prepare_runtime(workers):
    import tomography.tomography_em as em
    import tomography.tomography_events as event_module
    import raytracing
    assert event_module.SOURCE_RADIUS_CELLS == RADIUS_CELLS
    from numba.core.caching import NullCache
    # Do not create/refresh production __pycache__ or Numba cache artifacts.
    for value in vars(raytracing).values():
        if hasattr(value, "_cache"):
            value._cache = NullCache()
    from instruments import likelihood
    for value in vars(likelihood).values():
        if hasattr(value, "_cache"):
            value._cache = NullCache()
    spec = importlib.util.spec_from_file_location(
        "flat4_saved_fixed_oracle", ROOT / "studies/fixed_hypocentres/run_fixed.py")
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    oracle.install_true_hypocentres(EXPERIMENT_ID)
    em.warm_up_jit()
    runs = discover_runs()
    run, meta = runs["fixed-c500"]
    allowed = {f.name for f in dataclass_fields(InversionConfig)}
    params = {k: v for k, v in meta["run_params"].items() if k in allowed}
    config = replace(InversionConfig(**params), n_workers=workers, save_runs=False)
    data = load_tomography_experiment(EXPERIMENT_ID)
    noise = PickNoise(0.01, 0.01, 0.02)
    assert config.n_cycles == 12 and config.cell_size == 500 and config.subdivision == 4
    return em, run, config, data, noise


def run_cycles(stage, workers, budget_seconds):
    results = load_results()
    if stage == "invert" and not results.get("baseline_reproduction", {}).get("passed"):
        raise ValueError("Baseline reproduction must pass before patched inversion")
    from tomography.tomography_logging import summarize_ray_status
    started = time.perf_counter()
    em, saved_run, config, data, noise = prepare_runtime(workers)
    model = VelocityModel(np.load(saved_run / "initial_model.npy"), config.cell_size)
    initial = model.velocity.copy()
    cycles = 2 if stage == "reproduce" else config.n_cycles
    mode = "baseline" if stage == "reproduce" else "homogenized"
    key = "baseline_reproduction" if stage == "reproduce" else "patched_fixed_c500"
    section = {"status": "running", "forward": mode, "config": asdict(config),
               "fixed_true_hypocentres": True, "initial_model_prior": False,
               "cycle_models_are": "post-update; model_after_cycle_1 matches saved iter_1/model.npy",
               "initial_model_sha256": sha256(saved_run / "initial_model.npy"),
               "models_shape": list(model.shape), "cycles": [], "models_m_s": [],
               "initial_model_metrics": model_metrics(initial, config.cell_size, data),
               "budget_seconds": budget_seconds, "runtime_setup_seconds": time.perf_counter() - started,
               "production_sha256_at_run": {name: sha256(ROOT / name) for name in (
                   "raytracing.py", "tomography/tomography_em.py", "tomography/tomography_math.py")}}
    results[key] = section
    save_results(results)
    original_run_events = em.run_events
    current = {}

    def overridden_fields(fine, stations_m, n_workers):
        tick = time.perf_counter()
        times = local_station_fields(fine, stations_m, n_workers, mode=mode)
        current["forward_seconds"] = time.perf_counter() - tick
        coords = metric_to_cell_coord(data.reference_event_coordinates_m, fine.cell_size)
        predicted = np.stack([sample_events(field, coords) for field in times], axis=1)
        current["pre_update_residuals"] = residual_metrics(predicted, data.arrival_times_s)
        return times

    def instrumented_events(*args, **kwargs):
        tick = time.perf_counter()
        hessian, rhs, logs = original_run_events(*args, **kwargs)
        current["events_and_normal_equations_seconds"] = time.perf_counter() - tick
        status = summarize_ray_status(logs, len(data.station_ids))
        assert status is not None and status["expected"] == 1920 * 52
        current["rays"] = status
        return hessian, rhs, logs

    em.station_travel_time_fields = overridden_fields
    em.run_events = instrumented_events
    for iteration in range(cycles):
        elapsed = time.perf_counter() - started
        if iteration and elapsed + max(r["wall_seconds"] for r in section["cycles"]) > budget_seconds:
            section["status"] = "resource_limited"
            section["resource_limit_reason"] = "Next complete cycle projected beyond requested wall-clock budget"
            break
        current.clear()
        tick = time.perf_counter()
        delta_s, _, _ = em.make_tomography_step(model, data.arrival_times_s,
            data.station_coordinates_m, noise, config, iteration=iteration, logger=None)
        velocity = em.update_velocity(model.velocity, delta_s, config.max_velocity_step_fraction)
        model = VelocityModel(velocity, config.cell_size)
        row = {"cycle": iteration + 1, "wall_seconds": time.perf_counter() - tick,
               **current, "model_metrics": model_metrics(velocity, config.cell_size, data)}
        target = saved_run / f"iter_{iteration + 1}/model.npy" if iteration < 11 else saved_run / "final_model.npy"
        saved = np.load(target)
        diff = velocity - saved
        row["difference_to_saved_baseline_m_s"] = {"rms": float(np.sqrt(np.mean(diff ** 2))),
                                                  "max_abs": float(np.max(np.abs(diff)))}
        if stage == "reproduce":
            saved_delta = np.load(saved_run / f"iter_{iteration}/delta_s.npy")
            row["delta_s_max_abs_difference_to_saved"] = float(np.max(np.abs(delta_s - saved_delta)))
            row["matches_saved_at_atol_1e-7_m_s"] = bool(np.allclose(velocity, saved, rtol=0, atol=1e-7))
            row["saved_ray_status"] = json.loads((saved_run / f"iter_{iteration}/ray_status.json").read_text())
        section["cycles"].append(row)
        section["models_m_s"].append(velocity.tolist())
        section["wall_seconds"] = time.perf_counter() - started
        save_results(results)
        print(f"{mode.upper()} cycle {iteration + 1}/{cycles}: {row['wall_seconds']:.2f}s; "
              f"rays failed {row['rays']['failed']}/{row['rays']['expected']}; "
              f"model RMS {row['model_metrics']['rms_error_m_s']:.6f} m/s; "
              f"vs saved max {row['difference_to_saved_baseline_m_s']['max_abs']:.9g} m/s", flush=True)
        if stage == "reproduce" and not row["matches_saved_at_atol_1e-7_m_s"]:
            section["status"] = "reproduction_failed"
            break
    if section["status"] == "running":
        section["status"] = "complete"
    section["wall_seconds"] = time.perf_counter() - started
    section["completed_cycles"] = len(section["cycles"])
    if stage == "reproduce":
        section["passed"] = (section["status"] == "complete" and len(section["cycles"]) == 2
                             and all(r["matches_saved_at_atol_1e-7_m_s"] for r in section["cycles"]))
    else:
        final_fine = model.refined(config.subdivision, config.slowness_interpolation)
        tick = time.perf_counter()
        section["final_model_residuals"] = {}
        for forward in ("baseline", "homogenized"):
            times = local_station_fields(final_fine, data.station_coordinates_m, workers, mode=forward)
            coords = metric_to_cell_coord(data.reference_event_coordinates_m, final_fine.cell_size)
            predicted = np.stack([sample_events(field, coords) for field in times], axis=1)
            section["final_model_residuals"][forward] = residual_metrics(predicted, data.arrival_times_s)
            del times
        section["final_cross_evaluation_seconds"] = time.perf_counter() - tick
        section["wall_seconds_including_cross_evaluation"] = time.perf_counter() - started
    save_results(results)


def validate_local_controls():
    from instruments.instruments_coords import sample_cell_centered_trilinear_batch
    rng = np.random.default_rng(381)
    velocity = rng.uniform(3500, 6000, (9, 8, 7))
    original = velocity.copy()
    velocity.flags.writeable = False
    baseline = baseline_field(velocity, 125, (200, 300, 0))
    homogenized_field(velocity, 125, (200, 300, 0))
    assert np.array_equal(velocity, original)
    uniform = np.full((9, 8, 7), 4000.0)
    assert np.array_equal(baseline_field(uniform, 125, (200, 300, 0)),
                          homogenized_field(uniform, 125, (200, 300, 0)))
    positions = rng.uniform((0, 0, 0), (8, 7, 6), (20, 3))
    expected = np.array([sample_cell_centered_trilinear_batch(baseline, p) for p in positions])
    difference = float(np.max(np.abs(sample_events(baseline, positions) - expected)))
    assert difference < 1e-14
    return {"readonly_velocity_unchanged": True, "uniform_baseline_equals_patched": True,
            "sampling_agrees_with_production_max_abs_s": difference}


def summarize_results():
    results = load_results()
    assert results["baseline_reproduction"]["passed"]
    patched = results["patched_fixed_c500"]
    assert patched["completed_cycles"] > 0
    runs = discover_runs()
    saved_run, _ = runs["fixed-c500"]
    data = load_tomography_experiment(EXPERIMENT_ID)
    completed = patched["completed_cycles"]
    baseline_path = saved_run / "final_model.npy" if completed == 12 else saved_run / f"iter_{completed}/model.npy"
    final_baseline = np.load(baseline_path)
    final_patched = np.asarray(patched["models_m_s"][-1])
    a = model_metrics(final_baseline, 500, data)
    b = patched["cycles"][-1]["model_metrics"]
    comparisons = {}

    def add_metric(name, baseline_value, patched_value):
        comparisons[name] = {"baseline": baseline_value, "patched": patched_value,
            "change": patched_value - baseline_value,
            "relative_change_pct": (patched_value / baseline_value - 1) * 100}

    for name in ("rms_error_m_s", "mean_abs_percentage_error", "station_cell_rms_error_m_s",
                 "station_cell_lateral_rms_m_s"):
        add_metric(name, a[name], b[name])
    for region in ("top_500m", "top_1000m", "top_2000m", "near_station_xy_le750m", "away_station_xy_gt750m"):
        for metric in a[region]:
            if metric != "n_columns":
                add_metric(f"{region}.{metric}", a[region][metric], b[region][metric])
    base_fit = results["field_comparisons"]["final-fixed-c500"]["baseline_residuals"]
    patch_fit = patched["final_model_residuals"]["homogenized"]
    add_metric("own_forward_centered_residual_rms_s", base_fit["centered_rms_s"], patch_fit["centered_rms_s"])
    diff = final_patched - final_baseline
    index = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
    xyz = (np.asarray(index) + 0.5) * 500
    failed = sum(row["rays"]["failed"] for row in patched["cycles"])
    expected = sum(row["rays"]["expected"] for row in patched["cycles"])
    assert results["baseline_reproduction"]["production_sha256_at_run"] == patched["production_sha256_at_run"]
    for name, digest in patched["production_sha256_at_run"].items():
        assert results["provenance"]["production_sha256_at_fields"][name] == digest
    baseline_quality = [json.loads(line) for line in (saved_run / "quality.jsonl").read_text().splitlines()]
    trajectories = []
    for i, row in enumerate(patched["cycles"]):
        baseline_path = saved_run / f"iter_{i + 1}/model.npy" if i < 11 else saved_run / "final_model.npy"
        baseline_model = np.load(baseline_path)
        base_metrics = model_metrics(baseline_model, 500, data)
        trajectories.append({"cycle": i + 1, "baseline_rms_error_m_s": baseline_quality[i]["rms_m_s"],
            "patched_rms_error_m_s": row["model_metrics"]["rms_error_m_s"],
            "baseline_top500_lateral_rms_m_s": base_metrics["top_500m"]["lateral_rms_m_s"],
            "patched_top500_lateral_rms_m_s": row["model_metrics"]["top_500m"]["lateral_rms_m_s"],
            "baseline_station_cell_lateral_rms_m_s": base_metrics["station_cell_lateral_rms_m_s"],
            "patched_station_cell_lateral_rms_m_s": row["model_metrics"]["station_cell_lateral_rms_m_s"]})
    summary = {"script_sha256": sha256(__file__), "checks": validate_local_controls(),
        "input_array_sha256": {name: hashlib.sha256(np.asarray(value, dtype=np.float64).tobytes()).hexdigest()
            for name, value in (("observed_arrivals", data.arrival_times_s),
                                ("true_event_coordinates", data.reference_event_coordinates_m),
                                ("station_coordinates", data.station_coordinates_m))},
        "fields_ms": {label: {"centered_rms": row["centered_rms_s"] * 1000,
            "centered_max_abs": row["centered_max_abs_s"] * 1000,
            "station_systematic_rms": row["station_systematic_rms_s"] * 1000,
            "station_systematic_max_abs": row["station_systematic_max_abs_s"] * 1000}
            for label, row in results["field_comparisons"].items()},
        "baseline_reproduction_exact": all(row["difference_to_saved_baseline_m_s"]["max_abs"] == 0
            and row["delta_s_max_abs_difference_to_saved"] == 0
            for row in results["baseline_reproduction"]["cycles"]),
        "paired_fixed_c500": {"completed_cycles": patched["completed_cycles"],
            "wall_seconds": patched["wall_seconds"], "ray_failures": failed, "ray_expected": expected,
            "production_ray_and_update_hashes_unchanged_between_controls": True,
            "metrics": comparisons, "trajectory": trajectories,
            "model_difference_m_s": {"rms": float(np.sqrt(np.mean(diff ** 2))),
                "max_abs": float(np.max(np.abs(diff))), "max_abs_index": list(map(int, index)),
                "max_abs_cell_center_m": xyz.tolist(),
                "max_abs_cell_nearest_station_distance_m": float(np.min(np.linalg.norm(
                    data.station_coordinates_m - xyz, axis=1))),
                "rms_by_depth_m_s": np.sqrt(np.mean(diff ** 2, axis=(0, 1))).tolist()}},
        "interpretation_ru": (
            "На этих данных узкий fix изменяет поля на доли миллисекунды RMS; "
            "на initial поля и первый цикл совпадают точно. В paired fixed-c500 на 12 циклах "
            "подавления околостанционных/горизонтальных артефактов по выбранным метрикам не видно: "
            "общая ошибка практически прежняя, lateral RMS верхних 500 м и station-cell немного выше. "
            "Это не исключает корректности локального fix; контроль не устанавливает другие причины артефактов."),
        "limits": ["One paired fixed-TRUE-hypocentre c500 inversion; hard/c1000 field controls only",
            "Baseline full inversion reused from saved run after exact two-cycle reproduction",
            "Single saved noise realization; no statistical-significance claim",
            "All fields use all 52 stations and 1920 true events, not a subset",
            "Model error truth is volume-averaged slowness; lateral anomaly is relative to each horizontal plane mean",
            "Station-cell arrays follow unique coarse-cell mask order; station field offsets follow provenance.station_ids",
            "No initial-model prior; no alteration of update, clipping, ray kernel or prescribed cycle count"]}
    if patched["completed_cycles"] != 12:
        summary["interpretation_ru"] = "Ресурсно ограниченный контроль: полного paired результата на 12 циклах нет."
    results["summary"] = summary
    save_results(results)
    print(json.dumps({"fields_ms": summary["fields_ms"], "paired_final_metrics": comparisons,
                      "conclusion": summary["interpretation_ru"]}, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fields", "reproduce", "invert", "summarize"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--budget-seconds", type=float, default=900)
    args = parser.parse_args()
    if args.workers < 1 or args.budget_seconds <= 0:
        parser.error("positive workers and budget required")
    with threadpool_limits(limits=args.workers):
        if args.stage == "fields":
            compare_fields(args.workers)
        elif args.stage == "summarize":
            summarize_results()
        else:
            run_cycles(args.stage, args.workers, args.budget_seconds)
            if args.stage == "invert":
                summarize_results()


if __name__ == "__main__":
    main()
