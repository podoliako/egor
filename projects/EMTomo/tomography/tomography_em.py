from __future__ import annotations

from dataclasses import asdict, replace
from time import perf_counter
from typing import Optional

import numpy as np

from config import InversionConfig
from eikonal import station_travel_time_fields
from instruments.instruments_coords import metric_to_cell_coord
from instruments.instruments_ops import restriction_tables
from instruments.hypocentre_objective import make_refinement_objective
from instruments.likelihood import PickNoise, negative_log_likelihood, negative_log_likelihood_batch
from raytracing import compute_G_all_stations_compact, compute_G_all_stations_serial
from velocity_model import VelocityModel
from .tomography_events import EventSettings, IterationFields, run_events, validate_arrivals
from .tomography_logging import TomographyLogger, summarize_ray_status
from .tomography_math import MatrixFreeNormalEquations, accumulate_normal_equations, solve_slowness_update
from .tomography_validation import evaluate_control_rmse, split_control_arrivals

# Recorded in meta.json so that runs with different formulations are not compared.
METHOD_DESCRIPTION = {
    "weight_likelihood": "gaussian_independent_absolute_picks_marginal_origin_uniform_candidates",
    "normal_equations": "station_precision_weighted_profiled_origin",
    "candidate_selection": "likelihood_shortlist_of_n_candidates_refined_then_best_top_n",
    "travel_times": "skfmm_order2_exact_station_homogeneous_sphere_v2",
    "early_stopping": "held_out_pick_rmse_consecutive_insufficient_relative_improvements_min_3_updates",
    "control_prediction": "best_refined_training_hypocentre_precision_weighted_training_origin",
}
_RUNTIME_FIELDS = ("runs_dir", "save_runs")


def warm_up_jit() -> None:
    """Compile Numba kernels before worker processes are forked."""
    gradient = np.zeros((1, 2, 2, 2), dtype=np.float32)
    stations = np.zeros((1, 3), dtype=np.float64)
    point = np.zeros(3, dtype=np.float64)
    lo, hi = np.zeros(3), np.ones(3)
    tables = restriction_tables((1, 1, 1), 2, "nearest")
    # Do not initialize a parallel Numba/OpenMP runtime before forking workers.
    for compute_G in (compute_G_all_stations_serial, compute_G_all_stations_compact):
        compute_G(gradient, gradient, gradient, stations, point, 1.0, 0.5, 0.5, 5, lo, hi, 3.0,
                  *tables, (1, 1, 1))
    noise = PickNoise(0.01, 0.1, 0.0)
    negative_log_likelihood(np.zeros((2, 1), dtype=np.float32), np.zeros(2), noise)
    negative_log_likelihood(np.zeros(2), np.zeros(2), noise)
    fields = np.zeros((2, 2, 2, 2), dtype=np.float32)
    negative_log_likelihood_batch(fields, np.zeros((2, 2)), noise, normalization=0.0)
    make_refinement_objective(fields, np.zeros(2), noise)(point)
    indices = np.arange(2, dtype=np.int64)
    negative_log_likelihood(fields, np.zeros(2), noise, station_indices=indices)
    make_refinement_objective(fields, np.zeros(2), noise, station_indices=indices)(point)
    hessian, rhs = np.zeros((1, 1)), np.zeros(1)
    accumulate_normal_equations(hessian, rhs, np.ones((2, 1)), np.zeros(2),
                                np.ones(2, dtype=bool), np.ones(2), 1.0)
    operator = MatrixFreeNormalEquations(1)
    accumulate_normal_equations(operator, rhs, np.ones((2, 1)), np.zeros(2),
                                np.ones(2, dtype=bool), np.ones(2), 1.0)
    operator.matvec(np.ones(1))


def run_em(
    config: InversionConfig,
    initial_model: VelocityModel,
    arrivals,
    stations_m,
    noise: PickNoise,
    *,
    reference_model: Optional[VelocityModel] = None,
    source_experiment: Optional[dict] = None,
    run_name: Optional[str] = None,
    logger: Optional[TomographyLogger] = None,
) -> Optional[TomographyLogger]:
    """Alternate hypocentre hypotheses (E-step) and one damped slowness update (M-step).

    Runs at most ``config.n_cycles`` cycles, with a fixed held-out pick split.
    Stops after consecutive insufficient relative control-RMSE improvements,
    never before three full updates. Each score relocalizes events in the updated
    model using training picks only. The stopping update is retained, not rolled
    back. With no control picks, runs all requested cycles. ``reference_model``
    is used only for quality logging, never for stopping or fitting.
    Returns the run logger, or None when runs are not saved.
    Arrivals are dense (events, global stations); NaN marks unobserved P picks.
    Each event must have at least two finite picks; infinities are rejected.
    """
    stations_m = np.asarray(stations_m, dtype=np.float64)
    arrivals = validate_arrivals(arrivals, len(stations_m))
    training_arrivals, control_mask = split_control_arrivals(
        arrivals, fraction=config.control_fraction, seed=config.control_seed,
        min_training_picks=config.control_min_training_picks,
    )
    n_observed = int(np.count_nonzero(np.isfinite(arrivals)))
    n_control = int(np.count_nonzero(control_mask))
    control_split = {
        "seed": int(config.control_seed),
        "requested_fraction": float(config.control_fraction),
        "actual_fraction": n_control / n_observed if n_observed else 0.0,
        "n_observed": n_observed,
        "n_control": n_control,
        "n_training": n_observed - n_control,
        "min_training_picks": int(config.control_min_training_picks),
        "enabled": n_control > 0,
    }
    if n_control:
        print(f"[Control] Held out {n_control}/{n_observed} picks "
              f"({control_split['actual_fraction']:.2%}), seed={config.control_seed}", flush=True)
    else:
        reason = "disabled" if config.control_fraction == 0 else "no eligible control picks"
        print(f"[Control] Early stopping {reason}; running up to {config.n_cycles} updates", flush=True)
    run_name = run_name or config.run_name
    if config.save_runs and logger is None:
        logger = TomographyLogger(
            base_dir=config.runs_dir,
            run_name=run_name,
            run_version=config.run_version,
            run_tags={
                "ncand": config.n_candidates,
                "topn": config.weights_top_n,
                "mode": config.candidate_mode,
                "dmin": config.weights_min_distance,
                "lam": config.lambda_reg,
                "temp": config.temperature,
                "sub": config.subdivision,
                "cov": config.coverage_damping_power,
            },
        )

    if logger is not None:
        fine_side = initial_model.cell_size / config.subdivision
        run_params = {
            key: value for key, value in asdict(config).items() if key not in _RUNTIME_FIELDS
        }
        run_params.update(
            METHOD_DESCRIPTION,
            run_name=run_name,
            n_events=len(arrivals),
            weight_noise_relative_sigma=noise.relative_sigma,
            weight_noise_absolute_sigma_s=noise.absolute_sigma_s,
            weight_model_sigma_s=noise.model_sigma_s,
            coarse_side_m=round(initial_model.cell_size, 2),
            fine_side_m=round(fine_side, 2),
            viewer_quality_expected=reference_model is not None,
            control_n_picks=n_control,
            control_actual_fraction=control_split["actual_fraction"],
        )
        grid_info = {
            "coarse_cell_size": initial_model.cell_size,
            "coarse_shape": list(initial_model.shape),
            "coarse_side_m": [initial_model.cell_size * n for n in initial_model.shape],
            "fine_cell_size": fine_side,
            "fine_shape": [n * config.subdivision for n in initial_model.shape],
            "fine_side_m": [initial_model.cell_size * n for n in initial_model.shape],
        }
        logger.save_initial_model(initial_model.velocity)
        logger._save_npy(logger.run_dir / "control_mask.npy", control_mask)
        logger._save_json(logger.run_dir / "control_split.json", control_split)
        # meta.json is the ready marker consumed by the live viewer.
        logger.save_meta(
            run_params=run_params,
            station_locs=stations_m.tolist(),
            event_locs=[],
            grid_info=grid_info,
            source_experiment=source_experiment,
        )

    model = initial_model
    previous_rmse = None
    insufficient_steps = 0
    stop_reason = "max_cycles"
    for iteration in range(config.n_cycles):
        print(f"{iteration + 1}/{config.n_cycles}", flush=True)
        if logger is not None:
            logger.start_iteration(iteration)
            logger.save_iteration_model(iteration, model.velocity)

        step_config = (replace(config, temperature=config.temperature_schedule[iteration])
                       if config.temperature_schedule is not None else config)
        delta_s, sensitivity, confidence = make_tomography_step(
            model, training_arrivals, stations_m, noise, step_config, iteration=iteration, logger=logger,
        )
        model = VelocityModel(
            update_velocity(model.velocity, delta_s, config.max_velocity_step_fraction),
            model.cell_size,
        )

        control_score = None
        if n_control:
            started = perf_counter()
            rmse = evaluate_control_rmse(
                model, training_arrivals, arrivals, control_mask, stations_m, noise, step_config,
            )
            improvement = None
            if previous_rmse is not None:
                # Zero is already a perfect score: it cannot improve relatively.
                improvement = ((previous_rmse - rmse) / previous_rmse
                               if previous_rmse > 0 else None)
                significant = rmse < previous_rmse * (1 - config.early_stopping_min_relative_improvement)
                insufficient_steps = 0 if significant else insufficient_steps + 1
            should_stop = iteration + 1 >= 3 and insufficient_steps >= config.early_stopping_patience
            control_score = {
                "iter": iteration,
                "n_control": n_control,
                "rmse_s": rmse,
                "relative_improvement": improvement,
                "insufficient_steps": insufficient_steps,
                "stop": should_stop,
                "wall_s": perf_counter() - started,
            }
            print(f"[Control] RMSE={rmse:.6g} s; insufficient improvements "
                  f"{insufficient_steps}/{config.early_stopping_patience}", flush=True)
            previous_rmse = rmse

        if logger is not None:
            logger.end_iteration(iteration)
            logger.save_delta_s(iteration, delta_s)
            logger.save_inversion_diagnostics(iteration, sensitivity, confidence)
            if reference_model is not None:
                logger.save_quality(iteration, *model_errors(model.velocity, reference_model.velocity))
            if control_score is not None:
                logger._save_json(logger.iter_dir(iteration) / "control.json", control_score)
            if config.temperature_schedule is not None:
                logger._save_json(
                    logger.iter_dir(iteration) / "temperature.json",
                    {"iter": iteration, "temperature": float(step_config.temperature)},
                )
            logger.complete_iteration(iteration)

        if control_score is not None and control_score["stop"]:
            stop_reason = "control_rmse_plateau"
            print(
                f"[EM] Stopping after {iteration + 1} velocity updates: control RMSE "
                f"did not improve by more than {config.early_stopping_min_relative_improvement:.2%} "
                f"for {insufficient_steps} consecutive steps",
                flush=True,
            )
            break

    if logger is not None:
        logger.save_final_model(model.velocity)
        logger.save_timing_summary()
        logger._save_json(logger.run_dir / "stopping.json", {
            "reason": stop_reason,
            "completed_updates": iteration + 1,
            "control_rmse_s": previous_rmse,
            "insufficient_steps": insufficient_steps,
        })
        print(f"[TomographyLogger] Run saved to: {logger.run_dir}")
    return logger


def _normal_solver(config: InversionConfig, n_vox: int) -> str:
    if config.linear_solver == "matrix_free":
        return "matrix_free"
    if (config.linear_solver == "auto" and config.lambda_reg > 0
            and n_vox >= config.matrix_free_min_voxels):
        return "matrix_free"
    return "dense"


def make_tomography_step(
    model: VelocityModel,
    arrivals: np.ndarray,
    stations_m: np.ndarray,
    noise: PickNoise,
    config: InversionConfig,
    iteration: int = 0,
    logger: Optional[TomographyLogger] = None,
    diagnostics: Optional[dict] = None,
):
    """One EM cycle; returns ``(delta_s, sensitivity_diagonal, coverage_confidence)``."""
    arrivals = validate_arrivals(arrivals, len(stations_m))
    wall = {}
    started = perf_counter()
    fine = model.refined(config.subdivision, config.slowness_interpolation)
    wall["refine_model_s"] = perf_counter() - started
    started = perf_counter()
    times = station_travel_time_fields(fine, stations_m, config.n_workers)
    wall["station_fields_s"] = perf_counter() - started
    started = perf_counter()
    if logger is not None:
        logger.save_station_fields(iteration, times)
    wall["save_fields_s"] = perf_counter() - started
    started = perf_counter()
    fields = IterationFields.from_times(
        times, metric_to_cell_coord(stations_m, fine.cell_size), fine.cell_size,
        config.subdivision, noise, config.slowness_interpolation,
    )
    del times, fine
    wall["prepare_fields_s"] = perf_counter() - started
    settings = EventSettings(
        subdivision=config.subdivision,
        slowness_interpolation=config.slowness_interpolation,
        n_candidates=config.n_candidates,
        weights_top_n=config.weights_top_n,
        weights_min_distance=config.weights_min_distance,
        candidate_mode=config.candidate_mode,
        temperature=config.temperature,
        noise=noise,
        log_g_per_weight=config.log_g_per_weight and logger is not None,
        log_misfit=logger is not None and logger.save_misfit,
        log_diagnostics=logger is not None,
        likelihood_batch_size=config.likelihood_batch_size,
        likelihood_batch_memory_mb=config.likelihood_batch_memory_mb,
        compact_rays=config.compact_rays,
        fast_refinement=config.fast_refinement,
        normal_solver=_normal_solver(config, fields.n_vox),
    )
    event_diagnostics = {}
    started = perf_counter()
    hessian, rhs, event_logs = run_events(
        arrivals, fields, settings, config.n_workers, diagnostics=event_diagnostics,
    )
    wall["events_s"] = perf_counter() - started

    started = perf_counter()
    if logger is not None:
        ray_counts = None
        for event, log in event_logs:
            logger.save_event_data(iteration, event, log)
            for counts in log.ray_count_per_weight.values():
                if ray_counts is None:
                    ray_counts = np.zeros(model.shape, dtype=np.result_type(counts.dtype, np.int64))
                ray_counts += counts
        if ray_counts is not None:
            logger.save_ray_count(iteration, ray_counts)
        del ray_counts
        ray_status = logger.save_ray_status(iteration, event_logs, len(fields.stations))
    else:
        ray_status = summarize_ray_status(event_logs, len(fields.stations))

    del fields, event_logs
    wall["save_events_s"] = perf_counter() - started
    if ray_status is None:
        print("[Rays] unavailable: missing active-hypothesis masks", flush=True)
    else:
        warning = " — WARNING: failed rays" if ray_status["failed"] else ""
        print(
            f"[Rays] {ray_status['reached']}/{ray_status['expected']} reached, "
            f"{ray_status['failed']} failed, {ray_status['used']} used{warning}",
            flush=True,
        )

    solver_diagnostics = {}
    started = perf_counter()
    result = solve_slowness_update(
        hessian,
        rhs,
        model.shape,
        lambda_reg=config.lambda_reg,
        coverage_damping_power=config.coverage_damping_power,
        coverage_floor=config.coverage_floor,
        coverage_reference_percentile=config.coverage_reference_percentile,
        solver="legacy" if config.linear_solver == "legacy" else "auto",
        rtol=config.solver_rtol,
        maxiter=config.solver_maxiter,
        diagnostics=solver_diagnostics,
    )
    wall["solve_s"] = perf_counter() - started
    performance = {"wall_s": wall, "events": event_diagnostics, "solver": solver_diagnostics}
    if diagnostics is not None:
        diagnostics.update(performance)
    if logger is not None:
        logger.save_performance(iteration, performance)
    return result


def update_velocity(
    velocity: np.ndarray,
    delta_s: np.ndarray,
    max_step_fraction: Optional[float],
) -> np.ndarray:
    """Apply a slowness increment, then limit each cell to ``±max_step_fraction`` of its velocity."""
    slowness = 1.0 / velocity + delta_s
    if max_step_fraction is None:
        if np.any(slowness <= 0):
            raise ValueError("Slowness update is non-positive; set max_velocity_step_fraction")
        return 1.0 / slowness
    if not 0.0 < max_step_fraction < 1.0:
        raise ValueError("max_velocity_step_fraction must be in (0, 1)")
    proposed = 1.0 / np.maximum(slowness, 1e-12)
    return np.clip(proposed, velocity * (1.0 - max_step_fraction), velocity * (1.0 + max_step_fraction))


def model_errors(velocity: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """Mean absolute percentage deviation and RMS (m/s) from the reference."""
    difference = np.asarray(velocity) - np.asarray(reference)
    return (
        float(np.mean(np.abs(difference) / np.abs(reference)) * 100.0),
        float(np.sqrt(np.mean(difference ** 2))),
    )
