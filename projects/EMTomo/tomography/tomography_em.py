from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from instruments.instruments import compute_station_travel_time_fields, metric_to_cell_index
from raytracing import compute_G_all_stations, compute_G_all_stations_serial
from .tomography_events import _process_event_single, _run_events_parallel
from .tomography_logging import TomographyLogger
from .tomography_math import _solve_delta_s


def warm_up_jit() -> None:
    from raytracing import _rasterize_nb, _trace_ray_nb  # noqa: F401

    dummy = np.zeros((1, 2, 2, 2), dtype=np.float64)
    sl = np.zeros((1, 3), dtype=np.float64)
    epic = np.zeros(3, dtype=np.float64)
    lo = np.zeros(3, dtype=np.float64)
    hi = np.ones(3, dtype=np.float64)
    compute_G_all_stations_serial(
        dummy, dummy, dummy, sl, epic, 1.0, 1.0, 1.0, 0.5, 0.5, 5, lo, hi
    )
    compute_G_all_stations(
        dummy, dummy, dummy, sl, epic, 1.0, 1.0, 1.0, 0.5, 0.5, 5, lo, hi
    )


def run_em(
    n_cycles,
    initial_model,
    arrivals_table,
    station_locs,
    wave_type: str = "P",
    solver: Union[str, object] = "skfmm",
    lambda_reg: float = 1e-3,
    temperature: float = 1.0,
    weights_top_n: int = 1,
    weights_min_distance: int = 2,
    subdivision: int = 1,
    n_workers: int = 1,
    log_G_per_weight: bool = False,
    v_bounds: Optional[Tuple[float, float]] = None,
    v_reg_strength: float = 0.0,
    v_left_mode: str = "exp",
    v_right_mode: str = "poly",
    v_left_rate: float = 6.0,
    v_right_rate: float = 2.0,
    v_left_power: float = 2.0,
    v_right_power: float = 2.0,
    true_model=None,
    event_locs=None,
    logger=None,
    save_runs: bool = True,
    runs_dir: str = "runs",
    slowness_interpolation: str = "nearest",
    coverage_damping_power: float = 0.0,
    coverage_floor: float = 0.05,
    coverage_reference_percentile: float = 75.0,
    max_velocity_step_fraction: Optional[float] = None,
):
    if save_runs and logger is None:
        logger = TomographyLogger(base_dir=runs_dir)

    if logger is not None:
        grid_info, coarse_side, fine_side = _build_grid_info(initial_model, subdivision)
        run_params = dict(
            n_cycles=n_cycles,
            wave_type=wave_type,
            solver=str(solver),
            lambda_reg=lambda_reg,
            temperature=temperature,
            weights_top_n=weights_top_n,
            weights_min_distance=weights_min_distance,
            subdivision=subdivision,
            slowness_interpolation=slowness_interpolation,
            n_workers=n_workers,
            coverage_damping_power=coverage_damping_power,
            coverage_floor=coverage_floor,
            coverage_reference_percentile=coverage_reference_percentile,
            max_velocity_step_fraction=max_velocity_step_fraction,
            coarse_side_m=round(coarse_side, 2),
            fine_side_m=round(fine_side, 2),
        )
        logger.save_meta(
            run_params=run_params,
            station_locs=station_locs,
            event_locs=event_locs or [],
            grid_info=grid_info,
        )
        logger.save_initial_model(initial_model)
        logger.save_true_model(true_model)

    model = initial_model

    for i in range(n_cycles):
        print(f"{i + 1}/{n_cycles}")
        if logger is not None:
            logger.start_iteration(i)
            logger.save_iteration_model(i, model)

        delta_s, sensitivity_diagonal, coverage_confidence = make_tomography_step(
            model,
            arrivals_table,
            station_locs,
            wave_type,
            solver,
            lambda_reg,
            temperature,
            weights_top_n,
            weights_min_distance,
            subdivision,
            n_workers=n_workers,
            slowness_interpolation=slowness_interpolation,
            log_G_per_weight=log_G_per_weight,
            iteration=i,
            logger=logger,
            coverage_damping_power=coverage_damping_power,
            coverage_floor=coverage_floor,
            coverage_reference_percentile=coverage_reference_percentile,
            return_diagnostics=True,
        )

        current_vp = model.get_geo_grid(subdivision=1).vp
        s = 1.0 / current_vp + delta_s
        s = np.where(s <= 1e-12, 1e-12, s)
        new_velocities = 1.0 / s
        if v_bounds and v_reg_strength > 0:
            new_velocities = _apply_velocity_bounds(
                new_velocities,
                v_min=v_bounds[0],
                v_max=v_bounds[1],
                strength=v_reg_strength,
                left_mode=v_left_mode,
                right_mode=v_right_mode,
                left_rate=v_left_rate,
                right_rate=v_right_rate,
                left_power=v_left_power,
                right_power=v_right_power,
            )
        new_velocities = _limit_velocity_update(
            current_vp,
            new_velocities,
            max_velocity_step_fraction,
        )
        model.set_vp_array(new_velocities)

        if logger is not None:
            logger.end_iteration(i)
            logger.save_delta_s(i, delta_s)
            logger.save_inversion_diagnostics(
                i,
                sensitivity_diagonal,
                coverage_confidence,
            )
            if true_model is not None:
                q = _avg_abs_percent_deviation(model, true_model)
                logger.save_quality(i, q)

    if logger is not None:
        logger.save_timing_summary()
        print(f"[TomographyLogger] Run saved to: {logger.run_dir}")

    return logger


def make_tomography_step(
    initial_model,
    arrivals_table,
    station_locs,
    wave_type: str = "P",
    solver: Union[str, object] = "skfmm",
    lambda_reg: float = 1e-3,
    temperature: float = 1.0,
    weights_top_n: int = 1,
    weights_min_distance: int = 2,
    subdivision: int = 1,
    n_workers: int = 1,
    log_G_per_weight: bool = False,
    iteration: int = 0,
    logger: Optional[TomographyLogger] = None,
    slowness_interpolation: str = "nearest",
    coverage_damping_power: float = 0.0,
    coverage_floor: float = 0.05,
    coverage_reference_percentile: float = 75.0,
    return_diagnostics: bool = False,
):
    coarse_grid = initial_model.get_geo_grid(
        subdivision=1,
        slowness_interpolation=slowness_interpolation,
    )
    coarse_shape = tuple(int(v) for v in coarse_grid.shape)

    fine_grid = initial_model.get_geo_grid(
        subdivision=subdivision,
        slowness_interpolation=slowness_interpolation,
    )
    fine_shape = tuple(int(v) for v in fine_grid.shape)
    fine_cell_size = float(fine_grid.cell_size)

    station_idx_fine = [metric_to_cell_index(s, fine_cell_size, fine_shape) for s in station_locs]

    station_fields = compute_station_travel_time_fields(
        fine_grid, station_idx_fine, wave_type, solver
    )
    sf_array = np.asarray(station_fields, dtype=np.float64)

    grads = np.stack([np.gradient(sf, 1.0, 1.0, 1.0, edge_order=1) for sf in sf_array])
    gx = np.ascontiguousarray(grads[:, 0])
    gy = np.ascontiguousarray(grads[:, 1])
    gz = np.ascontiguousarray(grads[:, 2])

    sl = np.asarray(station_idx_fine, dtype=np.float64)
    x_lo = np.zeros(3, dtype=np.float64)
    x_hi = np.asarray(fine_shape, dtype=np.float64) - 1.0

    if logger is not None:
        logger.save_station_fields(iteration, sf_array)

    if n_workers > 1:
        results = _run_events_parallel(
            arrivals_table,
            gx,
            gy,
            gz,
            sf_array,
            sl,
            x_lo,
            x_hi,
            fine_cell_size,
            subdivision,
            slowness_interpolation,
            temperature,
            weights_top_n,
            weights_min_distance,
            n_workers,
            log_G_per_weight=log_G_per_weight and logger is not None,
        )
    else:
        results = [
            _process_event_single(
                event_idx=i,
                observed=np.asarray(obs, dtype=np.float64),
                sf=sf_array,
                gx=gx,
                gy=gy,
                gz=gz,
                sl=sl,
                x_lo=x_lo,
                x_hi=x_hi,
                fine_cell_size=fine_cell_size,
                subdivision=subdivision,
                slowness_interpolation=slowness_interpolation,
                temperature=temperature,
                weights_top_n=weights_top_n,
                weights_min_distance=weights_min_distance,
                log_G_per_weight=log_G_per_weight and logger is not None,
            )
            for i, obs in enumerate(arrivals_table)
        ]

    hessian_acc = []
    rhs_acc = []
    for event_idx, (hessian_ev, rhs_ev, log_data) in enumerate(results):
        hessian_acc.append(hessian_ev)
        rhs_acc.append(rhs_ev)
        if logger is not None:
            weights, misfit, first_residuals, G_per_weight, rc_per_weight = log_data
            logger.save_event_data(
                iteration=iteration,
                event_idx=event_idx,
                weights=weights,
                misfit=misfit,
                residuals=first_residuals,
                G_per_weight=G_per_weight,
                ray_count_per_weight=rc_per_weight,
            )

    return _solve_delta_s(
        hessian=np.add.reduce(hessian_acc),
        rhs=np.add.reduce(rhs_acc),
        model_shape=coarse_shape,
        lambda_reg=lambda_reg,
        coverage_damping_power=coverage_damping_power,
        coverage_floor=coverage_floor,
        coverage_reference_percentile=coverage_reference_percentile,
        return_diagnostics=return_diagnostics,
    )


def _build_grid_info(model, subdivision: int) -> Tuple[dict, float, float]:
    coarse_grid = model.get_geo_grid(subdivision=1)
    coarse_side = float(coarse_grid.cell_size)
    fine_side = coarse_side
    grid_info = {
        "coarse_cell_size": coarse_side,
        "coarse_shape": [int(v) for v in coarse_grid.shape],
        "coarse_side_m": [coarse_side * int(v) for v in coarse_grid.shape],
    }
    if subdivision > 1:
        fine_grid = model.get_geo_grid(subdivision=subdivision)
        fine_side = float(fine_grid.cell_size)
        grid_info.update(
            {
                "fine_cell_size": fine_side,
                "fine_shape": [int(v) for v in fine_grid.shape],
                "fine_side_m": [fine_side * int(v) for v in fine_grid.shape],
            }
        )
    else:
        grid_info.update(
            {
                "fine_cell_size": coarse_side,
                "fine_shape": [int(v) for v in coarse_grid.shape],
                "fine_side_m": [coarse_side * int(v) for v in coarse_grid.shape],
            }
        )

    return grid_info, coarse_side, fine_side


def _avg_abs_percent_deviation(model, true_model, eps: float = 1e-12) -> float:
    m = model.get_geo_grid(subdivision=1).vp
    t = true_model.get_geo_grid(subdivision=1).vp
    denom = np.where(np.abs(t) < eps, np.nan, np.abs(t))
    pct = np.abs(m - t) / denom * 100.0
    return float(np.nanmean(pct))


def _limit_velocity_update(
    current_velocities: np.ndarray,
    proposed_velocities: np.ndarray,
    max_step_fraction: Optional[float],
) -> np.ndarray:
    """Apply a cellwise trust region in velocity space."""
    if max_step_fraction is None:
        return proposed_velocities
    if not 0.0 <= max_step_fraction < 1.0:
        raise ValueError("max_velocity_step_fraction must be in [0, 1)")

    current = np.asarray(current_velocities)
    lower = current * (1.0 - max_step_fraction)
    upper = current * (1.0 + max_step_fraction)
    return np.clip(proposed_velocities, lower, upper)


def _apply_velocity_bounds(
    velocities: np.ndarray,
    v_min: float,
    v_max: float,
    strength: float,
    left_mode: str,
    right_mode: str,
    left_rate: float,
    right_rate: float,
    left_power: float,
    right_power: float,
) -> np.ndarray:
    if v_min >= v_max:
        return velocities

    v = velocities.astype(np.float64, copy=True)
    eps = 1e-12

    left = v < v_min
    if np.any(left):
        dist = (v_min - v[left]) / max(v_min, eps)
        w = _unbounded_weight(dist, left_mode, left_rate, left_power)
        d = v_min - v[left]
        v[left] = v_min - d / (1.0 + strength * w)

    right = v > v_max
    if np.any(right):
        dist = (v[right] - v_max) / max(v_max, eps)
        w = _unbounded_weight(dist, right_mode, right_rate, right_power)
        d = v[right] - v_max
        v[right] = v_max + d / (1.0 + strength * w)

    return v.astype(velocities.dtype, copy=False)


def _unbounded_weight(dist, mode: str, rate: float, power: float) -> np.ndarray:
    dist = np.maximum(dist, 0.0)
    if mode == "exp":
        x = np.minimum(rate * dist, 50.0)
        w = np.expm1(x)
    else:
        w = np.power(dist, power)
    return w
