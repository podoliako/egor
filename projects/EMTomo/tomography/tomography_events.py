from __future__ import annotations

import multiprocessing as mp
from typing import Callable, Dict, Optional

import numpy as np

from instruments.instruments import (
    coarsen_G,
    compute_cellwise_pairwise_misfit,
    compute_weights_from_misfit,
)
from raytracing import compute_G_all_stations, compute_G_all_stations_serial
from .tomography_math import (
    _calculate_residuals,
    _normal_equation_contribution,
    _refine_epicenter_in_cell,
    _select_top_n_cells_by_misfit,
    _station_residuals_at_coord,
)

_MP: dict = {}


def _mp_worker_init(state: Optional[dict] = None) -> None:
    global _MP
    if state is not None:
        _MP = state
    try:
        from numba import set_num_threads # pyright: ignore[reportMissingImports]

        set_num_threads(1)
    except Exception:
        pass


def _process_event(
    observed: np.ndarray,
    sf: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
    sl: np.ndarray,
    x_lo: np.ndarray,
    x_hi: np.ndarray,
    fine_cell_size: float,
    subdivision: int,
    slowness_interpolation: str,
    temperature: float,
    weights_top_n: int,
    weights_min_distance: int,
    compute_G: Callable[..., tuple[np.ndarray, np.ndarray]],
    log_G_per_weight: bool,
) -> tuple:
    step = 0.1

    misfit = compute_cellwise_pairwise_misfit(sf, observed)
    weights_indices = _select_top_n_cells_by_misfit(
        misfit,
        weights_top_n,
        min_distance=weights_min_distance,
    )

    refined_positions = []
    refined_misfits = []
    for cell_index in weights_indices:
        position, refined_misfit = _refine_epicenter_in_cell(sf, observed, cell_index)
        refined_positions.append(position)
        refined_misfits.append(refined_misfit)

    weights_values = compute_weights_from_misfit(
        np.asarray(refined_misfits, dtype=np.float64),
        temperature=temperature,
    )
    weights = np.zeros_like(misfit, dtype=np.float64)
    logged_misfit = misfit.copy()
    for cell_index, weight_value, refined_misfit in zip(
        weights_indices, weights_values, refined_misfits
    ):
        index = tuple(cell_index)
        weights[index] = weight_value
        logged_misfit[index] = refined_misfit

    coarse_shape = tuple(int(v) // subdivision for v in sf.shape[1:])
    n_vox = int(np.prod(coarse_shape))
    hessian = np.zeros((n_vox, n_vox), dtype=np.float64)
    rhs = np.zeros(n_vox, dtype=np.float64)
    first_residuals = None
    G_per_weight: Dict[int, list[np.ndarray]] = {}
    ray_count_per_weight: Dict[int, np.ndarray] = {}

    for w_idx, (cell_index, epic, weight_val) in enumerate(
        zip(weights_indices, refined_positions, weights_values)
    ):
        if weight_val <= 0.0:
            continue
        epic = np.asarray(epic, dtype=np.float64)
        G_fine, ray_reached = compute_G(
            gx,
            gy,
            gz,
            sl,
            epic,
            fine_cell_size,
            fine_cell_size,
            fine_cell_size,
            step,
            step,
            50000,
            x_lo,
            x_hi,
        )
        G_stations = np.array([
            coarsen_G(
                G_fine[si],
                subdivision,
                slowness_interpolation=slowness_interpolation,
            )
            for si in range(G_fine.shape[0])
        ])
        residuals = _calculate_residuals(sf, observed, epic)
        station_residuals = _station_residuals_at_coord(sf, observed, epic)
        hessian_w, rhs_w = _normal_equation_contribution(
            station_sensitivities=G_stations,
            station_residuals=station_residuals,
            model_shape=coarse_shape,
            weight=float(weight_val),
            valid_stations=ray_reached,
        )
        hessian += hessian_w
        rhs += rhs_w

        if first_residuals is None:
            first_residuals = residuals
        if log_G_per_weight:
            G_per_weight[w_idx] = [G_fine[si] for si in range(G_fine.shape[0])]
        ray_count_per_weight[w_idx] = (G_stations > 0).sum(axis=0).astype(np.int16)

    log_data = (
        weights,
        logged_misfit,
        first_residuals if first_residuals is not None else np.array([]),
        G_per_weight if log_G_per_weight else None,
        ray_count_per_weight
    )
    return hessian, rhs, log_data


def _mp_event_task(packed: tuple) -> tuple:
    event_idx, observed = packed

    gx = _MP["gx"]
    gy = _MP["gy"]
    gz = _MP["gz"]
    sf = _MP["sf"]
    sl = _MP["sl"]
    x_lo = _MP["x_lo"]
    x_hi = _MP["x_hi"]
    fcs = _MP["fine_cell_size"]
    sub = _MP["subdivision"]
    si_mode = _MP["slowness_interpolation"]
    T = _MP["temperature"]
    wtn = _MP["weights_top_n"]
    wmd = _MP["weights_min_distance"]
    log_G = _MP.get("log_G_per_weight", False)

    observed = np.asarray(observed, dtype=np.float64)

    return _process_event(
        observed=observed,
        sf=sf,
        gx=gx,
        gy=gy,
        gz=gz,
        sl=sl,
        x_lo=x_lo,
        x_hi=x_hi,
        fine_cell_size=fcs,
        subdivision=sub,
        slowness_interpolation=si_mode,
        temperature=T,
        weights_top_n=wtn,
        weights_min_distance=wmd,
        compute_G=compute_G_all_stations_serial,
        log_G_per_weight=log_G,
    )


def _process_event_single(
    event_idx,
    observed,
    sf,
    gx,
    gy,
    gz,
    sl,
    x_lo,
    x_hi,
    fine_cell_size,
    subdivision,
    slowness_interpolation,
    temperature,
    weights_top_n,
    weights_min_distance,
    log_G_per_weight: bool = False,
):
    observed = np.asarray(observed, dtype=np.float64)
    return _process_event(
        observed=observed,
        sf=sf,
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
        compute_G=compute_G_all_stations,
        log_G_per_weight=log_G_per_weight,
    )


def _run_events_parallel(
    arrivals_table,
    gx,
    gy,
    gz,
    sf,
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
    log_G_per_weight: bool = False,
):
    global _MP
    _MP = dict(
        gx=gx,
        gy=gy,
        gz=gz,
        sf=sf,
        sl=sl,
        x_lo=x_lo,
        x_hi=x_hi,
        fine_cell_size=fine_cell_size,
        subdivision=subdivision,
        slowness_interpolation=slowness_interpolation,
        temperature=temperature,
        weights_top_n=weights_top_n,
        weights_min_distance=weights_min_distance,
        log_G_per_weight=log_G_per_weight,
    )

    tasks = [(i, np.asarray(obs, dtype=np.float64).tolist()) for i, obs in enumerate(arrivals_table)]

    with mp.Pool(
        processes=n_workers,
        initializer=_mp_worker_init,
        initargs=(_MP,),
    ) as pool:
        return pool.map(_mp_event_task, tasks)
