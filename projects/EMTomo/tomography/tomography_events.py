from __future__ import annotations

import multiprocessing as mp
from typing import Callable, Dict

import numpy as np

from instruments.instruments import coarsen_G, compute_epicenter_weight_matrix
from raytracing import compute_G_all_stations, compute_G_all_stations_serial
from .tomography_math import (
    _calculate_residuals,
    _normal_equation_contribution,
    _select_top_n_weights,
)

_MP: dict = {}


def _mp_worker_init() -> None:
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
    temperature: float,
    weights_top_n: int,
    compute_G: Callable[..., tuple[np.ndarray, np.ndarray]],
    log_G_per_weight: bool,
) -> tuple:
    step = 0.1

    weights, misfit = compute_epicenter_weight_matrix(
        station_fields=sf, observed=observed, temperature=temperature, return_misfit=True
    )
    weights = _select_top_n_weights(weights, weights_top_n, normalize=True)
    weights_indices = np.argwhere(weights > 0)
    weights_values = weights[weights > 0]

    if weights_indices.size == 0:
        best_flat = int(np.argmax(weights))
        best_idx = np.unravel_index(best_flat, weights.shape)
        weights_indices = np.array([best_idx], dtype=np.int64)
        weights_values = np.array([weights[best_idx]], dtype=np.float64)

    coarse_shape = tuple(int(v) // subdivision for v in sf.shape[1:])
    n_vox = int(np.prod(coarse_shape))
    hessian = np.zeros((n_vox, n_vox), dtype=np.float64)
    rhs = np.zeros(n_vox, dtype=np.float64)
    first_residuals = None
    G_per_weight: Dict[int, list[np.ndarray]] = {}
    ray_count_per_weight: Dict[int, np.ndarray] = {}

    for w_idx, (weight_idx, weight_val) in enumerate(zip(weights_indices, weights_values)):
        epic = np.asarray(weight_idx, dtype=np.float64)
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
        G_stations = np.array([coarsen_G(G_fine[si], subdivision) for si in range(G_fine.shape[0])])
        residuals = _calculate_residuals(sf, observed, weight_idx)
        x, y, z = weight_idx
        station_residuals = observed - sf[:, x, y, z]
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
        misfit,
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
    T = _MP["temperature"]
    wtn = _MP["weights_top_n"]
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
        temperature=T,
        weights_top_n=wtn,
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
    temperature,
    weights_top_n,
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
        temperature=temperature,
        weights_top_n=weights_top_n,
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
    temperature,
    weights_top_n,
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
        temperature=temperature,
        weights_top_n=weights_top_n,
        log_G_per_weight=log_G_per_weight,
    )

    tasks = [(i, np.asarray(obs, dtype=np.float64).tolist()) for i, obs in enumerate(arrivals_table)]

    with mp.Pool(processes=n_workers, initializer=_mp_worker_init) as pool:
        return pool.map(_mp_event_task, tasks)
