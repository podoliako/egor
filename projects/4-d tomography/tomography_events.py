from __future__ import annotations

import multiprocessing as mp
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from instruments import coarsen_G, compute_epicenter_weight_matrix
from raytracing import compute_G_all_stations, compute_G_all_stations_serial
from tomography_math import _calculate_residuals, _select_top_n_weights

_MP: dict = {}


def _mp_worker_init() -> None:
    try:
        from numba import set_num_threads

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
    compute_G: Callable[..., np.ndarray],
    log_G_per_weight: bool,
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Dict[int, list]]]]:
    step = 0.5

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

    G_w: list = []
    r_w: list = []
    first_residuals = None
    G_per_weight: Dict[int, list] = {}

    for w_idx, (weight_idx, weight_val) in enumerate(zip(weights_indices, weights_values)):
        epic = np.asarray(weight_idx, dtype=np.float64)
        G_fine = compute_G(
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
        G_tilda = G_stations[:, np.newaxis] - G_stations[np.newaxis, :]
        residuals = _calculate_residuals(sf, observed, weight_idx)

        if first_residuals is None:
            first_residuals = residuals
        if log_G_per_weight:
            G_per_weight[w_idx] = [G_fine[si] for si in range(G_fine.shape[0])]

        G_w.append(G_tilda * weight_val)
        r_w.append(residuals * weight_val)

    if not G_w:
        zero_g = np.zeros((sf.shape[0], sf.shape[0]) + sf.shape[1:], dtype=np.float64)
        zero_r = np.zeros((sf.shape[0], sf.shape[0]), dtype=np.float64)
        log_data = (weights, misfit, np.array([]), G_per_weight if log_G_per_weight else None)
        return zero_g, zero_r, log_data

    log_data = (
        weights,
        misfit,
        first_residuals if first_residuals is not None else np.array([]),
        G_per_weight if log_G_per_weight else None,
    )
    return np.add.reduce(G_w), np.add.reduce(r_w), log_data


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
