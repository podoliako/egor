from __future__ import annotations

import multiprocessing as mp
from typing import Callable, Dict, Optional

import numpy as np

from instruments.instruments import (
    coarsen_G_all,
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
_THREADPOOL_LIMITER = None


def _sparsify_G_stations(G_fine: np.ndarray) -> dict[str, np.ndarray]:
    """Pack all station ray paths without transferring dense fine-grid zeros."""
    station, x, y, z = np.nonzero(G_fine)
    counts = np.bincount(station, minlength=G_fine.shape[0])
    offsets = np.empty(G_fine.shape[0] + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    coords = np.column_stack((x, y, z)).astype(np.int32, copy=False)
    values = G_fine[station, x, y, z].astype(np.float32, copy=False)
    return {
        "shape": np.asarray(G_fine.shape[1:], dtype=np.int32),
        "offsets": offsets,
        "coords": coords,
        "values": values,
    }


def _mp_worker_init(state: Optional[dict] = None) -> None:
    global _MP, _THREADPOOL_LIMITER
    if state is not None:
        _MP = state
    try:
        from numba import set_num_threads # pyright: ignore[reportMissingImports]

        set_num_threads(1)
    except Exception:
        pass
    try:
        from threadpoolctl import threadpool_limits

        # Event-level multiprocessing already provides CPU parallelism. Letting
        # every worker start a full OpenBLAS pool heavily oversubscribes the host.
        _THREADPOOL_LIMITER = threadpool_limits(limits=1)
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
    log_misfit: bool,
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
    logged_misfit = misfit.copy() if log_misfit else None
    if logged_misfit is not None:
        for cell_index, refined_misfit in zip(weights_indices, refined_misfits):
            logged_misfit[tuple(cell_index)] = refined_misfit
    compact_weights = {
        "shape": np.asarray(misfit.shape, dtype=np.int32),
        "indices": np.asarray(weights_indices, dtype=np.int32),
    }

    coarse_shape = tuple(int(v) // subdivision for v in sf.shape[1:])
    n_vox = int(np.prod(coarse_shape))
    hessian = np.zeros((n_vox, n_vox), dtype=np.float64)
    rhs = np.zeros(n_vox, dtype=np.float64)
    first_residuals = None
    G_per_weight: Dict[int, dict[str, np.ndarray]] = {}
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
        G_stations = coarsen_G_all(
            G_fine,
            subdivision,
            slowness_interpolation=slowness_interpolation,
        )
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
            G_per_weight[w_idx] = _sparsify_G_stations(G_fine)
        ray_count_per_weight[w_idx] = (G_stations > 0).sum(axis=0).astype(np.int16)

    log_data = (
        compact_weights,
        np.asarray(refined_positions, dtype=np.float64),
        np.asarray(weights_values, dtype=np.float64),
        logged_misfit,
        first_residuals if first_residuals is not None else np.array([]),
        G_per_weight if log_G_per_weight else None,
        ray_count_per_weight,
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
    log_misfit = _MP.get("log_misfit", False)

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
        log_misfit=log_misfit,
    )


def _aggregate_event_results(results):
    """Sum normal equations while retaining explicitly indexed event logs."""
    hessian_sum = None
    rhs_sum = None
    event_logs = []
    for event_idx, hessian_event, rhs_event, log_data in results:
        if hessian_sum is None:
            hessian_sum = hessian_event
            rhs_sum = rhs_event
        else:
            np.add(hessian_sum, hessian_event, out=hessian_sum)
            np.add(rhs_sum, rhs_event, out=rhs_sum)
        event_logs.append((event_idx, log_data))

    if hessian_sum is None or rhs_sum is None:
        raise ValueError("At least one event result is required")
    return hessian_sum, rhs_sum, event_logs


def _partition_event_tasks(arrivals_table, n_chunks: int):
    """Split indexed events into ordered, balanced, non-empty chunks."""
    if n_chunks < 1:
        raise ValueError("n_chunks must be >= 1")
    tasks = [
        (event_idx, np.asarray(observed, dtype=np.float64).tolist())
        for event_idx, observed in enumerate(arrivals_table)
    ]
    if not tasks:
        raise ValueError("arrivals_table must contain at least one event")

    n_chunks = min(n_chunks, len(tasks))
    chunk_size, remainder = divmod(len(tasks), n_chunks)
    chunks = []
    start = 0
    for chunk_idx in range(n_chunks):
        stop = start + chunk_size + (chunk_idx < remainder)
        chunks.append((chunk_idx, tasks[start:stop]))
        start = stop
    return chunks


def _mp_event_chunk_task(packed: tuple) -> tuple:
    """Process one event chunk and return one summed dense normal system."""
    chunk_idx, event_tasks = packed

    def event_results():
        for task in event_tasks:
            event_idx = task[0]
            try:
                hessian, rhs, log_data = _mp_event_task(task)
            except Exception as error:
                raise RuntimeError(f"Failed to process event {event_idx}") from error
            yield event_idx, hessian, rhs, log_data

    hessian, rhs, event_logs = _aggregate_event_results(event_results())
    return chunk_idx, hessian, rhs, event_logs


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
    log_misfit: bool = False,
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
        log_misfit=log_misfit,
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
    log_misfit: bool = False,
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
        log_misfit=log_misfit,
    )

    chunks = _partition_event_tasks(arrivals_table, n_workers)
    active_workers = min(n_workers, len(chunks))
    hessian = None
    rhs = None
    event_logs = []
    pool = mp.Pool(
        processes=active_workers,
        initializer=_mp_worker_init,
        initargs=(_MP,),
    )
    try:
        for expected_chunk_idx, result in enumerate(
            pool.imap(_mp_event_chunk_task, chunks, chunksize=1)
        ):
            chunk_idx, hessian_chunk, rhs_chunk, chunk_logs = result
            if chunk_idx != expected_chunk_idx:
                raise RuntimeError("Parallel event chunks arrived out of order")
            if hessian is None:
                hessian = hessian_chunk
                rhs = rhs_chunk
            else:
                np.add(hessian, hessian_chunk, out=hessian)
                np.add(rhs, rhs_chunk, out=rhs)
            event_logs.extend(chunk_logs)
        pool.close()
        pool.join()
    except Exception:
        pool.terminate()
        pool.join()
        raise

    if hessian is None or rhs is None:
        raise RuntimeError("Parallel event processing returned no normal equations")
    expected_indices = list(range(len(arrivals_table)))
    if [event_idx for event_idx, _log_data in event_logs] != expected_indices:
        raise RuntimeError("Parallel event logs are incomplete or out of order")
    return hessian, rhs, event_logs
