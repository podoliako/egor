"""E-step per event (hypotheses and their weights) and its normal-equation contribution."""
from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from typing import Callable, Optional

import numpy as np

from instruments.instruments_coords import sample_cell_centered_trilinear_batch
from instruments.instruments_ops import restriction_tables
from instruments.likelihood import (
    PickNoise,
    candidate_weights,
    likelihood_normalization,
    negative_log_likelihood,
)
from raytracing import compute_G_all_stations, compute_G_all_stations_serial
from .tomography_math import (
    accumulate_normal_equations,
    refine_hypocentre_in_cell,
    select_candidate_cells,
)

RAY_STEP_CELLS = 0.1
RAY_MAX_STEPS = 50_000


@dataclass(frozen=True)
class EventSettings:
    subdivision: int
    slowness_interpolation: str
    n_candidates: int
    weights_top_n: int
    weights_min_distance: int
    candidate_mode: str
    temperature: float
    noise: PickNoise
    log_g_per_weight: bool = False
    log_misfit: bool = False


@dataclass(frozen=True)
class IterationFields:
    """Travel-time fields of the current model on the fine grid, shared by all events."""

    times: np.ndarray        # (n_stations, nx, ny, nz) seconds
    gx: np.ndarray           # travel-time gradients in index units, same shape
    gy: np.ndarray
    gz: np.ndarray
    stations: np.ndarray     # (n_stations, 3) continuous fine-grid index coordinates
    fine_cell_size: float
    coarse_shape: tuple[int, int, int]
    restriction: tuple       # instruments_ops.restriction_tables for the coarse grid
    normalization: object = 0.0  # likelihood_normalization of ``times``, shared by events

    @classmethod
    def from_times(cls, times, stations, fine_cell_size, subdivision, noise: PickNoise,
                   interpolation="nearest"):
        times = np.asarray(times, dtype=np.float32)
        if any(n % subdivision for n in times.shape[1:]):
            raise ValueError(f"Fine grid {times.shape[1:]} is not divisible by subdivision={subdivision}")
        gx, gy, gz = (np.empty_like(times) for _ in range(3))
        for station, field in enumerate(times):
            gx[station], gy[station], gz[station] = np.gradient(field, edge_order=1)
        coarse_shape = tuple(int(n) // subdivision for n in times.shape[1:])
        return cls(times, gx, gy, gz, np.asarray(stations, dtype=np.float64),
                   float(fine_cell_size), coarse_shape,
                   restriction_tables(coarse_shape, subdivision, interpolation),
                   likelihood_normalization(times, noise))

    @property
    def n_vox(self) -> int:
        return int(np.prod(self.coarse_shape))


@dataclass
class EventLog:
    candidate_cells: np.ndarray
    misfit_shape: np.ndarray
    positions: np.ndarray
    weights: np.ndarray
    misfit: Optional[np.ndarray]
    residuals: np.ndarray
    G_per_weight: Optional[dict]
    ray_count_per_weight: dict


def _sparsify_G_stations(G_fine: np.ndarray) -> dict[str, np.ndarray]:
    """Pack all station ray paths without transferring dense fine-grid zeros."""
    station, x, y, z = np.nonzero(G_fine)
    counts = np.bincount(station, minlength=G_fine.shape[0])
    offsets = np.empty(G_fine.shape[0] + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return {
        "shape": np.asarray(G_fine.shape[1:], dtype=np.int32),
        "offsets": offsets,
        "coords": np.column_stack((x, y, z)).astype(np.int32, copy=False),
        "values": G_fine[station, x, y, z].astype(np.float32, copy=False),
    }


def process_event(
    observed: np.ndarray,
    fields: IterationFields,
    settings: EventSettings,
    hessian: np.ndarray,
    rhs: np.ndarray,
    compute_G: Callable = compute_G_all_stations,
) -> EventLog:
    """Select and weight hypocentre hypotheses, then add their rays to ``hessian`` and ``rhs``."""
    if settings.candidate_mode not in ("soft", "hard"):
        raise ValueError("candidate_mode must be 'soft' or 'hard'")
    observed = np.asarray(observed, dtype=np.float64)
    noise = settings.noise

    cost = negative_log_likelihood(fields.times, observed, noise, fields.normalization)
    if settings.weights_top_n > settings.n_candidates:
        raise ValueError("weights_top_n must not exceed n_candidates")
    cells = select_candidate_cells(cost, settings.n_candidates, settings.weights_min_distance)
    refined = [refine_hypocentre_in_cell(fields.times, observed, cell, noise) for cell in cells]
    # Rank by the refined likelihood: the best grid cell need not hold the best position.
    best = np.argsort([value for _, value in refined], kind="stable")[:settings.weights_top_n]
    cells = cells[best]
    positions = np.asarray([refined[i][0] for i in best], dtype=np.float64)
    costs = np.asarray([refined[i][1] for i in best], dtype=np.float64)
    predicted = np.stack([sample_cell_centered_trilinear_batch(fields.times, p) for p in positions])

    weights = candidate_weights(costs, settings.temperature)
    if settings.candidate_mode == "hard":
        best = int(np.argmax(weights))
        weights = np.zeros_like(weights)
        weights[best] = 1.0

    x_lo = np.zeros(3, dtype=np.float64)
    x_hi = np.asarray(fields.times.shape[1:], dtype=np.float64) - 1.0
    h = fields.fine_cell_size
    first_residuals = None
    G_per_weight = {} if settings.log_g_per_weight else None
    ray_count_per_weight = {}
    for index, (position, weight) in enumerate(zip(positions, weights)):
        if weight <= 0.0:
            continue
        trace = (fields.gx, fields.gy, fields.gz, fields.stations, position,
                 h, RAY_STEP_CELLS, RAY_STEP_CELLS, RAY_MAX_STEPS, x_lo, x_hi)
        G_stations, reached = compute_G(*trace, *fields.restriction, fields.coarse_shape)
        residuals = observed - predicted[index]
        accumulate_normal_equations(
            hessian, rhs, G_stations, residuals, reached,
            noise.sigmas(predicted[index]), float(weight),
        )
        if first_residuals is None:
            first_residuals = residuals[:, None] - residuals[None, :]
        if G_per_weight is not None:
            fine_shape = fields.times.shape[1:]
            G_fine, _ = compute_G(*trace, *restriction_tables(fine_shape, 1, "identity"), fine_shape)
            G_per_weight[index] = _sparsify_G_stations(G_fine)
        ray_count_per_weight[index] = (G_stations > 0).sum(axis=0).astype(np.int16)

    misfit = None
    if settings.log_misfit:
        misfit = cost.copy()
        for cell, value in zip(cells, costs):
            misfit[tuple(cell)] = value
    return EventLog(
        candidate_cells=cells.astype(np.int32),
        misfit_shape=np.asarray(cost.shape, dtype=np.int32),
        positions=positions,
        weights=weights,
        misfit=misfit,
        residuals=first_residuals if first_residuals is not None else np.array([]),
        G_per_weight=G_per_weight,
        ray_count_per_weight=ray_count_per_weight,
    )


def partition_events(n_events: int, n_chunks: int) -> list[range]:
    """Split event indices into ordered, balanced, non-empty chunks."""
    if n_chunks < 1:
        raise ValueError("n_chunks must be >= 1")
    if n_events < 1:
        raise ValueError("At least one event is required")
    n_chunks = min(n_chunks, n_events)
    size, remainder = divmod(n_events, n_chunks)
    chunks, start = [], 0
    for chunk in range(n_chunks):
        stop = start + size + (chunk < remainder)
        chunks.append(range(start, stop))
        start = stop
    return chunks


def _process_events(event_indices, arrivals, fields, settings, compute_G):
    """Sum one normal system over ``event_indices``; logs keep explicit event indices."""
    hessian = np.zeros((fields.n_vox, fields.n_vox), dtype=np.float64)
    rhs = np.zeros(fields.n_vox, dtype=np.float64)
    logs = []
    for event in event_indices:
        try:
            log = process_event(arrivals[event], fields, settings, hessian, rhs, compute_G)
        except Exception as error:
            raise RuntimeError(f"Failed to process event {event}") from error
        logs.append((event, log))
    return hessian, rhs, logs


_WORKER: dict = {}


def _init_worker(arrivals, fields, settings) -> None:
    _WORKER.update(arrivals=arrivals, fields=fields, settings=settings)
    # Event-level processes already use every core; keep each one single-threaded.
    try:
        from numba import set_num_threads

        set_num_threads(1)
    except Exception:
        pass
    try:
        from threadpoolctl import threadpool_limits

        _WORKER["threadpool_limits"] = threadpool_limits(limits=1)
    except Exception:
        pass


def _process_chunk(event_indices):
    return _process_events(
        event_indices, _WORKER["arrivals"], _WORKER["fields"], _WORKER["settings"],
        compute_G_all_stations_serial,
    )


def run_events(arrivals, fields: IterationFields, settings: EventSettings, n_workers: int = 1):
    """Return the summed ``(hessian, rhs)`` and per-event logs ordered by event index."""
    arrivals = np.asarray(arrivals, dtype=np.float64)
    chunks = partition_events(len(arrivals), n_workers)
    if len(chunks) == 1:
        return _process_events(chunks[0], arrivals, fields, settings, compute_G_all_stations)

    hessian = rhs = None
    logs = []
    context = mp.get_context("fork")
    with context.Pool(len(chunks), initializer=_init_worker,
                      initargs=(arrivals, fields, settings)) as pool:
        for chunk_hessian, chunk_rhs, chunk_logs in pool.imap(_process_chunk, chunks):
            if hessian is None:
                hessian, rhs = chunk_hessian, chunk_rhs
            else:
                hessian += chunk_hessian
                rhs += chunk_rhs
            logs.extend(chunk_logs)
    return hessian, rhs, logs
