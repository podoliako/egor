"""E-step per event (hypotheses and their weights) and its normal-equation contribution."""
from __future__ import annotations

from dataclasses import dataclass, field
import multiprocessing as mp
from time import perf_counter
from typing import Callable, Optional

import numpy as np
from scipy import sparse

from eikonal import SOURCE_RADIUS_CELLS
from instruments.instruments_coords import sample_cell_centered_trilinear_batch
from instruments.instruments_ops import restriction_tables
from instruments.likelihood import (
    PickNoise,
    candidate_weights,
    likelihood_normalization,
    negative_log_likelihood,
    negative_log_likelihood_batch,
)
from raytracing import (
    compute_G_all_stations_serial, compute_G_all_stations_compact,
)
from .tomography_math import (
    MatrixFreeNormalEquations,
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
    likelihood_batch_size: int = 8
    likelihood_batch_memory_mb: int = 128
    compact_rays: bool = True
    fast_refinement: bool = True
    normal_solver: str = "dense"
    log_diagnostics: bool = True


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
    ray_reached_per_weight: dict = field(default_factory=dict)


def _sparsify_G_stations(G_fine, fine_shape=None) -> dict[str, np.ndarray]:
    """Pack dense or CSR rays in the viewer's existing sparse artifact format."""
    if sparse.issparse(G_fine):
        if fine_shape is None or int(np.prod(fine_shape)) != G_fine.shape[1]:
            raise ValueError("fine_shape must match sparse ray columns")
        rows = G_fine.tocsr(copy=True)
        rows.sum_duplicates(); rows.eliminate_zeros(); rows.sort_indices()
        return {
            "shape": np.asarray(fine_shape, dtype=np.int32),
            "offsets": rows.indptr.astype(np.int64),
            "coords": np.column_stack(np.unravel_index(rows.indices, fine_shape)).astype(np.int32),
            "values": rows.data.astype(np.float32),
        }
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
    compute_G: Optional[Callable] = None,
    *,
    cost: Optional[np.ndarray] = None,
    timings: Optional[dict] = None,
) -> EventLog:
    """Select and weight hypocentre hypotheses, then add their rays to ``hessian`` and ``rhs``."""
    if settings.candidate_mode not in ("soft", "hard"):
        raise ValueError("candidate_mode must be 'soft' or 'hard'")
    observed = np.asarray(observed, dtype=np.float64)
    noise = settings.noise

    if compute_G is None:
        compute_G = compute_G_all_stations_compact if settings.compact_rays else compute_G_all_stations_serial
    if cost is None:
        started = perf_counter()
        cost = negative_log_likelihood(fields.times, observed, noise, fields.normalization)
        _record_time(timings, "likelihood_s", started)
    started = perf_counter()
    if settings.weights_top_n > settings.n_candidates:
        raise ValueError("weights_top_n must not exceed n_candidates")
    cells = select_candidate_cells(cost, settings.n_candidates, settings.weights_min_distance)
    if settings.fast_refinement:
        refined = [refine_hypocentre_in_cell(fields.times, observed, cell, noise) for cell in cells]
    else:
        refined = [refine_hypocentre_in_cell(fields.times, observed, cell, noise, accelerated=False)
                   for cell in cells]
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

    _record_time(timings, "refinement_s", started)
    x_lo = np.zeros(3, dtype=np.float64)
    x_hi = np.asarray(fields.times.shape[1:], dtype=np.float64) - 1.0
    h = fields.fine_cell_size
    first_residuals = None
    G_per_weight = {} if settings.log_g_per_weight else None
    ray_count_per_weight = {}
    ray_reached_per_weight = {}
    n_stations = len(fields.stations)
    for index, (position, weight) in enumerate(zip(positions, weights)):
        if not weight > 0.0:
            continue
        trace = (fields.gx, fields.gy, fields.gz, fields.stations, position,
                 h, RAY_STEP_CELLS, RAY_STEP_CELLS, RAY_MAX_STEPS, x_lo, x_hi, SOURCE_RADIUS_CELLS)
        started = perf_counter()
        G_stations, reached = compute_G(*trace, *fields.restriction, fields.coarse_shape)
        _record_time(timings, "rays_s", started)
        started = perf_counter()
        reached = np.array(reached, dtype=bool, copy=True)
        if reached.shape != (n_stations,):
            raise ValueError(
                f"Ray reached mask for hypothesis {index} must have shape "
                f"({n_stations},), got {reached.shape}"
            )
        # Keep the normal-equation trace, not the optional fine-grid re-trace.
        ray_reached_per_weight[index] = reached
        residuals = observed - predicted[index]
        accumulate_normal_equations(
            hessian, rhs, G_stations, residuals, reached,
            noise.sigmas(predicted[index]), float(weight),
        )
        _record_time(timings, "assembly_s", started)
        started = perf_counter()
        if settings.log_diagnostics and first_residuals is None:
            first_residuals = residuals[:, None] - residuals[None, :]
        if G_per_weight is not None:
            fine_shape = fields.times.shape[1:]
            G_fine, _ = compute_G(*trace, *restriction_tables(fine_shape, 1, "identity"), fine_shape)
            G_per_weight[index] = _sparsify_G_stations(G_fine, fine_shape)
        if settings.log_diagnostics:
            counts = np.asarray((G_stations > 0).sum(axis=0)).reshape(fields.coarse_shape)
            ray_count_per_weight[index] = counts.astype(np.int16)
        _record_time(timings, "diagnostics_s", started)

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
        ray_reached_per_weight=ray_reached_per_weight,
    )


def _record_time(timings, key, started):
    if timings is not None:
        timings[key] = timings.get(key, 0.0) + perf_counter() - started


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


def _effective_batch_size(fields, settings):
    bytes_per_event = int(np.prod(fields.times.shape[1:])) * np.dtype(np.float64).itemsize
    budget = settings.likelihood_batch_memory_mb * 1024**2
    return max(1, min(settings.likelihood_batch_size, budget // max(bytes_per_event, 1)))


def _process_events(event_indices, arrivals, fields, settings, compute_G=None,
                    *, hessian=None, rhs=None, timings=None):
    """Sum in event order, batching only the observation likelihood evaluation."""
    if hessian is None:
        hessian = (MatrixFreeNormalEquations(fields.n_vox) if settings.normal_solver == "matrix_free"
                   else np.zeros((fields.n_vox, fields.n_vox), dtype=np.float64))
    if rhs is None:
        rhs = np.zeros(fields.n_vox, dtype=np.float64)
    logs = []
    batch_size = _effective_batch_size(fields, settings)
    for start in range(0, len(event_indices), batch_size):
        batch = event_indices[start:start + batch_size]
        costs = None
        if batch_size > 1:
            started = perf_counter()
            costs = negative_log_likelihood_batch(
                fields.times, arrivals[list(batch)], settings.noise, fields.normalization,
            )
            _record_time(timings, "likelihood_s", started)
        for offset, event in enumerate(batch):
            try:
                log = process_event(
                    arrivals[event], fields, settings, hessian, rhs, compute_G,
                    cost=None if costs is None else costs[offset], timings=timings,
                )
            except Exception as error:
                raise RuntimeError(f"Failed to process event {event}") from error
            logs.append((event, log))
        del costs
    return hessian, rhs, logs


_WORKER: dict = {}


def _init_worker(arrivals, fields, settings, buffer=None, n_chunks=0) -> None:
    _WORKER.clear()
    _WORKER.update(arrivals=arrivals, fields=fields, settings=settings)
    if buffer is not None:
        stride = fields.n_vox * (fields.n_vox + 1)
        _WORKER["normal_buffers"] = np.frombuffer(buffer, dtype=np.float64).reshape(n_chunks, stride)
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


def _process_chunk(task):
    index, event_indices = task
    fields = _WORKER["fields"]
    hessian = rhs = None
    shared = _WORKER.get("normal_buffers")
    if shared is not None:
        n = fields.n_vox
        hessian = shared[index, :n*n].reshape(n, n)
        rhs = shared[index, n*n:]
    timings = {}
    started = perf_counter()
    hessian, rhs, logs = _process_events(
        event_indices, _WORKER["arrivals"], fields, _WORKER["settings"],
        hessian=hessian, rhs=rhs, timings=timings,
    )
    timings["chunk_elapsed_s"] = perf_counter() - started
    # Shared dense slots have one writer each; only compact results cross the pipe.
    return index, None if shared is not None else hessian, None if shared is not None else rhs, logs, timings


def run_events(arrivals, fields: IterationFields, settings: EventSettings, n_workers: int = 1,
               *, diagnostics: Optional[dict] = None):
    """Return ordered normal equations/logs; diagnostics sum worker times, not wall time."""
    arrivals = np.asarray(arrivals, dtype=np.float64)
    if arrivals.ndim != 2 or arrivals.shape[1] != len(fields.stations) or not np.all(np.isfinite(arrivals)):
        raise ValueError("Arrivals must be a finite (events, stations) array")
    if settings.normal_solver not in ("dense", "matrix_free"):
        raise ValueError("normal_solver must be dense or matrix_free")
    if settings.likelihood_batch_size < 1 or settings.likelihood_batch_memory_mb < 1:
        raise ValueError("Likelihood batch size and memory budget must be positive")
    chunks = partition_events(len(arrivals), n_workers)
    totals = {}
    chunk_times = []
    if len(chunks) == 1:
        started = perf_counter()
        result = _process_events(chunks[0], arrivals, fields, settings, timings=totals)
        chunk_times.append(perf_counter() - started)
    else:
        context = mp.get_context("fork")
        buffer = shared = None
        if settings.normal_solver == "dense":
            n = fields.n_vox
            stride = n * (n + 1)
            buffer = context.RawArray("d", len(chunks) * stride)
            shared = np.frombuffer(buffer, dtype=np.float64).reshape(len(chunks), stride)
        hessian = rhs = None
        logs = []
        with context.Pool(len(chunks), initializer=_init_worker,
                          initargs=(arrivals, fields, settings, buffer, len(chunks))) as pool:
            for index, chunk_hessian, chunk_rhs, chunk_logs, times in pool.imap(_process_chunk, enumerate(chunks)):
                if shared is not None:
                    chunk_hessian = shared[index, :n*n].reshape(n, n)
                    chunk_rhs = shared[index, n*n:]
                if hessian is None:
                    # Own only the final matrix after releasing all shared partial systems.
                    hessian = chunk_hessian.copy() if shared is not None else chunk_hessian
                    rhs = chunk_rhs.copy()
                else:
                    if isinstance(hessian, MatrixFreeNormalEquations):
                        hessian.merge(chunk_hessian)
                    else:
                        hessian += chunk_hessian
                    rhs += chunk_rhs
                logs.extend(chunk_logs)
                chunk_times.append(times.pop("chunk_elapsed_s"))
                for key, value in times.items():
                    totals[key] = totals.get(key, 0.0) + value
        result = hessian, rhs, logs
    if diagnostics is not None:
        diagnostics.update(worker_time_s=totals, chunk_elapsed_s=chunk_times,
                           workers=len(chunks), likelihood_batch_size=_effective_batch_size(fields, settings),
                           normal_solver=settings.normal_solver)
    return result
