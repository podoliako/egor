"""
tomography.py — EM tomography inversion.

Parallelism strategy
--------------------
n_workers = 1  (default)
    compute_G_all_stations uses Numba prange → ~40 threads for 40 stations.
    Saturates ~40 cores. Simple, no IPC overhead.

n_workers > 1  (Linux fork, recommended for 48-core servers)
    Events are split across worker processes via multiprocessing.Pool (fork).
    Each worker calls compute_G_all_stations_serial (single-threaded Numba).
    Numba JIT cache is shared via copy-on-write after fork.
    set n_workers = min(n_events, n_cpu_cores) for maximum throughput.
    Per-event logging (G_per_weight) is disabled in this mode.

Warm-up
-------
On first import, call warm_up_jit() to trigger compilation before
launching a Pool (avoids each worker recompiling independently).
"""
from __future__ import annotations

import io
import json
import multiprocessing as mp
import pstats
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from instruments import (
    MetricPoint,
    coarsen_G,
    compute_epicenter_weight_matrix,
    compute_station_travel_time_fields,
    metric_to_index,
)
from raytracing import (
    compute_G_all_stations,
    compute_G_all_stations_serial,
    rasterize_path_lengths,
    trace_ray_from_timefield,
)


# ── JIT warm-up ───────────────────────────────────────────────────────────────

def warm_up_jit() -> None:
    """
    Force Numba to compile all kernels before forking workers.
    Takes ~3 s on first run; reads from cache on subsequent runs (< 0.1 s).
    Call once at the top of your main script, before run_em.
    """
    from raytracing import _trace_ray_nb, _rasterize_nb  # noqa: F401
    dummy = np.zeros((1, 2, 2, 2), dtype=np.float64)
    sl    = np.zeros((1, 3),       dtype=np.float64)
    epic  = np.zeros(3,            dtype=np.float64)
    lo    = np.zeros(3,            dtype=np.float64)
    hi    = np.ones(3,             dtype=np.float64)
    compute_G_all_stations_serial(
        dummy, dummy, dummy, sl, epic, 1.0, 1.0, 1.0, 0.5, 0.5, 5, lo, hi
    )
    compute_G_all_stations(
        dummy, dummy, dummy, sl, epic, 1.0, 1.0, 1.0, 0.5, 0.5, 5, lo, hi
    )


# ── Multiprocessing worker state (populated in parent; inherited via fork) ────

_MP: dict = {}   # module-level; set by _prepare_mp_shared before Pool creation


def _mp_worker_init() -> None:
    """Pool initializer: restrict Numba to 1 thread so we don't over-subscribe."""
    try:
        from numba import set_num_threads
        set_num_threads(1)
    except Exception:
        pass


def _mp_event_task(packed: tuple) -> tuple:
    """
    Process a single event inside a worker process.
    Reads heavy arrays from _MP (inherited via fork, copy-on-write).
    """
    event_idx, observed = packed

    gx  = _MP['gx'];   gy = _MP['gy'];   gz = _MP['gz']
    sf  = _MP['sf'];   sl = _MP['sl']
    x_lo = _MP['x_lo'];  x_hi = _MP['x_hi']
    fcs  = _MP['fine_cell_size']
    sub  = _MP['subdivision']
    T    = _MP['temperature']
    wtn  = _MP['weights_top_n']

    observed = np.asarray(observed, dtype=np.float64)
    step = 0.5

    weights, _ = compute_epicenter_weight_matrix(
        station_fields=sf, observed=observed, temperature=T, return_misfit=True,
    )
    weights = _select_top_n_weights(weights, wtn, normalize=True)
    weights_indices = np.argwhere(weights > 0)
    weights_values  = weights[weights > 0]

    G_w: list = [];  r_w: list = []
    for weight_idx, weight_val in zip(weights_indices, weights_values):
        epic = np.asarray(weight_idx, dtype=np.float64)
        G_fine = compute_G_all_stations_serial(
            gx, gy, gz, sl, epic, fcs, fcs, fcs, step, step, 50000, x_lo, x_hi,
        )
        G_stations = np.array(
            [coarsen_G(G_fine[si], sub) for si in range(G_fine.shape[0])]
        )
        G_tilda   = G_stations[:, np.newaxis] - G_stations[np.newaxis, :]
        residuals = _calculate_residuals(sf, observed, weight_idx)
        G_w.append(G_tilda * weight_val)
        r_w.append(residuals * weight_val)

    return np.add.reduce(G_w), np.add.reduce(r_w)


# ── Logger ────────────────────────────────────────────────────────────────────

class TomographyLogger:
    """
    Directory layout:
        runs/run_<timestamp>/
          meta.json
          initial_model.npy / true_model.npy
          timing.jsonl / timing_summary.json
          profile.txt / profile_top30.json
          iter_<i>/
            model.npy / delta_s.npy / station_fields.npy
            event_<j>/
              weights.npy / misfit.npy / residuals.npy
              weight_<w>/G_station_<k>.npy
    """

    def __init__(self, base_dir: str = "runs"):
        self.run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir  = Path(base_dir) / f"run_{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._iter_start: float = 0.0
        self._run_start:  float = time.perf_counter()
        self.timing: dict = {}

    # ── meta ──────────────────────────────────────────────────────────────────

    def save_meta(self, run_params, station_locs, event_locs, grid_info=None):
        meta = {
            "run_id":      self.run_id,
            "run_params":  run_params,
            "station_locs": [list(s) for s in station_locs],
            "event_locs":   [list(e) for e in event_locs],
            "grid_info":    grid_info or {},
        }
        with open(self.run_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    # ── models ────────────────────────────────────────────────────────────────

    def save_initial_model(self, model):
        np.save(self.run_dir / "initial_model.npy",
                model.get_geo_grid(subdivision=1).vp)

    def save_true_model(self, model):
        if model is not None:
            np.save(self.run_dir / "true_model.npy",
                    model.get_geo_grid(subdivision=1).vp)

    # ── per-iteration ─────────────────────────────────────────────────────────

    def iter_dir(self, iteration: int) -> Path:
        d = self.run_dir / f"iter_{iteration}"
        d.mkdir(exist_ok=True)
        return d

    def save_iteration_model(self, iteration: int, model):
        np.save(self.iter_dir(iteration) / "model.npy",
                model.get_geo_grid(subdivision=1).vp)

    def save_delta_s(self, iteration: int, delta_s: np.ndarray):
        np.save(self.iter_dir(iteration) / "delta_s.npy", delta_s)

    def save_station_fields(self, iteration: int, station_fields: np.ndarray):
        np.save(self.iter_dir(iteration) / "station_fields.npy",
                np.asarray(station_fields))

    # ── per-event ─────────────────────────────────────────────────────────────

    def save_event_data(
        self,
        iteration: int,
        event_idx: int,
        weights: np.ndarray,
        misfit: Optional[np.ndarray] = None,
        residuals: Optional[np.ndarray] = None,
        G_per_weight: Optional[Dict[int, List[np.ndarray]]] = None,
    ):
        event_dir = self.iter_dir(iteration) / f"event_{event_idx}"
        event_dir.mkdir(exist_ok=True)
        np.save(event_dir / "weights.npy", weights)
        if misfit     is not None:  np.save(event_dir / "misfit.npy",    misfit)
        if residuals  is not None:  np.save(event_dir / "residuals.npy", residuals)
        if G_per_weight is not None:
            for w_idx, g_list in G_per_weight.items():
                w_dir = event_dir / f"weight_{w_idx}"
                w_dir.mkdir(exist_ok=True)
                for si, g in enumerate(g_list):
                    np.save(w_dir / f"G_station_{si}.npy", g)

    # ── timing ────────────────────────────────────────────────────────────────

    def start_iteration(self, iteration: int):
        self._iter_start = time.perf_counter()

    def end_iteration(self, iteration: int):
        elapsed = time.perf_counter() - self._iter_start
        self.timing[iteration] = elapsed
        with open(self.run_dir / "timing.jsonl", "a") as f:
            json.dump({"iter": iteration, "elapsed_s": elapsed}, f)
            f.write("\n")

    def save_profiling(self, profiler):
        buf   = io.StringIO()
        stats = pstats.Stats(profiler, stream=buf).strip_dirs().sort_stats("cumulative")
        stats.print_stats(50)
        (self.run_dir / "profile.txt").write_text(buf.getvalue())
        rows = []
        for func, (cc, nc, tt, ct, _) in list(stats.stats.items())[:30]:
            rows.append({
                "func":       f"{func[0]}:{func[1]}:{func[2]}",
                "n_calls":    nc,
                "tottime_s":  round(tt, 6),
                "cumtime_s":  round(ct, 6),
            })
        with open(self.run_dir / "profile_top30.json", "w") as f:
            json.dump(rows, f, indent=2)

    def save_timing_summary(self):
        total = time.perf_counter() - self._run_start
        summary = {
            "total_s":     round(total, 3),
            "per_iter":    {str(k): round(v, 3) for k, v in self.timing.items()},
            "mean_iter_s": round(
                sum(self.timing.values()) / len(self.timing), 3
            ) if self.timing else None,
        }
        with open(self.run_dir / "timing_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        return summary


# ── EM loop ───────────────────────────────────────────────────────────────────

def run_em(
    n_cycles,
    initial_model,
    arrivals_table,
    station_locs,
    wave_type: str = 'P',
    solver: Union[str, object] = 'skfmm',
    lambda_reg: float = 1e-3,
    temperature: float = 1.0,
    weights_top_n: int = 1,
    subdivision: int = 1,
    n_workers: int = 1,
    true_model=None,
    event_locs=None,
    logger=None,
    save_runs: bool = True,
    runs_dir: str = "runs",
):
    if save_runs and logger is None:
        logger = TomographyLogger(base_dir=runs_dir)

    if logger is not None:
        coarse_grid = initial_model.get_geo_grid(subdivision=1)
        coarse_side = float(coarse_grid.cell_size)
        fine_side   = coarse_side
        grid_info   = {
            "coarse_cell_size": coarse_side,
            "coarse_shape":     [int(v) for v in coarse_grid.shape],
            "coarse_side_m":    [coarse_side * int(v) for v in coarse_grid.shape],
        }
        if subdivision > 1:
            fine_grid = initial_model.get_geo_grid(subdivision=subdivision)
            fine_side = float(fine_grid.cell_size)
            grid_info.update({
                "fine_cell_size": fine_side,
                "fine_shape":     [int(v) for v in fine_grid.shape],
                "fine_side_m":    [fine_side * int(v) for v in fine_grid.shape],
            })
        else:
            grid_info.update({
                "fine_cell_size": coarse_side,
                "fine_shape":     [int(v) for v in coarse_grid.shape],
                "fine_side_m":    [coarse_side * int(v) for v in coarse_grid.shape],
            })

        run_params = dict(
            n_cycles=n_cycles,
            wave_type=wave_type,
            solver=str(solver),
            lambda_reg=lambda_reg,
            temperature=temperature,
            weights_top_n=weights_top_n,
            subdivision=subdivision,
            n_workers=n_workers,
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
        print(f'{i + 1}/{n_cycles}')
        if logger is not None:
            logger.start_iteration(i)
            logger.save_iteration_model(i, model)

        delta_s = make_tomography_step(
            model, arrivals_table, station_locs,
            wave_type, solver, lambda_reg, temperature, weights_top_n,
            subdivision, n_workers=n_workers, iteration=i, logger=logger,
        )

        new_velocities = 1 / (1 / model.get_geo_grid(subdivision=1).vp + delta_s)
        model.set_vp_array(new_velocities)

        if logger is not None:
            logger.end_iteration(i)
            logger.save_delta_s(i, delta_s)

    if logger is not None:
        logger.save_timing_summary()
        print(f"[TomographyLogger] Run saved to: {logger.run_dir}")

    return logger


# ── One tomography step ───────────────────────────────────────────────────────

def make_tomography_step(
    initial_model,
    arrivals_table,
    station_locs,
    wave_type: str = 'P',
    solver: Union[str, object] = 'skfmm',
    lambda_reg: float = 1e-3,
    temperature: float = 1.0,
    weights_top_n: int = 1,
    subdivision: int = 1,
    n_workers: int = 1,
    iteration: int = 0,
    logger: Optional[TomographyLogger] = None,
):
    coarse_grid = initial_model.get_geo_grid(subdivision=1)
    coarse_shape = tuple(int(v) for v in coarse_grid.shape)

    fine_grid     = initial_model.get_geo_grid(subdivision=subdivision)
    fine_shape    = tuple(int(v) for v in fine_grid.shape)
    fine_cell_size = float(fine_grid.cell_size)

    station_idx_fine = [metric_to_index(s, fine_cell_size, fine_shape) for s in station_locs]

    # ── station fields + stacked gradients (computed once per iteration) ──────
    station_fields = compute_station_travel_time_fields(
        fine_grid, station_idx_fine, wave_type, solver
    )
    sf_array = np.asarray(station_fields, dtype=np.float64)   # (n_st, nx, ny, nz)

    grads = np.stack(
        [np.gradient(sf, 1.0, 1.0, 1.0, edge_order=1) for sf in sf_array]
    )  # (n_st, 3, nx, ny, nz)
    gx = np.ascontiguousarray(grads[:, 0])
    gy = np.ascontiguousarray(grads[:, 1])
    gz = np.ascontiguousarray(grads[:, 2])

    # Station positions as float64 for Numba
    sl   = np.asarray(station_idx_fine, dtype=np.float64)   # (n_st, 3)
    x_lo = np.zeros(3, dtype=np.float64)
    x_hi = np.asarray(fine_shape, dtype=np.float64) - 1.0

    if logger is not None:
        logger.save_station_fields(iteration, sf_array)

    # ── dispatch ──────────────────────────────────────────────────────────────
    if n_workers > 1:
        results = _run_events_parallel(
            arrivals_table, gx, gy, gz, sf_array, sl, x_lo, x_hi,
            fine_cell_size, subdivision, temperature, weights_top_n, n_workers,
        )
    else:
        results = [
            _process_event_single(
                event_idx=i,
                observed=np.asarray(obs, dtype=np.float64),
                sf=sf_array, gx=gx, gy=gy, gz=gz, sl=sl,
                x_lo=x_lo, x_hi=x_hi,
                fine_cell_size=fine_cell_size,
                subdivision=subdivision,
                temperature=temperature,
                weights_top_n=weights_top_n,
                iteration=iteration,
                logger=logger,
            )
            for i, obs in enumerate(arrivals_table)
        ]

    G_acc = [r[0] for r in results]
    r_acc = [r[1] for r in results]

    return _solve_delta_s(
        g_tilde_prime=np.add.reduce(G_acc),
        r_prime=np.add.reduce(r_acc),
        model_shape=coarse_shape,
        lambda_reg=lambda_reg,
        use_upper_triangle_pairs=True,
    )


def _process_event_single(
    event_idx, observed, sf, gx, gy, gz, sl, x_lo, x_hi,
    fine_cell_size, subdivision, temperature, weights_top_n,
    iteration, logger,
):
    """Single-process path: uses parallel Numba (prange over stations)."""
    step = 0.5

    weights, misfit = compute_epicenter_weight_matrix(
        station_fields=sf, observed=observed, temperature=temperature, return_misfit=True,
    )
    weights = _select_top_n_weights(weights, weights_top_n, normalize=True)
    weights_indices = np.argwhere(weights > 0)
    weights_values  = weights[weights > 0]

    G_per_weight: Optional[dict] = {} if logger else None
    first_residuals = None
    G_w: list = [];  r_w: list = []

    for w_idx, (weight_idx, weight_val) in enumerate(
        zip(weights_indices, weights_values)
    ):
        epic = np.asarray(weight_idx, dtype=np.float64)

        # All stations traced in parallel via Numba prange
        G_fine = compute_G_all_stations(
            gx, gy, gz, sl, epic,
            fine_cell_size, fine_cell_size, fine_cell_size,
            step, step, 50000, x_lo, x_hi,
        )  # (n_st, nx_fine, ny_fine, nz_fine)

        G_stations = np.array(
            [coarsen_G(G_fine[si], subdivision) for si in range(G_fine.shape[0])]
        )
        G_tilda   = G_stations[:, np.newaxis] - G_stations[np.newaxis, :]
        residuals = _calculate_residuals(sf, observed, weight_idx)

        if first_residuals is None:
            first_residuals = residuals
        if logger is not None and G_per_weight is not None:
            G_per_weight[w_idx] = [G_fine[si] for si in range(G_fine.shape[0])]

        G_w.append(G_tilda * weight_val)
        r_w.append(residuals * weight_val)

    if logger is not None:
        logger.save_event_data(
            iteration=iteration,
            event_idx=event_idx,
            weights=weights,
            misfit=misfit,
            residuals=first_residuals if first_residuals is not None else np.array([]),
            G_per_weight=G_per_weight,
        )

    return np.add.reduce(G_w), np.add.reduce(r_w)


def _run_events_parallel(
    arrivals_table, gx, gy, gz, sf, sl, x_lo, x_hi,
    fine_cell_size, subdivision, temperature, weights_top_n, n_workers,
):
    """
    Fork-based event parallelism (Linux).  Heavy arrays are set as module
    globals before Pool creation so fork inherits them copy-on-write — no
    serialization overhead.  Per-event logging is skipped in this mode.
    """
    global _MP
    _MP = dict(
        gx=gx, gy=gy, gz=gz,
        sf=sf, sl=sl,
        x_lo=x_lo, x_hi=x_hi,
        fine_cell_size=fine_cell_size,
        subdivision=subdivision,
        temperature=temperature,
        weights_top_n=weights_top_n,
    )

    tasks = [
        (i, np.asarray(obs, dtype=np.float64).tolist())
        for i, obs in enumerate(arrivals_table)
    ]

    with mp.Pool(processes=n_workers, initializer=_mp_worker_init) as pool:
        return pool.map(_mp_event_task, tasks)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _calculate_residuals(station_fields: np.ndarray, arrivals: np.ndarray, weight_idx):
    x, y, z = weight_idx
    predicted       = station_fields[:, x, y, z]
    residual_vector = arrivals - predicted
    return residual_vector[:, np.newaxis] - residual_vector[np.newaxis, :]


def _solve_delta_s(
    g_tilde_prime: np.ndarray,
    r_prime: np.ndarray,
    model_shape: Tuple[int, int, int],
    lambda_reg: float,
    use_upper_triangle_pairs: bool,
):
    g_tilde_prime = np.asarray(g_tilde_prime)
    n_st = g_tilde_prime.shape[0]
    if n_st != g_tilde_prime.shape[1]:
        raise ValueError("g_tilde_prime must have shape (n_stations, n_stations, ...)")
    if r_prime.shape != (n_st, n_st):
        raise ValueError("r_prime shape must be (n_stations, n_stations)")

    pair_mask = (
        np.triu(np.ones((n_st, n_st), dtype=bool), k=1)
        if use_upper_triangle_pairs
        else ~np.eye(n_st, dtype=bool)
    )

    g_rows = g_tilde_prime[pair_mask].reshape(-1, int(np.prod(model_shape)))
    r_vec  = r_prime[pair_mask].reshape(-1)
    if g_rows.shape[0] == 0:
        raise ValueError("No station pairs available to solve tomography system")

    gg_t     = g_rows @ g_rows.T
    gg_t_reg = gg_t + float(lambda_reg) * np.eye(gg_t.shape[0], dtype=np.float64)

    try:
        alpha = np.linalg.solve(gg_t_reg, r_vec)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(gg_t_reg, r_vec, rcond=None)[0]

    return (g_rows.T @ alpha).reshape(model_shape)


def _select_top_n_weights(weights_matrix, n: int, normalize: bool = False):
    w = np.asarray(weights_matrix, dtype=np.float64)
    if w.ndim != 3:
        raise ValueError("weights_matrix must be a 3-D array")
    if not isinstance(n, (int, np.integer)):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("n must be >= 0")

    out = np.zeros_like(w)
    if n == 0:
        return out
    if n >= w.size:
        out = w.copy()
    else:
        flat    = w.ravel()
        top_idx = np.argpartition(flat, -n)[-n:]
        out.ravel()[top_idx] = flat[top_idx]

    if normalize:
        s = out.sum()
        if s > 0:
            out /= s
    return out


# Keep for backward compatibility / direct use
def _calculate_G(station_field, origin_loc, station_loc, geo_shape, voxel_size, gradT=None):
    path = trace_ray_from_timefield(
        T=station_field,
        station_xyz=station_loc,
        epic_xyz=origin_loc,
        spacing_xyz=(1.0, 1.0, 1.0),
        gradT=gradT,
    )
    return rasterize_path_lengths(
        path_xyz=path, shape=geo_shape, voxel_size=voxel_size, dtype=np.float64,
    )