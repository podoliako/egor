"""
Prototype tomography inversion from arrival tables.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import time

from instruments import (
    compute_epicenter_weight_matrix,
    compute_station_travel_time_fields,
    coarsen_G,
    metric_to_index,
    MetricPoint,
)
from raytracing import trace_ray_from_timefield, rasterize_path_lengths


GridPoint = Tuple[int, int, int]
StationArrival = Dict[str, Union[GridPoint, float]]
EventArrivals = Dict[str, Union[GridPoint, List[StationArrival]]]
TomographyEventResult = Dict[str, Union[int, GridPoint, np.ndarray]]


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class TomographyLogger:
    """
    Структура директорий:
        runs/
          run_<timestamp>/
            meta.json
            initial_model.npy
            true_model.npy
            iter_<i>/
              model.npy
              delta_s.npy
              station_fields.npy        # (n_stations, nx, ny, nz)  fine grid
              event_<j>/
                weights.npy             # (nx, ny, nz)  coarse grid
                misfit.npy              # (nx, ny, nz)  coarse grid
                residuals.npy
                weight_<w>/
                  G_station_<k>.npy     # (nx, ny, nz)  fine grid
    """

    def __init__(self, base_dir: str = "runs"):
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(base_dir) / f"run_{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    # ── meta ──────────────────────────────────────────────────────────────

    def save_meta(
        self,
        run_params: dict,
        station_locs: Sequence,
        event_locs: Sequence,
        grid_info=None,
    ):
        meta = {
            "run_id": self.run_id,
            "run_params": run_params,
            "station_locs": [list(s) for s in station_locs],
            "event_locs": [list(e) for e in event_locs],
            "grid_info": grid_info or {},
        }
        with open(self.run_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    # ── models ────────────────────────────────────────────────────────────

    def save_initial_model(self, model):
        np.save(self.run_dir / "initial_model.npy",
                model.get_geo_grid(subdivision=1).vp)

    def save_true_model(self, model):
        if model is not None:
            np.save(self.run_dir / "true_model.npy",
                    model.get_geo_grid(subdivision=1).vp)

    # ── per-iteration ──────────────────────────────────────────────────────

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
        # shape: (n_stations, nx, ny, nz)  — fine grid
        np.save(self.iter_dir(iteration) / "station_fields.npy",
                np.array(station_fields))

    # ── per-event ──────────────────────────────────────────────────────────

    def save_event_data(
        self,
        iteration: int,
        event_idx: int,
        weights: np.ndarray,
        misfit: Optional[np.ndarray] = None,
        residuals: Optional[np.ndarray] = None,
        G_per_weight: Optional[Dict[int, List[np.ndarray]]] = None,
    ):
        """
        G_per_weight: {weight_index: [g_fine_station_0, g_fine_station_1, ...]}
                      g_fine arrays are on the fine grid.
        """
        event_dir = self.iter_dir(iteration) / f"event_{event_idx}"
        event_dir.mkdir(exist_ok=True)

        np.save(event_dir / "weights.npy", weights)

        if misfit is not None:
            np.save(event_dir / "misfit.npy", misfit)

        if residuals is not None:
            np.save(event_dir / "residuals.npy", residuals)

        if G_per_weight is not None:
            for w_idx, g_list in G_per_weight.items():
                w_dir = event_dir / f"weight_{w_idx}"
                w_dir.mkdir(exist_ok=True)
                for si, g in enumerate(g_list):
                    np.save(w_dir / f"G_station_{si}.npy", g)


# ---------------------------------------------------------------------------
# EM loop
# ---------------------------------------------------------------------------

def run_em(
    n_cycles,
    initial_model,
    arrivals_table: Sequence[EventArrivals],
    station_locs: Sequence[MetricPoint],
    wave_type: str = 'P',
    solver: Union[str, object] = 'skfmm',
    lambda_reg: float = 1e-3,
    temperature: float = 1.0,
    weights_top_n: int = 1,
    subdivision: int = 1,
    true_model=None,
    event_locs: Optional[Sequence] = None,
    logger: Optional[TomographyLogger] = None,
    save_runs: bool = True,
    runs_dir: str = "runs",
):
    if save_runs and logger is None:
        logger = TomographyLogger(base_dir=runs_dir)

    if logger is not None:
        run_params = dict(
            n_cycles=n_cycles,
            wave_type=wave_type,
            solver=str(solver),
            lambda_reg=lambda_reg,
            temperature=temperature,
            weights_top_n=weights_top_n,
            subdivision=subdivision,
        )
        coarse_grid = initial_model.get_geo_grid(subdivision=1)
        grid_info = {
            "coarse_cell_size": float(coarse_grid.cell_size),
            "coarse_shape": [int(v) for v in coarse_grid.shape],
        }
        if subdivision > 1:
            fine_grid = initial_model.get_geo_grid(subdivision=subdivision)
            grid_info["fine_cell_size"] = float(fine_grid.cell_size)
            grid_info["fine_shape"] = [int(v) for v in fine_grid.shape]
        else:
            grid_info["fine_cell_size"] = float(coarse_grid.cell_size)
            grid_info["fine_shape"] = [int(v) for v in coarse_grid.shape]

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
        print(f'{i+1}/{n_cycles}')
        if logger is not None:
            logger.save_iteration_model(i, model)

        delta_s = make_tomography_step(
            model,
            arrivals_table,
            station_locs,
            wave_type,
            solver,
            lambda_reg,
            temperature,
            weights_top_n,
            subdivision,
            iteration=i,
            logger=logger,
        )

        new_velocities = 1 / (1 / model.get_geo_grid(subdivision=1).vp + delta_s)
        model.set_vp_array(new_velocities)

        if logger is not None:
            logger.save_delta_s(i, delta_s)

    if logger is not None:
        logger.save_iteration_model(n_cycles, model)
        print(f"[TomographyLogger] Run saved to: {logger.run_dir}")

    return logger


# ---------------------------------------------------------------------------
# One tomography step
# ---------------------------------------------------------------------------

def make_tomography_step(
        initial_model,
        arrivals_table: Sequence[EventArrivals],
        station_locs: Sequence[MetricPoint],
        wave_type: str = 'P',
        solver: Union[str, object] = 'skfmm',
        lambda_reg: float = 1e-3,
        temperature: float = 1.0,
        weights_top_n: int = 1,
        subdivision: int = 1,
        iteration: int = 0,
        logger: Optional[TomographyLogger] = None,
    ):
    coarse_grid = initial_model.get_geo_grid(subdivision=1)
    coarse_shape = tuple(int(v) for v in coarse_grid.shape)

    fine_grid = initial_model.get_geo_grid(subdivision=subdivision)
    fine_shape = tuple(int(v) for v in fine_grid.shape)
    fine_cell_size = float(fine_grid.cell_size)
    fine_voxel_size = (fine_cell_size,) * 3

    station_idx_fine = [metric_to_index(s, fine_cell_size, fine_shape) for s in station_locs]

    G_acc = []
    r_acc = []

    station_fields = compute_station_travel_time_fields(
        fine_grid, station_idx_fine, wave_type, solver
    )

    if logger is not None:
        logger.save_station_fields(iteration, np.array(station_fields))

    for event_idx, observed in enumerate(arrivals_table):
        observed = np.array(observed)

        # Get both weights and misfit in one call
        weights, misfit = compute_epicenter_weight_matrix(
            station_fields=station_fields,
            observed=observed,
            temperature=temperature,
            return_misfit=True,
        )
        weights = _select_top_n_weights(weights, weights_top_n, normolize=True)

        weights_indices = np.argwhere(weights > 0)
        weights_values = weights[weights > 0]

        G_weights = []
        r_weights = []

        # G_per_weight: {w_idx: [g_fine_st0, g_fine_st1, ...]}
        G_per_weight: Optional[Dict[int, List[np.ndarray]]] = {} if logger else None
        first_residuals: Optional[np.ndarray] = None

        for w_idx, (weight_idx, weight_val) in enumerate(
            zip(weights_indices, weights_values)
        ):
            G_stations = []
            g_fine_list: List[np.ndarray] = []

            for station_fine_idx, station_fine_loc in zip(
                range(len(station_idx_fine)), station_idx_fine
            ):
                station_field = station_fields[station_fine_idx]

                g_fine = _calculate_G(
                    station_field=station_field,
                    origin_loc=weight_idx,
                    station_loc=station_fine_loc,
                    geo_shape=fine_shape,
                    voxel_size=fine_voxel_size,
                )
                g_coarse = coarsen_G(g_fine, subdivision)
                G_stations.append(g_coarse)

                if logger is not None:
                    g_fine_list.append(g_fine)

            if logger is not None and G_per_weight is not None:
                G_per_weight[w_idx] = g_fine_list

            G_stations = np.array(G_stations)  # (n_stations, nx, ny, nz) coarse
            G_tilda = (
                G_stations[:, np.newaxis, :, :, :]
                - G_stations[np.newaxis, :, :, :, :]
            )
            residuals = _calculate_residuals(station_fields, observed, weight_idx)

            if first_residuals is None:
                first_residuals = residuals

            G_weights.append(G_tilda * weight_val)
            r_weights.append(residuals * weight_val)

        if logger is not None:
            logger.save_event_data(
                iteration=iteration,
                event_idx=event_idx,
                weights=weights,
                misfit=misfit,
                residuals=first_residuals if first_residuals is not None else np.array([]),
                G_per_weight=G_per_weight,
            )

        G_acc.append(sum(G_weights))
        r_acc.append(sum(r_weights))

    delta_s = _solve_delta_s(
        g_tilde_prime=sum(G_acc),
        r_prime=sum(r_acc),
        model_shape=coarse_shape,
        lambda_reg=lambda_reg,
        use_upper_triangle_pairs=True,
    )

    return delta_s


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _calculate_G(station_field, origin_loc, station_loc, geo_shape, voxel_size):
    path = trace_ray_from_timefield(
        T=station_field,
        station_xyz=station_loc,
        epic_xyz=origin_loc,
        spacing_xyz=(1.0, 1.0, 1.0),
    )
    G = rasterize_path_lengths(
        path_xyz=path,
        shape=geo_shape,
        voxel_size=voxel_size,
        dtype=np.float64,
    )
    return G


def _calculate_residuals(station_fields, arrivals_table, weight_idx):
    x, y, z = weight_idx
    predicted_arrivals = station_fields[:, x, y, z]
    residual_vector = arrivals_table - predicted_arrivals
    residual_matrix = residual_vector[:, np.newaxis] - residual_vector[np.newaxis, :]
    return residual_matrix


def _solve_delta_s(
        g_tilde_prime: np.ndarray,
        r_prime: np.ndarray,
        model_shape: Tuple[int, int, int],
        lambda_reg: float,
        use_upper_triangle_pairs: bool,
    ):
    g_tilde_prime = np.array(g_tilde_prime)
    n_stations = g_tilde_prime.shape[0]
    if n_stations != g_tilde_prime.shape[1]:
        raise ValueError("g_tilde_prime must have shape (n_stations, n_stations, ...)")
    if r_prime.shape != (n_stations, n_stations):
        raise ValueError("r_prime shape must be (n_stations, n_stations)")

    if use_upper_triangle_pairs:
        pair_mask = np.triu(np.ones((n_stations, n_stations), dtype=bool), k=1)
    else:
        pair_mask = ~np.eye(n_stations, dtype=bool)

    g_rows = g_tilde_prime[pair_mask].reshape((-1, int(np.prod(model_shape))))
    r_vec = r_prime[pair_mask].reshape(-1)
    if g_rows.shape[0] == 0:
        raise ValueError("No station pairs available to solve tomography system")

    gg_t = g_rows @ g_rows.T
    gg_t_reg = gg_t + float(lambda_reg) * np.eye(gg_t.shape[0], dtype=np.float64)

    try:
        alpha = np.linalg.solve(gg_t_reg, r_vec)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(gg_t_reg, r_vec, rcond=None)[0]

    delta_flat = g_rows.T @ alpha
    return delta_flat.reshape(model_shape)


def _select_top_n_weights(weights_martix, n, normolize=False):
    w = np.asarray(weights_martix, dtype=np.float64)
    if w.ndim != 3:
        raise ValueError("weights_martix must be a 3D array")
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
        flat = w.ravel()
        top_idx = np.argpartition(flat, -n)[-n:]
        out_flat = out.ravel()
        out_flat[top_idx] = flat[top_idx]

    if normolize:
        s = out.sum()
        if s > 0:
            out /= s

    return out