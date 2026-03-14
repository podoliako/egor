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
    Сохраняет все данные инверсии по итерациям и запускам.
    Структура директорий:
        runs/
          run_<timestamp>/
            meta.json
            initial_model.npy
            true_model.npy          (опционально)
            iter_<i>/
              model.npy
              delta_s.npy
              station_fields.npy    # (n_stations, nx, ny, nz)
              event_<j>/
                weights.npy
                residuals.npy
                G_station_<k>.npy
                G_s<a>_s<b>.npy     # G для каждой пары станций
    """

    def __init__(self, base_dir: str = "runs"):
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(base_dir) / f"run_{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def save_meta(
        self,
        run_params: dict,
        station_locs: Sequence,
        event_locs: Sequence,
        grid_info=None
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

    def save_initial_model(self, model):
        np.save(self.run_dir / "initial_model.npy", model.get_geo_grid(subdivision=1).vp)

    def save_true_model(self, model):
        if model is not None:
            np.save(self.run_dir / "true_model.npy", model.get_geo_grid(subdivision=1).vp)

    def iter_dir(self, iteration: int) -> Path:
        d = self.run_dir / f"iter_{iteration}"
        d.mkdir(exist_ok=True)
        return d

    def save_iteration_model(self, iteration: int, model):
        np.save(self.iter_dir(iteration) / "model.npy", model.get_geo_grid(subdivision=1).vp)

    def save_delta_s(self, iteration: int, delta_s: np.ndarray):
        np.save(self.iter_dir(iteration) / "delta_s.npy", delta_s)

    def save_station_fields(self, iteration: int, station_fields: np.ndarray):
        # station_fields: (n_stations, nx, ny, nz)
        np.save(self.iter_dir(iteration) / "station_fields.npy", station_fields)

    def save_event_data(
        self,
        iteration: int,
        event_idx: int,
        weights: np.ndarray,
        residuals: np.ndarray,
        G_per_station: Optional[List[np.ndarray]] = None,
    ):
        event_dir = self.iter_dir(iteration) / f"event_{event_idx}"
        event_dir.mkdir(exist_ok=True)
        np.save(event_dir / "weights.npy", weights)
        np.save(event_dir / "residuals.npy", residuals)
        if G_per_station is not None:
            for si, g in enumerate(G_per_station):
                np.save(event_dir / f"G_station_{si}.npy", g)

    def save_G_pair(
        self,
        iteration: int,
        event_idx: int,
        sta_a: int,
        sta_b: int,
        G: np.ndarray,
    ):
        event_dir = self.iter_dir(iteration) / f"event_{event_idx}"
        event_dir.mkdir(exist_ok=True)
        np.save(event_dir / f"G_s{sta_a}_s{sta_b}.npy", G)


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
    # --- новые параметры ---
    true_model=None,
    event_locs: Optional[Sequence] = None,
    logger: Optional[TomographyLogger] = None,
    save_runs: bool = True,
    runs_dir: str = "runs",
):
    # Инициализируем логгер если нужен
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
            "coarse_shape":     [int(v) for v in coarse_grid.shape],
        }
        if subdivision > 1:
            fine_grid = initial_model.get_geo_grid(subdivision=subdivision)
            grid_info["fine_cell_size"] = float(fine_grid.cell_size)
            grid_info["fine_shape"]     = [int(v) for v in fine_grid.shape]
        else:
            grid_info["fine_cell_size"] = float(coarse_grid.cell_size)
            grid_info["fine_shape"]     = [int(v) for v in coarse_grid.shape]
        
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

    # Сохраняем финальную модель как iter_n_cycles
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
        # --- логирование ---
        iteration: int = 0,
        logger: Optional[TomographyLogger] = None,
    ):
    """
    station_locs — координаты станций в метрах: [(x_m, y_m, z_m), ...].
    """
    coarse_grid = initial_model.get_geo_grid(subdivision=1)
    coarse_shape = tuple(int(v) for v in coarse_grid.shape)

    fine_grid = initial_model.get_geo_grid(subdivision=subdivision)
    fine_shape = tuple(int(v) for v in fine_grid.shape)
    fine_cell_size = float(fine_grid.cell_size)
    fine_voxel_size = (fine_cell_size,) * 3

    station_idx_fine = [metric_to_index(s, fine_cell_size) for s in station_locs]

    G = []
    r = []

    station_fields = compute_station_travel_time_fields(
        fine_grid, station_idx_fine, wave_type, solver
    )

    # Сохраняем поля станций один раз на итерацию
    if logger is not None:
        logger.save_station_fields(iteration, np.array(station_fields))

    for event_idx, observed in enumerate(arrivals_table):
        observed = np.array(observed)

        weights = compute_epicenter_weight_matrix(
            station_fields=station_fields,
            observed=observed,
            temperature=temperature,
        )
        weights = _select_top_n_weights(weights, weights_top_n, normolize=True)

        weights_indices = np.argwhere(weights > 0)
        weights_values = weights[weights > 0]

        G_weights = []
        r_weights = []

        # Для логирования: собираем G по станциям для первого ненулевого веса
        event_G_stations_log: Optional[List[np.ndarray]] = [] if logger else None
        event_residuals_log: Optional[np.ndarray] = None

        for weight_idx, weight_val in zip(weights_indices, weights_values):
            G_stations = []

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

                # Логируем G на fine сетке для каждой станции
                if logger is not None and event_G_stations_log is not None:
                    event_G_stations_log.append(g_fine)

            G_stations = np.array(G_stations)
            G_tilda = (
                G_stations[:, np.newaxis, :, :, :]
                - G_stations[np.newaxis, :, :, :, :]
            )
            residuals = _calculate_residuals(station_fields, observed, weight_idx)

            if event_residuals_log is None:
                event_residuals_log = residuals

            G_weights.append(G_tilda * weight_val)
            r_weights.append(residuals * weight_val)

        # Сохраняем данные по событию
        if logger is not None:
            logger.save_event_data(
                iteration=iteration,
                event_idx=event_idx,
                weights=weights,
                residuals=event_residuals_log if event_residuals_log is not None else np.array([]),
                G_per_station=event_G_stations_log,
            )
            # Также сохраняем G_tilda для верхнего треугольника пар
            if len(G_weights) > 0:
                G_tilda_sum = sum(G_weights)
                n_st = G_tilda_sum.shape[0]
                for sa in range(n_st):
                    for sb in range(sa + 1, n_st):
                        logger.save_G_pair(
                            iteration=iteration,
                            event_idx=event_idx,
                            sta_a=sa,
                            sta_b=sb,
                            G=G_tilda_sum[sa, sb],
                        )

        G.append(sum(G_weights))
        r.append(sum(r_weights))

    delta_s = _solve_delta_s(
        g_tilde_prime=sum(G),
        r_prime=sum(r),
        model_shape=coarse_shape,
        lambda_reg=lambda_reg,
        use_upper_triangle_pairs=True,
    )

    return delta_s


# ---------------------------------------------------------------------------
# Helpers (без изменений)
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
        use_upper_triangle_pairs: bool
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
    total = w.size

    if n == 0:
        return out

    if n >= total:
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