"""
Prototype tomography inversion from arrival tables.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
from components.graphics import simple_heatmap


GridPoint = Tuple[int, int, int]
StationArrival = Dict[str, Union[GridPoint, float]]
EventArrivals = Dict[str, Union[GridPoint, List[StationArrival]]]
TomographyEventResult = Dict[str, Union[int, GridPoint, np.ndarray]]


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
):
    model = initial_model
    for i in range(n_cycles):
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
        )
        # delta_s имеет форму coarse сетки (subdivision=1)
        new_velocities = 1 / (1 / model.get_geo_grid(subdivision=1).vp + delta_s)
        model.set_vp_array(new_velocities)

        # print(f"iteration: {i+1}")
        # simple_heatmap(model.get_geo_grid(subdivision=1).vp[:, 0, :])


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
    ):
    """
    station_locs — координаты станций в метрах: [(x_m, y_m, z_m), ...].

    Внутри:
      - travel time fields считаются на fine сетке (subdivision)
      - G считается на fine сетке, затем схлопывается до coarse (subdivision=1)
      - система Ax=b решается для coarse сетки
    """
    # --- Coarse сетка (для решения системы и обновления модели) ---
    coarse_grid = initial_model.get_geo_grid(subdivision=1)
    coarse_shape = tuple(int(v) for v in coarse_grid.shape)

    # --- Fine сетка (для точного ray tracing) ---
    fine_grid = initial_model.get_geo_grid(subdivision=subdivision)
    fine_shape = tuple(int(v) for v in fine_grid.shape)
    fine_cell_size = float(fine_grid.cell_size)
    fine_voxel_size = (fine_cell_size,) * 3

    # simple_heatmap(fine_grid.vp[:, 0, :])

    # Конвертируем метрические координаты станций → индексы fine сетки
    station_idx_fine = [metric_to_index(s, fine_cell_size) for s in station_locs]

    G = []
    r = []

    station_fields = compute_station_travel_time_fields(
        fine_grid, station_idx_fine, wave_type, solver
    )

    for observed in arrivals_table:
        observed = np.array(observed)

        # Веса считаются на fine сетке
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

        for weight_idx, weight_val in zip(weights_indices, weights_values):
            G_stations = []

            for station_fine_idx, station_fine_loc in zip(
                range(len(station_idx_fine)), station_idx_fine
            ):
                station_field = station_fields[station_fine_idx]

                g_fine = _calculate_G(
                    station_field=station_field,
                    origin_loc=weight_idx,       # индекс на fine сетке
                    station_loc=station_fine_loc,
                    geo_shape=fine_shape,
                    voxel_size=fine_voxel_size,
                )
                print('g_heatmap')
                # simple_heatmap(g_fine[:, 0, :])
                time.sleep(1)
                # Схлопываем G с fine до coarse
                g_coarse = coarsen_G(g_fine, subdivision)
                G_stations.append(g_coarse)

            G_stations = np.array(G_stations)  # (n_stations, nx, ny, nz) coarse
            G_tilda = (
                G_stations[:, np.newaxis, :, :, :]
                - G_stations[np.newaxis, :, :, :, :]
            )
            residuals = _calculate_residuals(station_fields, observed, weight_idx)

            G_weights.append(G_tilda * weight_val)
            r_weights.append(residuals * weight_val)

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


def _calculate_G(station_field, origin_loc, station_loc, geo_shape, voxel_size):
    path = trace_ray_from_timefield(
        T=station_field,
        station_xyz=station_loc,
        epic_xyz=origin_loc,
        spacing_xyz=(1.0, 1.0, 1.0),   # единичный шаг в индексном пространстве
    )

    G = rasterize_path_lengths(
        path_xyz=path,
        shape=geo_shape,
        voxel_size=voxel_size,          # реальный размер вокселя fine сетки
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