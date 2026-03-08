"""
Prototype tomography inversion from arrival tables.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from time import perf_counter

from instruments import compute_epicenter_weight_matrix, _compute_station_travel_time_fields
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
    station_locs,
    wave_type: str = 'P',
    solver: Union[str, object] = 'skfmm',
    lambda_reg: float = 1e-3,
    abs_misfit_threshold: Optional[float] = None,
    temperature: float = 1.0,
    weight_epsilon: float = 0.0,
    weights_top_n: int = 1  
):
    model = initial_model
    for i in range(n_cycles):
        delta_s = run_tomography_prototype(
            initial_model,
            arrivals_table,
            station_locs,
            wave_type,
            solver,
            lambda_reg,
            abs_misfit_threshold,
            temperature,
            weight_epsilon,
            weights_top_n
        )
        model.set_vp_array(1/(1/(model.get_geo_grid(subdivision=1).vp) + delta_s))
        print(f"iteration: {i+1}")
        simple_heatmap(model.get_geo_grid(subdivision=1).vp[:,1,:])


def run_tomography_prototype(
    initial_model,
    arrivals_table: Sequence[EventArrivals],
    station_locs,
    wave_type: str = 'P',
    solver: Union[str, object] = 'skfmm',
    lambda_reg: float = 1e-3,
    abs_misfit_threshold: Optional[float] = None,
    temperature: float = 1.0,
    weight_epsilon: float = 0.0,
    weights_top_n: int = 1
) -> Dict[str, Any]:
    """
    Global tomography inversion over all events simultaneously.

    Objective:
        min_x sum_e ||G_e x - r_e||^2 + lambda_reg * ||x||^2
    where each (G_e, r_e) is built from one event.
    """
    if lambda_reg <= 0:
        raise ValueError("lambda_reg must be > 0")
    if weight_epsilon < 0:
        raise ValueError("weight_epsilon must be >= 0")
    if weights_top_n is not None and weights_top_n < 1:
        raise ValueError("weights_top_n must be >= 1 or None")

    geo_grid = initial_model.get_geo_grid(subdivision=1)
    shape = tuple(int(v) for v in geo_grid.shape)
    voxel_size = (float(geo_grid.cell_size),) * 3

    G = []
    r = []

    station_fields = _compute_station_travel_time_fields(geo_grid, station_locs, wave_type, solver)

    for observed in arrivals_table:
        observed = np.array(observed)
        weights = compute_epicenter_weight_matrix(
            station_fields=station_fields,
            observed=observed,
            abs_misfit_threshold=abs_misfit_threshold,
            temperature=temperature
        )
        weights = _select_top_n_weights(weights, weights_top_n, normolize=True)
        weights_indices = np.argwhere(weights > 0)
        weights_values = weights[weights > 0] 
        G_weights = []
        r_weights = []
        for weight_idx, weight_val in zip(weights_indices, weights_values):
            G_stations = []
            
            for station_index, station_loc in enumerate(station_locs):
                g = calculate_G(station_fields, weight_idx, station_loc, station_index, shape, voxel_size)
                G_stations.append(g)

            G_stations = np.array(G_stations)
            g_tilda = G_stations[:, np.newaxis, :, :, :] - G_stations[np.newaxis, :, :, :, :]
            residuals = calculate_residuals(station_fields, observed, weight_idx)

            G_weights.append(g_tilda * weight_val)
            r_weights.append(residuals * weight_val)

        G.append(sum(G_weights))
        r.append(sum(r_weights))

    delta_s = _solve_delta_s(
        g_tilde_prime=sum(G),
        r_prime=sum(r),
        model_shape=shape,
        lambda_reg=lambda_reg,
        use_upper_triangle_pairs=True 
    )

    return delta_s

def calculate_G(station_fields, weight_idx, station_loc, station_index, shape, voxel_size):
    path = trace_ray_from_timefield(
                T=station_fields[station_index],
                station_xyz=station_loc,
                epic_xyz=weight_idx,
                spacing_xyz=(1.0, 1.0, 1.0)
            )

    G = rasterize_path_lengths(
            path_xyz=path,
            shape=shape,
            voxel_size=voxel_size,
            dtype=np.float64
        )
    return G

def calculate_residuals(station_fields, arrivals_table, weight_idx):
    x, y, z = weight_idx
    predicted_arrivals = station_fields[:, x, y, z]
    residual_vector = arrivals_table - predicted_arrivals
    residual_matrix = residual_vector[:, np.newaxis] - residual_vector[np.newaxis, :]
    return residual_matrix


def _pairwise_to_rows(
    g_tilde_prime: np.ndarray,
    r_prime: np.ndarray,
    model_shape: Tuple[int, int, int],
    use_upper_triangle_pairs: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert pairwise tensors/matrices into linear rows for inversion."""
    n_stations = g_tilde_prime.shape[0]
    if g_tilde_prime.shape[1] != n_stations:
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
        raise ValueError("No station pairs available to build inversion rows")

    return g_rows, r_vec


def _solve_delta_s_from_rows(
    g_rows: np.ndarray,
    r_vec: np.ndarray,
    model_shape: Tuple[int, int, int],
    lambda_reg: float
) -> np.ndarray:
    """
    Solve ridge system in dual form:
        x = G^T (G G^T + lambda I)^-1 r
    """
    if g_rows.ndim != 2:
        raise ValueError("g_rows must be 2D")
    if r_vec.ndim != 1:
        raise ValueError("r_vec must be 1D")
    if g_rows.shape[0] != r_vec.shape[0]:
        raise ValueError("g_rows and r_vec row counts must match")

    gg_t = g_rows @ g_rows.T
    gg_t_reg = gg_t + float(lambda_reg) * np.eye(gg_t.shape[0], dtype=np.float64)

    try:
        alpha = np.linalg.solve(gg_t_reg, r_vec)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(gg_t_reg, r_vec, rcond=None)[0]

    delta_flat = g_rows.T @ alpha
    return delta_flat.reshape(model_shape)


def _build_weighted_pairwise_system(
    weights: np.ndarray,
    station_fields: np.ndarray,
    observed_arrivals: np.ndarray,
    station_obs: Sequence[Dict[str, Union[GridPoint, float]]],
    voxel_size: Tuple[float, float, float],
    weight_epsilon: float
) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, float]]:
    """
    Build weighted G_tilde_prime and r_prime for one event.
    """
    stage_start = perf_counter()
    n_stations = station_fields.shape[0]
    shape = tuple(int(v) for v in station_fields.shape[1:])
    station_locs = [tuple(int(v) for v in st['loc']) for st in station_obs]

    g_tilde_prime = np.zeros((n_stations, n_stations, *shape), dtype=np.float64)
    r_prime = np.zeros((n_stations, n_stations), dtype=np.float64)

    t_cells_start = perf_counter()
    weighted_cells = np.argwhere(weights > weight_epsilon)
    select_weighted_cells_s = perf_counter() - t_cells_start
    if weighted_cells.size == 0:
        raise ValueError(
            "No cells with non-zero weight after thresholding. "
            "Lower weight_epsilon or relax abs_misfit_threshold."
        )

    origin_xyz = (0.0, 0.0, 0.0)
    spacing_xyz = (1.0, 1.0, 1.0)
    residual_s = 0.0
    raytrace_s = 0.0
    rasterize_s = 0.0
    g_tilde_s = 0.0
    ray_count = 0

    for cell_idx in weighted_cells:
        ci, cj, ck = (int(cell_idx[0]), int(cell_idx[1]), int(cell_idx[2]))
        cell_weight = float(weights[ci, cj, ck])
        if cell_weight <= 0:
            continue

        t_residual_start = perf_counter()
        predicted_arrivals = station_fields[:, ci, cj, ck]
        residual_vector = observed_arrivals - predicted_arrivals
        residual_matrix = residual_vector[:, np.newaxis] - residual_vector[np.newaxis, :]
        r_prime += cell_weight * residual_matrix
        residual_s += perf_counter() - t_residual_start

        g_station = np.zeros((n_stations, *shape), dtype=np.float64)
        for station_idx, station_loc in enumerate(station_locs):
            t_ray_start = perf_counter()
            path = trace_ray_from_timefield(
                T=station_fields[station_idx],
                station_xyz=station_loc,
                epic_xyz=(ci, cj, ck),
                origin_xyz=origin_xyz,
                spacing_xyz=spacing_xyz
            )
            raytrace_s += perf_counter() - t_ray_start

            t_raster_start = perf_counter()
            g_station[station_idx] = rasterize_path_lengths(
                path_xyz=path,
                shape=shape,
                voxel_size=voxel_size,
                dtype=np.float64
            )
            rasterize_s += perf_counter() - t_raster_start
            ray_count += 1

        t_g_tilde_start = perf_counter()
        g_tilde = g_station[:, np.newaxis, :, :, :] - g_station[np.newaxis, :, :, :, :]
        g_tilde_prime += cell_weight * g_tilde
        g_tilde_s += perf_counter() - t_g_tilde_start

    build_total_s = perf_counter() - stage_start
    timings = {
        'select_weighted_cells': float(select_weighted_cells_s),
        'residuals': float(residual_s),
        'raytrace': float(raytrace_s),
        'rasterize': float(rasterize_s),
        'g_tilde': float(g_tilde_s),
        'build_total': float(build_total_s),
        'weighted_cell_count': float(weighted_cells.shape[0]),
        'ray_count': float(ray_count)
    }
    return g_tilde_prime, r_prime, int(weighted_cells.shape[0]), timings


def _solve_delta_s(
    g_tilde_prime: np.ndarray,
    r_prime: np.ndarray,
    model_shape: Tuple[int, int, int],
    lambda_reg: float,
    use_upper_triangle_pairs: bool
) -> np.ndarray:
    """
    Solve Delta_s from weighted pairwise system.

    To avoid allocating huge (n_cells x n_cells) matrices, this uses the
    equivalent dual ridge form:
      x = G^T (G G^T + lambda I)^-1 r
    """
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

    # Dual solve for numerical practicality on 3D grids.
    gg_t = g_rows @ g_rows.T
    gg_t_reg = gg_t + float(lambda_reg) * np.eye(gg_t.shape[0], dtype=np.float64)

    try:
        alpha = np.linalg.solve(gg_t_reg, r_vec)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(gg_t_reg, r_vec, rcond=None)[0]

    delta_flat = g_rows.T @ alpha
    return delta_flat.reshape(model_shape)


def _parse_event_arrivals(
    event: EventArrivals,
    event_index: int,
    shape: Tuple[int, int, int]
) -> Tuple[GridPoint, List[Dict[str, Union[GridPoint, float]]]]:
    """Normalize one event entry and convert to stations format used elsewhere."""
    if not isinstance(event, dict):
        raise ValueError(f"Event #{event_index} must be a dictionary")
    if 'event_loc' not in event:
        raise ValueError(f"Event #{event_index} must contain 'event_loc'")
    if 'arrivals' not in event:
        raise ValueError(f"Event #{event_index} must contain 'arrivals'")

    event_loc = _normalize_grid_point(event['event_loc'], shape, f"Event #{event_index} event_loc")
    arrivals = event['arrivals']
    if not isinstance(arrivals, list):
        raise ValueError(f"Event #{event_index} 'arrivals' must be a list")
    if len(arrivals) == 0:
        raise ValueError(f"Event #{event_index} 'arrivals' must not be empty")

    stations: List[Dict[str, Union[GridPoint, float]]] = []
    for station_idx, station_entry in enumerate(arrivals):
        if not isinstance(station_entry, dict):
            raise ValueError(
                f"Event #{event_index} station #{station_idx} must be a dictionary"
            )
        if 'loc' not in station_entry:
            raise ValueError(
                f"Event #{event_index} station #{station_idx} must contain 'loc'"
            )
        if 'arrival' not in station_entry:
            raise ValueError(
                f"Event #{event_index} station #{station_idx} must contain 'arrival'"
            )

        station_loc = _normalize_grid_point(
            station_entry['loc'],
            shape,
            f"Event #{event_index} station #{station_idx} station_loc"
        )
        if station_loc[2] != 0:
            raise ValueError(
                f"Event #{event_index} station #{station_idx} must be on surface (k=0), "
                f"got {station_loc}"
            )

        try:
            arrival_rel_s = float(station_entry['arrival'])
        except (TypeError, ValueError):
            raise ValueError(
                f"Event #{event_index} station #{station_idx} arrival_rel_s must be numeric"
            )

        stations.append({'loc': station_loc, 'arrival': arrival_rel_s})

    return event_loc, stations


def _normalize_grid_point(
    value,
    shape: Tuple[int, int, int],
    label: str
) -> GridPoint:
    """Convert list/tuple point to int tuple and validate bounds."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must be a 3-element list/tuple")

    try:
        i, j, k = int(value[0]), int(value[1]), int(value[2])
    except (TypeError, ValueError):
        raise ValueError(f"{label} must contain integer-like values")

    n_x, n_y, n_z = shape
    if not (0 <= i < n_x and 0 <= j < n_y and 0 <= k < n_z):
        raise ValueError(f"{label} {(i, j, k)} out of bounds for shape {shape}")

    return (i, j, k)


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
        top_idx = np.argpartition(flat, -n)[-n:]  # indices of n largest values
        out_flat = out.ravel()
        out_flat[top_idx] = flat[top_idx]

    if normolize:
        s = out.sum()
        if s > 0:
            out /= s

    return out