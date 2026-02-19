"""
Prototype tomography inversion from arrival tables.

The module expects arrival tables with the same structure as produced by
generate_synthetic_arrivals_table() from instruments.py:
[
    {
        "event_loc": (i, j, k),
        "arrivals": [
            {"station_loc": (si, sj, sk), "arrival_rel_s": t_rel},
            ...
        ]
    },
    ...
]
"""
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from instruments import compute_epicenter_weight_matrix, precompute_station_travel_time_fields
from raytracing import trace_ray_from_timefield, rasterize_path_lengths


GridPoint = Tuple[int, int, int]
StationArrival = Dict[str, Union[GridPoint, float]]
EventArrivals = Dict[str, Union[GridPoint, List[StationArrival]]]
TomographyEventResult = Dict[str, Union[int, GridPoint, np.ndarray]]


def run_tomography_prototype(
    initial_model,
    arrivals_table: Sequence[EventArrivals],
    wave_type: str = 'P',
    solver: Union[str, object] = 'skfmm',
    lambda_reg: float = 1e-3,
    abs_misfit_threshold: Optional[float] = None,
    temperature: float = 1.0,
    weight_epsilon: float = 0.0,
    include_intermediate: bool = False,
    use_upper_triangle_pairs: bool = True
) -> List[TomographyEventResult]:
    """
    Compute prototype slowness update Delta_s for each event.

    For each event:
    1) compute epicenter weights matrix,
    2) for all cells with non-zero weight:
       - compute pairwise residual matrix r(i, j),
       - compute station sensitivity 3D G for each station,
       - compute pairwise difference tensor G_tilde(i, j, :, :, :),
    3) accumulate weighted sums:
       G_tilde_prime = sum(weight * G_tilde)
       r_prime       = sum(weight * r),
    4) solve ridge-regularized system for Delta_s:
       Delta_s = (G^T G + lambda I)^-1 G^T r.

    Notes:
    - Coordinates are interpreted in subdivision=1 geo-grid indices.
    - Stations must be on surface (k=0).
    - Delta_s is returned but not applied to initial_model.
    """
    if lambda_reg <= 0:
        raise ValueError("lambda_reg must be > 0")
    if weight_epsilon < 0:
        raise ValueError("weight_epsilon must be >= 0")

    geo_grid = initial_model.get_geo_grid(subdivision=1)
    shape = tuple(int(v) for v in geo_grid.shape)
    voxel_size = (
        float(geo_grid.cell_size),
        float(geo_grid.cell_size),
        float(geo_grid.cell_size)
    )

    results: List[TomographyEventResult] = []
    for event_index, event in enumerate(arrivals_table):
        event_loc, station_obs = _parse_event_arrivals(event, event_index, shape)
        n_stations = len(station_obs)
        if n_stations < 2:
            raise ValueError(
                f"Event #{event_index} must have at least 2 stations, got {n_stations}"
            )

        weights = compute_epicenter_weight_matrix(
            grid=geo_grid,
            stations=station_obs,
            wave_type=wave_type,
            solver=solver,
            abs_misfit_threshold=abs_misfit_threshold,
            temperature=temperature
        )

        station_fields, observed_arrivals = precompute_station_travel_time_fields(
            grid=geo_grid,
            stations=station_obs,
            wave_type=wave_type,
            solver=solver
        )
        station_fields = station_fields.astype(np.float64, copy=False)
        observed_arrivals = observed_arrivals.astype(np.float64, copy=False)

        g_tilde_prime, r_prime, weighted_cell_count = _build_weighted_pairwise_system(
            weights=weights,
            station_fields=station_fields,
            observed_arrivals=observed_arrivals,
            station_obs=station_obs,
            voxel_size=voxel_size,
            weight_epsilon=weight_epsilon
        )

        delta_s = _solve_delta_s(
            g_tilde_prime=g_tilde_prime,
            r_prime=r_prime,
            model_shape=shape,
            lambda_reg=lambda_reg,
            use_upper_triangle_pairs=use_upper_triangle_pairs
        )

        event_result: TomographyEventResult = {
            'event_index': int(event_index),
            'event_loc': event_loc,
            'weighted_cell_count': int(weighted_cell_count),
            'weights': weights,
            'delta_s': delta_s
        }
        if include_intermediate:
            event_result['r_prime'] = r_prime
            event_result['g_tilde_prime'] = g_tilde_prime
        results.append(event_result)

    return results


def _build_weighted_pairwise_system(
    weights: np.ndarray,
    station_fields: np.ndarray,
    observed_arrivals: np.ndarray,
    station_obs: Sequence[Dict[str, Union[GridPoint, float]]],
    voxel_size: Tuple[float, float, float],
    weight_epsilon: float
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Build weighted G_tilde_prime and r_prime for one event.
    """
    n_stations = station_fields.shape[0]
    shape = tuple(int(v) for v in station_fields.shape[1:])
    station_locs = [tuple(int(v) for v in st['loc']) for st in station_obs]

    g_tilde_prime = np.zeros((n_stations, n_stations, *shape), dtype=np.float64)
    r_prime = np.zeros((n_stations, n_stations), dtype=np.float64)

    weighted_cells = np.argwhere(weights > weight_epsilon)
    if weighted_cells.size == 0:
        raise ValueError(
            "No cells with non-zero weight after thresholding. "
            "Lower weight_epsilon or relax abs_misfit_threshold."
        )

    origin_xyz = (0.0, 0.0, 0.0)
    spacing_xyz = (1.0, 1.0, 1.0)

    for cell_idx in weighted_cells:
        ci, cj, ck = (int(cell_idx[0]), int(cell_idx[1]), int(cell_idx[2]))
        cell_weight = float(weights[ci, cj, ck])
        if cell_weight <= 0:
            continue

        predicted_arrivals = station_fields[:, ci, cj, ck]
        residual_vector = observed_arrivals - predicted_arrivals
        residual_matrix = residual_vector[:, np.newaxis] - residual_vector[np.newaxis, :]
        r_prime += cell_weight * residual_matrix

        g_station = np.zeros((n_stations, *shape), dtype=np.float64)
        for station_idx, station_loc in enumerate(station_locs):
            path = trace_ray_from_timefield(
                T=station_fields[station_idx],
                station_xyz=station_loc,
                epic_xyz=(ci, cj, ck),
                origin_xyz=origin_xyz,
                spacing_xyz=spacing_xyz
            )
            g_station[station_idx] = rasterize_path_lengths(
                path_xyz=path,
                shape=shape,
                voxel_size=voxel_size,
                dtype=np.float64
            )

        g_tilde = g_station[:, np.newaxis, :, :, :] - g_station[np.newaxis, :, :, :, :]
        g_tilde_prime += cell_weight * g_tilde

    return g_tilde_prime, r_prime, int(weighted_cells.shape[0])


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
        if 'station_loc' not in station_entry:
            raise ValueError(
                f"Event #{event_index} station #{station_idx} must contain 'station_loc'"
            )
        if 'arrival_rel_s' not in station_entry:
            raise ValueError(
                f"Event #{event_index} station #{station_idx} must contain 'arrival_rel_s'"
            )

        station_loc = _normalize_grid_point(
            station_entry['station_loc'],
            shape,
            f"Event #{event_index} station #{station_idx} station_loc"
        )
        if station_loc[2] != 0:
            raise ValueError(
                f"Event #{event_index} station #{station_idx} must be on surface (k=0), "
                f"got {station_loc}"
            )

        try:
            arrival_rel_s = float(station_entry['arrival_rel_s'])
        except (TypeError, ValueError):
            raise ValueError(
                f"Event #{event_index} station #{station_idx} arrival_rel_s must be numeric"
            )

        stations.append({'loc': station_loc, 'arrival_unix': arrival_rel_s})

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
