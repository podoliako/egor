"""
Pairwise travel-time misfit utilities for tomography workflows.

This module is intentionally separate from wave propagation and velocity model
logic. It combines observed station arrivals with predicted arrivals from a
target point and provides:
- pairwise misfit matrix for one target,
- SSE objective from pairwise residuals,
- full-grid cellwise misfit and epicenter probability weights.
"""
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from wave_propagation import WavePropagator


StationArrival = Dict[str, Union[Tuple[int, int, int], float, int]]
GridPoint = Tuple[int, int, int]


SyntheticArrival = Dict[str, Union[GridPoint, float]]
SyntheticEventArrivals = Dict[str, Union[GridPoint, List[SyntheticArrival]]]


def _validate_station(station: StationArrival, idx: int) -> Tuple[Tuple[int, int, int], float]:
    """Validate one station dictionary and normalize values."""
    if 'loc' not in station:
        raise ValueError(f"Station #{idx} must contain key 'loc'")
    if 'arrival_unix' not in station:
        raise ValueError(f"Station #{idx} must contain key 'arrival_unix'")

    loc = station['loc']
    arrival_unix = station['arrival_unix']

    if not isinstance(loc, tuple) or len(loc) != 3:
        raise ValueError(f"Station #{idx} 'loc' must be a tuple (i, j, k)")

    try:
        i, j, k = int(loc[0]), int(loc[1]), int(loc[2])
    except (TypeError, ValueError):
        raise ValueError(f"Station #{idx} 'loc' values must be integers")

    try:
        arrival = float(arrival_unix)
    except (TypeError, ValueError):
        raise ValueError(f"Station #{idx} 'arrival_unix' must be numeric")

    return (i, j, k), arrival


def compute_pairwise_misfit_matrix(
    grid,
    stations: Sequence[StationArrival],
    target_point: Tuple[int, int, int],
    wave_type: str = 'P',
    solver: Union[str, object] = 'simple'
) -> np.ndarray:
    """
    Compute pairwise differential-arrival misfit matrix for one target point.

    misfit[i, j] = (t_obs[i] - t_obs[j]) - (t_pred[i] - t_pred[j])
    """
    matrix, _ = compute_pairwise_misfit_matrix_and_sse(
        grid=grid,
        stations=stations,
        target_point=target_point,
        wave_type=wave_type,
        solver=solver
    )
    return matrix


def compute_pairwise_misfit_matrix_and_sse(
    grid,
    stations: Sequence[StationArrival],
    target_point: Tuple[int, int, int],
    wave_type: str = 'P',
    solver: Union[str, object] = 'simple'
) -> Tuple[np.ndarray, float]:
    """
    Compute pairwise misfit matrix and upper-triangle SSE in one pass.

    Returns:
    --------
    tuple[np.ndarray, float]
        (misfit_matrix, sse_upper_triangle)
    """
    if len(stations) < 2:
        raise ValueError("At least 2 stations are required for pairwise misfits")

    if not isinstance(target_point, tuple) or len(target_point) != 3:
        raise ValueError("target_point must be a tuple (i, j, k)")

    target = (int(target_point[0]), int(target_point[1]), int(target_point[2]))
    observed, predicted = _extract_observed_and_predicted(
        grid=grid,
        stations=stations,
        target=target,
        wave_type=wave_type,
        solver=solver
    )

    residual = observed - predicted
    misfit_matrix = residual[:, np.newaxis] - residual[np.newaxis, :]
    centered = residual - np.mean(residual, dtype=np.float64)
    sse = float(
        centered.size * np.sum(np.square(centered, dtype=np.float64), dtype=np.float64)
    )
    return misfit_matrix, sse


def pairwise_upper_triangle_sse_from_matrix(misfit_matrix: np.ndarray) -> float:
    """
    Sum of squared misfits above the main diagonal.

    Equivalent to np.sum(misfit_matrix[i, j]**2 for i < j).
    """
    if not isinstance(misfit_matrix, np.ndarray) or misfit_matrix.ndim != 2:
        raise ValueError("misfit_matrix must be a 2D numpy array")
    if misfit_matrix.shape[0] != misfit_matrix.shape[1]:
        raise ValueError("misfit_matrix must be square")

    idx = np.triu_indices(misfit_matrix.shape[0], k=1)
    return float(np.sum(np.square(misfit_matrix[idx], dtype=np.float64), dtype=np.float64))


def compute_pairwise_upper_triangle_sse(
    grid,
    stations: Sequence[StationArrival],
    target_point: Tuple[int, int, int],
    wave_type: str = 'P',
    solver: Union[str, object] = 'simple'
) -> float:
    """Compute upper-triangle pairwise SSE for one target point."""
    _, sse = compute_pairwise_misfit_matrix_and_sse(
        grid=grid,
        stations=stations,
        target_point=target_point,
        wave_type=wave_type,
        solver=solver
    )
    return sse


def precompute_station_travel_time_fields(
    grid,
    stations: Sequence[StationArrival],
    wave_type: str = 'P',
    solver: Union[str, object] = 'simple'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute travel-time fields from each station to all grid cells.

    Returns:
    --------
    tuple[np.ndarray, np.ndarray]
        (station_fields, observed_arrivals)
        station_fields shape: (n_stations, n_x, n_y, n_z)
        observed_arrivals shape: (n_stations,)
    """
    shape = tuple(int(v) for v in grid.shape)
    station_locs, observed = _extract_station_locs_and_observed(stations, shape)
    station_fields = _compute_station_travel_time_fields(
        grid=grid,
        station_locs=station_locs,
        wave_type=wave_type,
        solver=solver
    )
    return station_fields, observed


def generate_synthetic_arrivals_table(
    model,
    station_locs: Optional[Sequence[GridPoint]] = None,
    event_locs: Optional[Sequence[GridPoint]] = None,
    n_stations: Optional[int] = None,
    n_events: Optional[int] = None,
    wave_type: str = 'P',
    solver: Union[str, object] = 'skfmm',
    random_seed: Optional[int] = None
) -> List[SyntheticEventArrivals]:
    """
    Build synthetic relative-arrival table from a velocity model (subdivision=1).

    Output format:
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

    You can either pass explicit station/event coordinates or request random
    placement via n_stations/n_events.
    """
    geo_grid = model.get_geo_grid(subdivision=1)
    shape = tuple(int(v) for v in geo_grid.shape)
    rng = np.random.default_rng(seed=random_seed)

    stations = _resolve_station_locs(shape, station_locs, n_stations, rng)
    events = _resolve_event_locs(shape, event_locs, n_events, rng)

    fields = _compute_station_travel_time_fields(
        grid=geo_grid,
        station_locs=stations,
        wave_type=wave_type,
        solver=solver
    ).astype(np.float64, copy=False)

    synthetic: List[SyntheticEventArrivals] = []
    for event_loc in events:
        i, j, k = event_loc
        arrivals_abs = fields[:, i, j, k]

        if not np.all(np.isfinite(arrivals_abs)):
            raise ValueError(
                f"Non-finite travel times for event {event_loc}. "
                "Check model/solver settings."
            )

        t_min = float(np.min(arrivals_abs))
        arrivals_rel = arrivals_abs - t_min

        event_arrivals: List[SyntheticArrival] = []
        for station_idx, station_loc in enumerate(stations):
            event_arrivals.append({
                'station_loc': station_loc,
                'arrival_rel_s': float(arrivals_rel[station_idx])
            })

        synthetic.append({
            'event_loc': event_loc,
            'arrivals': event_arrivals
        })

    return synthetic


def compute_cellwise_pairwise_misfit(
    station_fields: np.ndarray,
    observed_arrivals: np.ndarray
) -> np.ndarray:
    """
    Compute pairwise-SSE misfit for every cell using precomputed station fields.

    Misfit at each cell is:
    sum_{i<j} ((obs_i - pred_i) - (obs_j - pred_j))^2
    """
    if not isinstance(station_fields, np.ndarray) or station_fields.ndim != 4:
        raise ValueError("station_fields must be a 4D numpy array (n_stations, n_x, n_y, n_z)")
    if not isinstance(observed_arrivals, np.ndarray) or observed_arrivals.ndim != 1:
        raise ValueError("observed_arrivals must be a 1D numpy array")
    if observed_arrivals.shape[0] != station_fields.shape[0]:
        raise ValueError("observed_arrivals length must equal number of station fields")
    if observed_arrivals.shape[0] < 2:
        raise ValueError("At least 2 stations are required")

    n_stations = station_fields.shape[0]
    sum_r = np.zeros(station_fields.shape[1:], dtype=np.float64)
    sum_r2 = np.zeros(station_fields.shape[1:], dtype=np.float64)

    for i in range(n_stations):
        residual = observed_arrivals[i] - station_fields[i].astype(np.float64)
        sum_r += residual
        sum_r2 += np.square(residual, dtype=np.float64)

    misfit = n_stations * sum_r2 - np.square(sum_r, dtype=np.float64)
    return np.maximum(misfit, 0.0)


def compute_epicenter_weight_matrix(
    grid,
    stations: Sequence[StationArrival],
    wave_type: str = 'P',
    solver: Union[str, object] = 'simple',
    abs_misfit_threshold: Optional[float] = None,
    temperature: float = 1.0,
    return_misfit: bool = False
):
    """
    Compute epicenter probability weights for all cells.

    Workflow:
    1. Precompute travel-time fields once per station.
    2. Compute pairwise-SSE misfit for every cell.
    3. Set weight=0 for cells with abs(misfit) > abs_misfit_threshold (if provided).
    4. Convert remaining misfits to probabilities and normalize to sum=1.

    Parameters:
    -----------
    abs_misfit_threshold : float or None
        Single threshold on abs(misfit). Cells above threshold get zero weight.
        If None, all cells are considered.
    temperature : float
        Controls sharpness of probability mapping (must be > 0).
    return_misfit : bool
        If True, returns (weights, misfit_grid).
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if abs_misfit_threshold is not None and abs_misfit_threshold < 0:
        raise ValueError("abs_misfit_threshold must be >= 0 when provided")

    station_fields, observed = precompute_station_travel_time_fields(
        grid=grid,
        stations=stations,
        wave_type=wave_type,
        solver=solver
    )
    misfit = compute_cellwise_pairwise_misfit(station_fields, observed)
    weights = _weights_from_misfit(misfit, abs_misfit_threshold, temperature)

    if return_misfit:
        return weights, misfit
    return weights


def _weights_from_misfit(
    misfit: np.ndarray,
    abs_misfit_threshold: Optional[float],
    temperature: float
) -> np.ndarray:
    """Map cellwise misfit to normalized weights with one misfit threshold."""
    if abs_misfit_threshold is None:
        mask = np.ones(misfit.shape, dtype=bool)
    else:
        mask = np.abs(misfit) <= abs_misfit_threshold

    if not np.any(mask):
        weights = np.zeros_like(misfit, dtype=np.float64)
        best = tuple(int(v) for v in np.unravel_index(np.argmin(misfit), misfit.shape))
        weights[best] = 1.0
        return weights

    selected = misfit[mask]
    delta = selected - float(np.min(selected))
    positive = delta[delta > 0]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, 1e-12)

    raw = np.exp(-delta / (temperature * scale))
    raw_sum = float(np.sum(raw, dtype=np.float64))

    weights = np.zeros_like(misfit, dtype=np.float64)
    if raw_sum <= 0:
        best_flat = int(np.argmin(selected))
        true_idx = np.argwhere(mask)[best_flat]
        weights[tuple(int(v) for v in true_idx)] = 1.0
        return weights

    weights[mask] = raw / raw_sum
    total = float(np.sum(weights, dtype=np.float64))
    if total > 0 and not np.isclose(total, 1.0):
        weights /= total
    return weights


def _resolve_station_locs(
    shape: Tuple[int, int, int],
    station_locs: Optional[Sequence[GridPoint]],
    n_stations: Optional[int],
    rng: np.random.Generator
) -> List[GridPoint]:
    """Use explicit station locations or sample random stations at k=0."""
    if station_locs is not None and n_stations is not None:
        raise ValueError("Provide either station_locs or n_stations, not both")

    if station_locs is not None:
        stations = _validate_points(station_locs, shape, name='station')
    else:
        if n_stations is None:
            raise ValueError("station_locs or n_stations must be provided")
        stations = _sample_random_points(shape, int(n_stations), rng, fixed_k=0)

    for idx, loc in enumerate(stations):
        if loc[2] != 0:
            raise ValueError(f"Station #{idx} must have k=0, got {loc}")

    if len(stations) == 0:
        raise ValueError("At least one station is required")

    return stations


def _resolve_event_locs(
    shape: Tuple[int, int, int],
    event_locs: Optional[Sequence[GridPoint]],
    n_events: Optional[int],
    rng: np.random.Generator
) -> List[GridPoint]:
    """Use explicit event locations or sample random events in full 3D grid."""
    if event_locs is not None and n_events is not None:
        raise ValueError("Provide either event_locs or n_events, not both")

    if event_locs is not None:
        events = _validate_points(event_locs, shape, name='event')
    else:
        if n_events is None:
            raise ValueError("event_locs or n_events must be provided")
        events = _sample_random_points(shape, int(n_events), rng, fixed_k=None)

    if len(events) == 0:
        raise ValueError("At least one event is required")

    return events


def _validate_points(
    points: Sequence[GridPoint],
    shape: Tuple[int, int, int],
    name: str
) -> List[GridPoint]:
    """Normalize point list to integer tuples and validate bounds."""
    n_x, n_y, n_z = shape
    normalized: List[GridPoint] = []

    for idx, point in enumerate(points):
        if not isinstance(point, tuple) or len(point) != 3:
            raise ValueError(f"{name.capitalize()} #{idx} must be tuple (i, j, k)")

        try:
            i, j, k = int(point[0]), int(point[1]), int(point[2])
        except (TypeError, ValueError):
            raise ValueError(
                f"{name.capitalize()} #{idx} must contain integer-like values, got {point}"
            )

        if not (0 <= i < n_x and 0 <= j < n_y and 0 <= k < n_z):
            raise ValueError(
                f"{name.capitalize()} #{idx} location {(i, j, k)} out of bounds for shape {shape}"
            )

        normalized.append((i, j, k))

    return normalized


def _sample_random_points(
    shape: Tuple[int, int, int],
    count: int,
    rng: np.random.Generator,
    fixed_k: Optional[int]
) -> List[GridPoint]:
    """Sample unique random grid points."""
    if count <= 0:
        raise ValueError("Random sample count must be > 0")

    n_x, n_y, n_z = shape
    points: List[GridPoint] = []

    if fixed_k is None:
        total = n_x * n_y * n_z
        if count > total:
            raise ValueError(f"Requested {count} points, but grid has only {total} cells")

        flat_idx = rng.choice(total, size=count, replace=False)
        for idx in flat_idx.tolist():
            i = idx // (n_y * n_z)
            rem = idx % (n_y * n_z)
            j = rem // n_z
            k = rem % n_z
            points.append((int(i), int(j), int(k)))
        return points

    if not (0 <= fixed_k < n_z):
        raise ValueError(f"fixed_k={fixed_k} out of bounds for n_z={n_z}")

    total = n_x * n_y
    if count > total:
        raise ValueError(
            f"Requested {count} points at k={fixed_k}, but only {total} unique positions exist"
        )

    flat_idx = rng.choice(total, size=count, replace=False)
    for idx in flat_idx.tolist():
        i = idx // n_y
        j = idx % n_y
        points.append((int(i), int(j), int(fixed_k)))

    return points


def _extract_observed_and_predicted(
    grid,
    stations: Sequence[StationArrival],
    target: Tuple[int, int, int],
    wave_type: str,
    solver: Union[str, object]
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect observed arrivals and predicted arrivals at station locations."""
    propagator = WavePropagator(solver=solver)
    travel_times = propagator.compute_from_geo_grid(
        geo_grid=grid,
        source_idx=target,
        wave_type=wave_type
    )

    n_x, n_y, n_z = travel_times.shape
    observed = np.empty(len(stations), dtype=np.float64)
    predicted = np.empty(len(stations), dtype=np.float64)

    for idx, station in enumerate(stations):
        loc, arrival = _validate_station(station, idx)
        i, j, k = loc

        if not (0 <= i < n_x and 0 <= j < n_y and 0 <= k < n_z):
            raise ValueError(
                f"Station #{idx} location {loc} out of bounds for shape {travel_times.shape}"
            )

        observed[idx] = arrival
        predicted[idx] = float(travel_times[i, j, k])

    return observed, predicted


def _extract_station_locs_and_observed(
    stations: Sequence[StationArrival],
    shape: Tuple[int, int, int]
) -> Tuple[List[Tuple[int, int, int]], np.ndarray]:
    """Validate station list and return station locations with observed arrivals."""
    n_x, n_y, n_z = shape
    station_locs: List[Tuple[int, int, int]] = []
    observed = np.empty(len(stations), dtype=np.float64)

    for idx, station in enumerate(stations):
        loc, arrival = _validate_station(station, idx)
        i, j, k = loc
        if not (0 <= i < n_x and 0 <= j < n_y and 0 <= k < n_z):
            raise ValueError(f"Station #{idx} location {loc} out of bounds for shape {shape}")
        station_locs.append(loc)
        observed[idx] = arrival

    return station_locs, observed


def _compute_station_travel_time_fields(
    grid,
    station_locs: Sequence[Tuple[int, int, int]],
    wave_type: str,
    solver: Union[str, object]
) -> np.ndarray:
    """Precompute travel-time fields from each station to all cells."""
    propagator = WavePropagator(solver=solver)
    fields = []
    for loc in station_locs:
        travel = propagator.compute_from_geo_grid(
            geo_grid=grid,
            source_idx=loc,
            wave_type=wave_type
        )
        fields.append(travel.astype(np.float32))
    return np.stack(fields, axis=0)
