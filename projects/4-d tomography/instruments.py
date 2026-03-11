"""
Local utilities
"""
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from wave_propagation import WavePropagator


StationArrival = Dict[str, Union[Tuple[int, int, int], float, int]]
GridPoint = Tuple[int, int, int]

# Координаты в метрах от угла сетки (x_m, y_m, z_m)
MetricPoint = Tuple[float, float, float]

SyntheticArrival = Dict[str, Union[GridPoint, float]]
SyntheticEventArrivals = Dict[str, Union[GridPoint, List[SyntheticArrival]]]


def metric_to_index(metric_point: MetricPoint, cell_size: float) -> GridPoint:
    """Конвертация метрических координат в индексы сетки с заданным cell_size."""
    return tuple(int(round(c / cell_size)) for c in metric_point)


def coarsen_G(G_fine: np.ndarray, subdivision: int) -> np.ndarray:
    """
    Схлопывает матрицу G с fine сетки (nx*sub, ny*sub, nz*sub)
    до coarse сетки (nx, ny, nz) суммированием по блокам sub×sub×sub.

    Физически корректно: сумма длин путей в под-вокселях = длина пути в крупном вокселе.
    """
    if subdivision == 1:
        return G_fine

    nx_f, ny_f, nz_f = G_fine.shape
    if nx_f % subdivision != 0 or ny_f % subdivision != 0 or nz_f % subdivision != 0:
        raise ValueError(
            f"Fine grid shape {G_fine.shape} not divisible by subdivision={subdivision}"
        )
    nx = nx_f // subdivision
    ny = ny_f // subdivision
    nz = nz_f // subdivision

    return (
        G_fine
        .reshape(nx, subdivision, ny, subdivision, nz, subdivision)
        .sum(axis=(1, 3, 5))
    )


def generate_synthetic_arrivals_table(
        model,
        station_locs: Optional[Sequence[MetricPoint]] = None,
        event_locs: Optional[Sequence[MetricPoint]] = None,
        n_stations: Optional[int] = None,
        n_events: Optional[int] = None,
        wave_type: str = 'P',
        solver: Union[str, object] = 'skfmm',
        random_seed: Optional[int] = None,
        subdivision: Optional[int] = 1,
    ):
    """
    station_locs / event_locs задаются в метрах от угла сетки: (x_m, y_m, z_m).
    При n_stations / n_events — случайная выборка, результирующие координаты
    также возвращаются в метрах.
    """
    geo_grid = model.get_geo_grid(subdivision=subdivision)
    cell_size = float(geo_grid.cell_size)
    shape = tuple(int(v) for v in geo_grid.shape)
    rng = np.random.default_rng(seed=random_seed)

    # Разрешаем метрические координаты → индексы fine сетки
    station_idx = _resolve_station_locs_metric(shape, cell_size, station_locs, n_stations, rng)
    event_idx   = _resolve_event_locs_metric(shape, cell_size, event_locs, n_events, rng)

    fields = compute_station_travel_time_fields(
        grid=geo_grid,
        station_locs=station_idx,
        wave_type=wave_type,
        solver=solver
    ).astype(np.float64, copy=False)

    synthetic: List[SyntheticEventArrivals] = []
    for event_loc in event_idx:
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
        for station_idx_i, _ in enumerate(station_idx):
            event_arrivals.append(float(arrivals_rel[station_idx_i]))

        synthetic.append(event_arrivals)

    # Возвращаем события в метрах (для внешнего использования)
    event_locs_metric = [
        tuple(c * cell_size for c in loc) for loc in event_idx
    ]
    return synthetic, event_locs_metric


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
    station_fields,
    observed,
    abs_misfit_threshold: Optional[float] = None,
    temperature: float = 1.0,
    return_misfit: bool = False
):
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if abs_misfit_threshold is not None and abs_misfit_threshold < 0:
        raise ValueError("abs_misfit_threshold must be >= 0 when provided")

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


# ---------------------------------------------------------------------------
# Внутренние хелперы — метрические координаты
# ---------------------------------------------------------------------------

def _resolve_station_locs_metric(
    shape: Tuple[int, int, int],
    cell_size: float,
    station_locs: Optional[Sequence[MetricPoint]],
    n_stations: Optional[int],
    rng: np.random.Generator
) -> List[GridPoint]:
    """
    Станции должны быть на поверхности (k=0, т.е. z_m=0).
    Принимает координаты в метрах, возвращает индексы fine сетки.
    """
    if station_locs is not None and n_stations is not None:
        raise ValueError("Provide either station_locs or n_stations, not both")

    if station_locs is not None:
        stations = _validate_metric_points(station_locs, shape, cell_size, name='station')
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


def _resolve_event_locs_metric(
    shape: Tuple[int, int, int],
    cell_size: float,
    event_locs: Optional[Sequence[MetricPoint]],
    n_events: Optional[int],
    rng: np.random.Generator
) -> List[GridPoint]:
    """
    Принимает координаты в метрах, возвращает индексы fine сетки.
    """
    if event_locs is not None and n_events is not None:
        raise ValueError("Provide either event_locs or n_events, not both")

    if event_locs is not None:
        events = _validate_metric_points(event_locs, shape, cell_size, name='event')
    else:
        if n_events is None:
            raise ValueError("event_locs or n_events must be provided")
        events = _sample_random_points(shape, int(n_events), rng, fixed_k=None)

    if len(events) == 0:
        raise ValueError("At least one event is required")

    return events


def _validate_metric_points(
    points: Sequence[MetricPoint],
    shape: Tuple[int, int, int],
    cell_size: float,
    name: str
) -> List[GridPoint]:
    """
    Конвертирует метрические координаты в индексы и проверяет границы.
    """
    n_x, n_y, n_z = shape
    normalized: List[GridPoint] = []

    for idx, point in enumerate(points):
        if not isinstance(point, (tuple, list)) or len(point) != 3:
            raise ValueError(f"{name.capitalize()} #{idx} must be tuple (x_m, y_m, z_m)")

        try:
            i, j, k = (int(round(float(point[0]) / cell_size)),
                       int(round(float(point[1]) / cell_size)),
                       int(round(float(point[2]) / cell_size)))
        except (TypeError, ValueError):
            raise ValueError(
                f"{name.capitalize()} #{idx} must contain numeric values, got {point}"
            )

        if not (0 <= i < n_x and 0 <= j < n_y and 0 <= k < n_z):
            raise ValueError(
                f"{name.capitalize()} #{idx} metric {tuple(point)} → index {(i, j, k)} "
                f"out of bounds for shape {shape} with cell_size={cell_size}"
            )

        normalized.append((i, j, k))

    return normalized


def _resolve_station_locs(
    shape: Tuple[int, int, int],
    station_locs: Optional[Sequence[GridPoint]],
    n_stations: Optional[int],
    rng: np.random.Generator
) -> List[GridPoint]:
    """Оставлено для обратной совместимости. Принимает индексные координаты."""
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
    """Оставлено для обратной совместимости. Принимает индексные координаты."""
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


def compute_station_travel_time_fields(
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