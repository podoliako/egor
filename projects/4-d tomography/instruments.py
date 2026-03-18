"""
Local utilities
"""
from typing import Dict, List, Optional, Sequence, Tuple, Union
from math import floor

import numpy as np

from wave_propagation import WavePropagator


StationArrival = Dict[str, Union[Tuple[int, int, int], float, int]]
GridPoint = Tuple[int, int, int]

# Координаты в метрах от верхнего угла сетки (x_m, y_m, z_m)
MetricPoint = Tuple[float, float, float]

SyntheticArrival = Dict[str, Union[GridPoint, float]]
SyntheticEventArrivals = Dict[str, Union[GridPoint, List[SyntheticArrival]]]


def metric_to_index(metric_point: MetricPoint, cell_size: float, shape: Tuple[int, int, int]) -> GridPoint:
    """
    Конвертация непрерывных метрических координат (от угла модели 0,0,0) в индексы сетки.
    Все дистанции должны быть положительными.
    """
    n_x, n_y, n_z = shape
    
    i = int(floor(metric_point[0] / cell_size))
    j = int(floor(metric_point[1] / cell_size))
    k = int(floor(metric_point[2] / cell_size))
    
    # Ограничиваем индексы границами сетки (на случай генерации точки ровно на границе)
    i = max(0, min(i, n_x - 1))
    j = max(0, min(j, n_y - 1))
    k = max(0, min(k, n_z - 1))
    
    return (i, j, k)


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
        depth_bias: float = 0.0,
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

    # Получаем метрические координаты и переведенные индексы
    metric_stations, station_idx = _resolve_station_locs_metric(
        shape, cell_size, station_locs, n_stations, rng
    )
    metric_events, event_idx = _resolve_event_locs_metric(
        shape, cell_size, event_locs, n_events, rng, depth_bias=depth_bias
    )

    fields = compute_station_travel_time_fields(
        grid=geo_grid,
        station_locs=station_idx,
        wave_type=wave_type,
        solver=solver
    ).astype(np.float64, copy=False)

    synthetic: List[SyntheticEventArrivals] = []
    
    for loc_idx, event_loc in enumerate(event_idx):
        i, j, k = event_loc
        arrivals_abs = fields[:, i, j, k]

        if not np.all(np.isfinite(arrivals_abs)):
            raise ValueError(
                f"Non-finite travel times for event {event_loc} "
                f"(metric: {metric_events[loc_idx]}). Check model/solver settings."
            )

        t_min = float(np.min(arrivals_abs))
        arrivals_rel = arrivals_abs - t_min

        event_arrivals: List[SyntheticArrival] = []
        for station_idx_i in range(len(station_idx)):
            event_arrivals.append(float(arrivals_rel[station_idx_i]))

        synthetic.append(event_arrivals)

    # Возвращаем изначально сгенерированные физические события в метрах
    return synthetic, metric_events


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
# Внутренние хелперы — генерация и маппинг метрических координат
# ---------------------------------------------------------------------------

def _sample_random_metric_points(
    shape: Tuple[int, int, int],
    cell_size: float,
    count: int,
    rng: np.random.Generator,
    fixed_z: Optional[float] = None,
    depth_bias: float = 0.0,
) -> Tuple[List[MetricPoint], List[GridPoint]]:
    """
    Генерирует случайные точки в физическом пространстве (метрах) от угла (0,0,0)
    и сразу прогоняет их через перевод в индексы.
    """
    n_x, n_y, n_z = shape
    max_x = n_x * cell_size
    max_y = n_y * cell_size
    max_z = n_z * cell_size

    metric_points = []
    grid_points = []
    seen_cells = set()

    max_attempts = count * 10
    attempts = 0

    while len(metric_points) < count and attempts < max_attempts:
        attempts += 1
        
        x = rng.uniform(0.0, max_x)
        y = rng.uniform(0.0, max_y)

        if fixed_z is not None:
            z = fixed_z
        else:
            if depth_bias == 0.0:
                z = rng.uniform(0.0, max_z)
            else:
                # Сэмплирование через обратную функцию распределения
                u = rng.uniform(0.0, 1.0)
                b = depth_bias
                z = (max_z / b) * np.log(1.0 + u * (np.exp(b) - 1.0))
                z = max(0.0, min(z, max_z))

        point = (float(x), float(y), float(z))
        idx = metric_to_index(point, cell_size, shape)

        if idx in seen_cells:
            continue

        seen_cells.add(idx)
        metric_points.append(point)
        grid_points.append(idx)

    if len(metric_points) < count:
        raise ValueError(f"Could not generate {count} unique locations. Grid might be too small.")

    return metric_points, grid_points


def _resolve_station_locs_metric(
    shape: Tuple[int, int, int],
    cell_size: float,
    station_locs: Optional[Sequence[MetricPoint]],
    n_stations: Optional[int],
    rng: np.random.Generator
) -> Tuple[List[MetricPoint], List[GridPoint]]:
    
    if station_locs is not None and n_stations is not None:
        raise ValueError("Provide either station_locs or n_stations, not both")

    if station_locs is not None:
        metric_stations = []
        grid_stations = []
        for idx, loc in enumerate(station_locs):
            if loc[2] != 0.0:
                raise ValueError(f"Station #{idx} must have z=0.0, got {loc}")
            metric_stations.append(tuple(float(c) for c in loc))
            grid_stations.append(metric_to_index(loc, cell_size, shape))
    else:
        if n_stations is None:
            raise ValueError("station_locs or n_stations must be provided")
        metric_stations, grid_stations = _sample_random_metric_points(
            shape, cell_size, int(n_stations), rng, fixed_z=0.0
        )

    if len(metric_stations) == 0:
        raise ValueError("At least one station is required")

    return metric_stations, grid_stations


def _resolve_event_locs_metric(
    shape: Tuple[int, int, int],
    cell_size: float,
    event_locs: Optional[Sequence[MetricPoint]],
    n_events: Optional[int],
    rng: np.random.Generator,
    depth_bias: float = 0.0,
) -> Tuple[List[MetricPoint], List[GridPoint]]:
    
    if event_locs is not None and n_events is not None:
        raise ValueError("Provide either event_locs or n_events, not both")

    if event_locs is not None:
        metric_events = []
        grid_events = []
        for loc in event_locs:
            metric_events.append(tuple(float(c) for c in loc))
            grid_events.append(metric_to_index(loc, cell_size, shape))
    else:
        if n_events is None:
            raise ValueError("event_locs or n_events must be provided")
        metric_events, grid_events = _sample_random_metric_points(
            shape, cell_size, int(n_events), rng, fixed_z=None, depth_bias=depth_bias
        )

    if len(metric_events) == 0:
        raise ValueError("At least one event is required")

    return metric_events, grid_events


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