from __future__ import annotations

from math import floor
from typing import List, Optional, Tuple

import numpy as np

GridPoint = Tuple[int, int, int]
MetricPoint = Tuple[float, float, float]


def metric_to_index(metric_point: MetricPoint, cell_size: float, shape: Tuple[int, int, int]) -> GridPoint:
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")
    if any(c < 0 for c in metric_point):
        raise ValueError("metric_point coordinates must be non-negative")

    n_x, n_y, n_z = shape

    i = int(floor(metric_point[0] / cell_size))
    j = int(floor(metric_point[1] / cell_size))
    k = int(floor(metric_point[2] / cell_size))

    i = max(0, min(i, n_x - 1))
    j = max(0, min(j, n_y - 1))
    k = max(0, min(k, n_z - 1))

    return (i, j, k)


def _sample_random_metric_points(
    shape: Tuple[int, int, int],
    cell_size: float,
    count: int,
    rng: np.random.Generator,
    fixed_z: Optional[float] = None,
    depth_bias: float = 0.0,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
) -> Tuple[List[MetricPoint], List[GridPoint]]:
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")
    if count <= 0:
        raise ValueError("count must be > 0")
    if depth_bias < 0:
        raise ValueError("depth_bias must be >= 0")

    n_x, n_y, n_z = shape
    max_x = n_x * cell_size
    max_y = n_y * cell_size
    max_z = n_z * cell_size

    if x_offset * 2 >= max_x or y_offset * 2 >= max_y:
        raise ValueError("x_offset/y_offset too large for grid size")

    metric_points = []
    grid_points = []
    seen_cells = set()

    max_attempts = count * 10
    attempts = 0

    while len(metric_points) < count and attempts < max_attempts:
        attempts += 1

        x = rng.uniform(0.0 + x_offset, max_x - x_offset)
        y = rng.uniform(0.0 + y_offset, max_y - y_offset)

        if fixed_z is not None:
            z = fixed_z
        else:
            if depth_bias == 0.0:
                z = rng.uniform(0.0, max_z)
            else:
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
    station_locs: Optional[List[MetricPoint]],
    n_stations: Optional[int],
    rng: np.random.Generator,
) -> Tuple[List[MetricPoint], List[GridPoint]]:
    if station_locs is not None and n_stations is not None:
        raise ValueError("Provide either station_locs or n_stations, not both")

    if station_locs is not None:
        metric_stations = []
        grid_stations = []
        for loc in station_locs:
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
    event_locs: Optional[List[MetricPoint]],
    n_events: Optional[int],
    rng: np.random.Generator,
    depth_bias: float = 0.0,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
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
            shape,
            cell_size,
            int(n_events),
            rng,
            fixed_z=None,
            depth_bias=depth_bias,
            x_offset=x_offset,
            y_offset=y_offset,
        )

    if len(metric_events) == 0:
        raise ValueError("At least one event is required")

    return metric_events, grid_events
