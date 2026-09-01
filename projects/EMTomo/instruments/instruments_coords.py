from __future__ import annotations

from math import floor
from typing import List, Optional, Tuple

import numpy as np

CellIndex = Tuple[int, int, int]
CellCoord = Tuple[float, float, float]
GridPoint = CellIndex
MetricPoint = Tuple[float, float, float]


def metric_to_cell_coord(
    metric_point: MetricPoint,
    cell_size: float,
    shape: Tuple[int, int, int],
) -> CellCoord:
    """Convert a metric point to continuous cell-centred array coordinates.

    Array index ``i`` represents the centre of the cell spanning
    ``[i * cell_size, (i + 1) * cell_size)``. Values outside the interpolation
    domain of cell centres are clamped to its nearest boundary centre.
    """
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")
    if any(c < 0 for c in metric_point):
        raise ValueError("metric_point coordinates must be non-negative")

    shape_array = np.asarray(shape, dtype=np.float64)
    metric = np.asarray(metric_point, dtype=np.float64)
    if metric.shape != (3,):
        raise ValueError("metric_point must contain exactly three coordinates")

    cell_coord = metric / cell_size - 0.5
    cell_coord = np.clip(cell_coord, 0.0, shape_array - 1.0)
    return tuple(float(value) for value in cell_coord)


def cell_coord_to_metric(cell_coord: CellCoord, cell_size: float) -> MetricPoint:
    """Convert a continuous cell-centred coordinate to metric coordinates."""
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")
    coord = np.asarray(cell_coord, dtype=np.float64)
    if coord.shape != (3,) or not np.all(np.isfinite(coord)):
        raise ValueError("cell_coord must contain three finite coordinates")
    return tuple(float((value + 0.5) * cell_size) for value in coord)


def cell_index_to_metric_center(cell_index: CellIndex, cell_size: float) -> MetricPoint:
    """Return the metric centre of an integer cell index."""
    return cell_coord_to_metric(tuple(float(value) for value in cell_index), cell_size)


def cell_coord_bounds(
    cell_index: CellIndex,
    shape: Tuple[int, int, int],
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """Bounds for sub-cell refinement, limited to the field interpolation domain."""
    bounds = []
    for index, size in zip(cell_index, shape):
        if not 0 <= index < size:
            raise ValueError(f"cell_index {cell_index} out of bounds for shape {shape}")
        bounds.append((max(0.0, index - 0.5), min(float(size - 1), index + 0.5)))
    return tuple(bounds)  # type: ignore[return-value]


def sample_cell_centered_trilinear(
    field: np.ndarray,
    cell_coord: Tuple[float, float, float],
) -> float:
    """Sample a 3-D cell-centred field at a continuous cell coordinate."""
    if not isinstance(field, np.ndarray) or field.ndim != 3:
        raise ValueError("field must be a 3-D numpy array")

    shape = np.asarray(field.shape, dtype=np.float64)
    coord = np.asarray(cell_coord, dtype=np.float64)
    if coord.shape != (3,):
        raise ValueError("cell_coord must contain exactly three coordinates")
    coord = np.clip(coord, 0.0, shape - 1.0)

    i0, j0, k0 = np.floor(coord).astype(np.int64)
    i1 = min(i0 + 1, field.shape[0] - 1)
    j1 = min(j0 + 1, field.shape[1] - 1)
    k1 = min(k0 + 1, field.shape[2] - 1)
    di, dj, dk = coord - np.array((i0, j0, k0), dtype=np.float64)

    c00 = field[i0, j0, k0] * (1.0 - di) + field[i1, j0, k0] * di
    c10 = field[i0, j1, k0] * (1.0 - di) + field[i1, j1, k0] * di
    c01 = field[i0, j0, k1] * (1.0 - di) + field[i1, j0, k1] * di
    c11 = field[i0, j1, k1] * (1.0 - di) + field[i1, j1, k1] * di
    c0 = c00 * (1.0 - dj) + c10 * dj
    c1 = c01 * (1.0 - dj) + c11 * dj
    return float(c0 * (1.0 - dk) + c1 * dk)


def sample_cell_centered_trilinear_batch(
    fields: np.ndarray,
    cell_coord: CellCoord,
) -> np.ndarray:
    """Vectorized trilinear sampling of fields shaped ``(..., nx, ny, nz)``."""
    if not isinstance(fields, np.ndarray) or fields.ndim < 3:
        raise ValueError("fields must have at least three dimensions")

    spatial_shape = fields.shape[-3:]
    shape = np.asarray(spatial_shape, dtype=np.float64)
    coord = np.asarray(cell_coord, dtype=np.float64)
    if coord.shape != (3,) or not np.all(np.isfinite(coord)):
        raise ValueError("cell_coord must contain three finite coordinates")
    coord = np.clip(coord, 0.0, shape - 1.0)

    i0, j0, k0 = np.floor(coord).astype(np.int64)
    i1 = min(i0 + 1, spatial_shape[0] - 1)
    j1 = min(j0 + 1, spatial_shape[1] - 1)
    k1 = min(k0 + 1, spatial_shape[2] - 1)
    di, dj, dk = coord - np.array((i0, j0, k0), dtype=np.float64)

    c00 = fields[..., i0, j0, k0] * (1.0 - di) + fields[..., i1, j0, k0] * di
    c10 = fields[..., i0, j1, k0] * (1.0 - di) + fields[..., i1, j1, k0] * di
    c01 = fields[..., i0, j0, k1] * (1.0 - di) + fields[..., i1, j0, k1] * di
    c11 = fields[..., i0, j1, k1] * (1.0 - di) + fields[..., i1, j1, k1] * di
    c0 = c00 * (1.0 - dj) + c10 * dj
    c1 = c01 * (1.0 - dj) + c11 * dj
    return np.asarray(c0 * (1.0 - dk) + c1 * dk, dtype=np.float64)


def metric_to_cell_index(
    metric_point: MetricPoint,
    cell_size: float,
    shape: Tuple[int, int, int],
) -> CellIndex:
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


def metric_to_index(metric_point: MetricPoint, cell_size: float, shape: Tuple[int, int, int]) -> GridPoint:
    """Backward-compatible alias for :func:`metric_to_cell_index`."""
    return metric_to_cell_index(metric_point, cell_size, shape)


def snap_metric_points_to_cell_centers(
    metric_points: List[MetricPoint],
    cell_size: float,
    shape: Tuple[int, int, int],
) -> List[MetricPoint]:
    """Snap metric points to centres of their owning grid cells."""
    centers: List[MetricPoint] = []
    seen_cells = set()
    for point in metric_points:
        index = metric_to_cell_index(point, cell_size, shape)
        if index in seen_cells:
            raise ValueError("Multiple points map to the same grid cell")
        seen_cells.add(index)
        centers.append(
            tuple((axis + 0.5) * cell_size for axis in index)
        )
    return centers


def _sample_random_metric_points(
    shape: Tuple[int, int, int],
    cell_size: float,
    count: int,
    rng: np.random.Generator,
    fixed_z: Optional[float] = None,
    depth_bias: float = 0.0,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    z_offset: float = 0.0,
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
    if not 0.0 <= z_offset < max_z:
        raise ValueError("z_offset must be within the grid depth")
    if fixed_z is not None and not z_offset <= fixed_z < max_z:
        raise ValueError("fixed_z must be within [z_offset, grid depth)")

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
                z = rng.uniform(z_offset, max_z)
            else:
                u = rng.uniform(0.0, 1.0)
                b = depth_bias
                depth_range = max_z - z_offset
                z = z_offset + (depth_range / b) * np.log(1.0 + u * (np.exp(b) - 1.0))
                z = max(z_offset, min(z, max_z))

        point = (float(x), float(y), float(z))
        idx = metric_to_cell_index(point, cell_size, shape)

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
            grid_stations.append(metric_to_cell_index(loc, cell_size, shape))
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
    z_offset: float = 0.0,
) -> Tuple[List[MetricPoint], List[GridPoint]]:
    if event_locs is not None and n_events is not None:
        raise ValueError("Provide either event_locs or n_events, not both")

    if event_locs is not None:
        metric_events = []
        grid_events = []
        for loc in event_locs:
            metric_events.append(tuple(float(c) for c in loc))
            grid_events.append(metric_to_cell_index(loc, cell_size, shape))
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
            z_offset=z_offset,
        )

    if len(metric_events) == 0:
        raise ValueError("At least one event is required")

    return metric_events, grid_events
