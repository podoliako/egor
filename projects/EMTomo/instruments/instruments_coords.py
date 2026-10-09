from __future__ import annotations

import numpy as np


def metric_to_cell_coord(points_m, cell_size: float) -> np.ndarray:
    """Continuous cell-centred index coordinates: integer ``i`` is the centre of cell ``i``."""
    return np.asarray(points_m, dtype=np.float64) / cell_size - 0.5


def cell_coord_bounds(cell_index, shape):
    """Bounds for sub-cell refinement, limited to the field interpolation domain."""
    bounds = []
    for index, size in zip(cell_index, shape):
        if not 0 <= index < size:
            raise ValueError(f"cell_index {tuple(cell_index)} out of bounds for shape {shape}")
        bounds.append((max(0.0, index - 0.5), min(float(size - 1), index + 0.5)))
    return tuple(bounds)


def sample_cell_centered_trilinear_batch(fields: np.ndarray, cell_coord) -> np.ndarray:
    """Trilinear sampling of cell-centred fields shaped ``(..., nx, ny, nz)``."""
    if not isinstance(fields, np.ndarray) or fields.ndim < 3:
        raise ValueError("fields must have at least three dimensions")

    spatial_shape = fields.shape[-3:]
    coord = np.asarray(cell_coord, dtype=np.float64)
    if coord.shape != (3,) or not np.all(np.isfinite(coord)):
        raise ValueError("cell_coord must contain three finite coordinates")
    coord = np.clip(coord, 0.0, np.asarray(spatial_shape, dtype=np.float64) - 1.0)

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
