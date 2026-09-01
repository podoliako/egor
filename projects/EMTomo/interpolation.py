from __future__ import annotations

import numpy as np


def cell_centered_axis_weights(coarse_size: int, subdivision: int):
    """Return adjacent coarse indices and weights for fine-cell centres."""
    if coarse_size < 1:
        raise ValueError("coarse_size must be >= 1")
    if subdivision < 1:
        raise ValueError("subdivision must be >= 1")

    fine_index = np.arange(coarse_size * subdivision, dtype=np.float64)
    coarse_coord = (fine_index + 0.5) / subdivision - 0.5
    coarse_coord = np.clip(coarse_coord, 0.0, coarse_size - 1.0)
    lower = np.floor(coarse_coord).astype(np.intp)
    upper = np.minimum(lower + 1, coarse_size - 1)
    upper_weight = coarse_coord - lower
    lower_weight = 1.0 - upper_weight
    return lower, upper, lower_weight, upper_weight


def prolongate_cell_centered_trilinear(
    values: np.ndarray,
    subdivision: int,
) -> np.ndarray:
    """Cell-centred trilinear prolongation from a coarse 3-D grid."""
    coarse = np.asarray(values, dtype=np.float64)
    if coarse.ndim != 3:
        raise ValueError("values must be a 3-D array")

    ix0, ix1, wx0, wx1 = cell_centered_axis_weights(coarse.shape[0], subdivision)
    iy0, iy1, wy0, wy1 = cell_centered_axis_weights(coarse.shape[1], subdivision)
    iz0, iz1, wz0, wz1 = cell_centered_axis_weights(coarse.shape[2], subdivision)
    out = np.zeros(
        tuple(size * subdivision for size in coarse.shape), dtype=np.float64
    )

    for ix, wx in ((ix0, wx0), (ix1, wx1)):
        for iy, wy in ((iy0, wy0), (iy1, wy1)):
            for iz, wz in ((iz0, wz0), (iz1, wz1)):
                out += (
                    coarse[ix[:, None, None], iy[None, :, None], iz[None, None, :]]
                    * wx[:, None, None]
                    * wy[None, :, None]
                    * wz[None, None, :]
                )
    return out


def trilinear_interpolation(
    values: np.ndarray, i: int, j: int, k: int, di: float, dj: float, dk: float
) -> float:
    n_x, n_y, n_z = values.shape

    i1 = min(i + 1, n_x - 1)
    j1 = min(j + 1, n_y - 1)
    k1 = min(k + 1, n_z - 1)

    c000 = values[i, j, k]
    c100 = values[i1, j, k]
    c010 = values[i, j1, k]
    c110 = values[i1, j1, k]
    c001 = values[i, j, k1]
    c101 = values[i1, j, k1]
    c011 = values[i, j1, k1]
    c111 = values[i1, j1, k1]

    c00 = c000 * (1 - di) + c100 * di
    c01 = c001 * (1 - di) + c101 * di
    c10 = c010 * (1 - di) + c110 * di
    c11 = c011 * (1 - di) + c111 * di

    c0 = c00 * (1 - dj) + c10 * dj
    c1 = c01 * (1 - dj) + c11 * dj

    return c0 * (1 - dk) + c1 * dk


def nearest_neighbor_interpolation(
    values: np.ndarray, i: int, j: int, k: int, di: float, dj: float, dk: float
) -> float:
    return values[i, j, k]
