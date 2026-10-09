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

