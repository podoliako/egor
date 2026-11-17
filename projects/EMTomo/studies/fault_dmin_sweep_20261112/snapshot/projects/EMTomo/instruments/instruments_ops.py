"""Tables that map fine-grid ray lengths onto inversion cells."""
from __future__ import annotations

import numpy as np

from interpolation import cell_centered_axis_weights


def _axis_table(coarse_size: int, subdivision: int, interpolation: str):
    if interpolation == "identity":
        index = np.arange(coarse_size * subdivision)
        weights = np.zeros((2, index.size))
        weights[0] = 1.0
        return np.stack([index, index]), weights
    if interpolation == "nearest":
        index = np.arange(coarse_size * subdivision) // subdivision
        weights = np.zeros((2, index.size))
        weights[0] = 1.0
        return np.stack([index, index]), weights
    if interpolation == "trilinear":
        lower, upper, lower_weight, upper_weight = cell_centered_axis_weights(coarse_size, subdivision)
        return np.stack([lower, upper]), np.stack([lower_weight, upper_weight])
    raise ValueError("interpolation must be 'nearest', 'trilinear' or 'identity'")


def restriction_tables(coarse_shape, subdivision: int, interpolation: str = "nearest"):
    """``(ix, wx, iy, wy, iz, wz)`` for ``raytracing._rasterize_nb``.

    ``nearest`` sums ray lengths of the subcells of each inversion cell;
    ``trilinear`` applies the adjoint of cell-centred trilinear prolongation of
    slowness, so that ``sum(G_coarse * s_coarse) == sum(G_fine * s_fine)``;
    ``identity`` keeps the fine grid (``coarse_shape`` is then the fine shape
    and ``subdivision`` must be 1).
    """
    if subdivision < 1:
        raise ValueError("subdivision must be >= 1")
    tables = []
    for size in coarse_shape:
        index, weights = _axis_table(int(size), subdivision, interpolation)
        tables += [np.ascontiguousarray(index, dtype=np.int64), np.ascontiguousarray(weights)]
    return tuple(tables)
