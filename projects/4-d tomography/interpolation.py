from __future__ import annotations

import numpy as np


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
