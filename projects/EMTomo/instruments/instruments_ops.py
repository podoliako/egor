from __future__ import annotations

import numpy as np
from numba import njit

from interpolation import cell_centered_axis_weights


@njit(cache=True)
def _coarsen_G_trilinear_batch_nb(
    G_fine: np.ndarray,
    ix0: np.ndarray,
    ix1: np.ndarray,
    wx0: np.ndarray,
    wx1: np.ndarray,
    iy0: np.ndarray,
    iy1: np.ndarray,
    wy0: np.ndarray,
    wy1: np.ndarray,
    iz0: np.ndarray,
    iz1: np.ndarray,
    wz0: np.ndarray,
    wz1: np.ndarray,
    coarse_shape: tuple[int, int, int],
) -> np.ndarray:
    """Apply the adjoint trilinear operator while skipping empty ray cells."""
    n_stations, nx_f, ny_f, nz_f = G_fine.shape
    coarse_G = np.zeros((n_stations,) + coarse_shape, dtype=np.float64)

    for station in range(n_stations):
        for i in range(nx_f):
            ci0 = ix0[i]
            ci1 = ix1[i]
            wi0 = wx0[i]
            wi1 = wx1[i]
            for j in range(ny_f):
                cj0 = iy0[j]
                cj1 = iy1[j]
                wij00 = wi0 * wy0[j]
                wij01 = wi0 * wy1[j]
                wij10 = wi1 * wy0[j]
                wij11 = wi1 * wy1[j]
                for k in range(nz_f):
                    length = G_fine[station, i, j, k]
                    if length == 0.0:
                        continue
                    ck0 = iz0[k]
                    ck1 = iz1[k]
                    wk0 = wz0[k]
                    wk1 = wz1[k]
                    coarse_G[station, ci0, cj0, ck0] += length * wij00 * wk0
                    coarse_G[station, ci0, cj0, ck1] += length * wij00 * wk1
                    coarse_G[station, ci0, cj1, ck0] += length * wij01 * wk0
                    coarse_G[station, ci0, cj1, ck1] += length * wij01 * wk1
                    coarse_G[station, ci1, cj0, ck0] += length * wij10 * wk0
                    coarse_G[station, ci1, cj0, ck1] += length * wij10 * wk1
                    coarse_G[station, ci1, cj1, ck0] += length * wij11 * wk0
                    coarse_G[station, ci1, cj1, ck1] += length * wij11 * wk1

    return coarse_G


def _coarse_shape(fine_shape: tuple[int, int, int], subdivision: int) -> tuple[int, int, int]:
    if subdivision < 1:
        raise ValueError("subdivision must be >= 1")
    if any(size % subdivision != 0 for size in fine_shape):
        raise ValueError(
            f"Fine grid shape {fine_shape} not divisible by subdivision={subdivision}"
        )
    return tuple(size // subdivision for size in fine_shape)


def coarsen_G_all(
    G_fine: np.ndarray,
    subdivision: int,
    slowness_interpolation: str = "nearest",
) -> np.ndarray:
    """Restrict all station sensitivities from the fine grid in one batch."""
    G_fine = np.asarray(G_fine)
    if G_fine.ndim != 4:
        raise ValueError("G_fine must be a 4D array (n_stations, n_x, n_y, n_z)")
    if slowness_interpolation not in {"nearest", "trilinear"}:
        raise ValueError("slowness_interpolation must be 'nearest' or 'trilinear'")
    if subdivision == 1:
        return G_fine

    nx, ny, nz = _coarse_shape(G_fine.shape[1:], subdivision)
    if slowness_interpolation == "nearest":
        return G_fine.reshape(
            G_fine.shape[0],
            nx,
            subdivision,
            ny,
            subdivision,
            nz,
            subdivision,
        ).sum(axis=(2, 4, 6))

    ix0, ix1, wx0, wx1 = cell_centered_axis_weights(nx, subdivision)
    iy0, iy1, wy0, wy1 = cell_centered_axis_weights(ny, subdivision)
    iz0, iz1, wz0, wz1 = cell_centered_axis_weights(nz, subdivision)
    return _coarsen_G_trilinear_batch_nb(
        G_fine,
        ix0,
        ix1,
        wx0,
        wx1,
        iy0,
        iy1,
        wy0,
        wy1,
        iz0,
        iz1,
        wz0,
        wz1,
        (nx, ny, nz),
    )


def coarsen_G(
    G_fine: np.ndarray,
    subdivision: int,
    slowness_interpolation: str = "nearest",
) -> np.ndarray:
    G_fine = np.asarray(G_fine)
    if G_fine.ndim != 3:
        raise ValueError("G_fine must be a 3D array")
    return coarsen_G_all(
        G_fine[np.newaxis, ...],
        subdivision,
        slowness_interpolation=slowness_interpolation,
    )[0]
