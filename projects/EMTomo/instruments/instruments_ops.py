from __future__ import annotations

import numpy as np

from interpolation import cell_centered_axis_weights


def coarsen_G(
    G_fine: np.ndarray,
    subdivision: int,
    slowness_interpolation: str = "nearest",
) -> np.ndarray:
    if slowness_interpolation not in {"nearest", "trilinear"}:
        raise ValueError("slowness_interpolation must be 'nearest' or 'trilinear'")
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

    if slowness_interpolation == "nearest":
        return (
            G_fine
            .reshape(nx, subdivision, ny, subdivision, nz, subdivision)
            .sum(axis=(1, 3, 5))
        )

    ix0, ix1, wx0, wx1 = cell_centered_axis_weights(nx, subdivision)
    iy0, iy1, wy0, wy1 = cell_centered_axis_weights(ny, subdivision)
    iz0, iz1, wz0, wz1 = cell_centered_axis_weights(nz, subdivision)
    coarse_G = np.zeros((nx, ny, nz), dtype=np.float64)

    for ix, wx in ((ix0, wx0), (ix1, wx1)):
        for iy, wy in ((iy0, wy0), (iy1, wy1)):
            for iz, wz in ((iz0, wz0), (iz1, wz1)):
                np.add.at(
                    coarse_G,
                    (ix[:, None, None], iy[None, :, None], iz[None, None, :]),
                    G_fine
                    * wx[:, None, None]
                    * wy[None, :, None]
                    * wz[None, None, :],
                )
    return coarse_G
