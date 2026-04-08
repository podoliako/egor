from __future__ import annotations

import numpy as np


def coarsen_G(G_fine: np.ndarray, subdivision: int) -> np.ndarray:
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
