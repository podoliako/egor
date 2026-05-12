from __future__ import annotations

from typing import Tuple

import numpy as np

from raytracing import rasterize_path_lengths, trace_ray_from_timefield


def _calculate_residuals(station_fields: np.ndarray, arrivals: np.ndarray, weight_idx):
    x, y, z = weight_idx
    predicted = station_fields[:, x, y, z]
    residual_vector = arrivals - predicted
    return residual_vector[:, np.newaxis] - residual_vector[np.newaxis, :]


def _solve_delta_s(
    g_tilde_prime: np.ndarray,
    r_prime: np.ndarray,
    model_shape: Tuple[int, int, int],
    lambda_reg: float,
    use_upper_triangle_pairs: bool,
):
    g_tilde_prime = np.asarray(g_tilde_prime)
    n_st = g_tilde_prime.shape[0]
    if n_st != g_tilde_prime.shape[1]:
        raise ValueError("g_tilde_prime must have shape (n_stations, n_stations, ...)")
    if r_prime.shape != (n_st, n_st):
        raise ValueError("r_prime shape must be (n_stations, n_stations)")

    pair_mask = (
        np.triu(np.ones((n_st, n_st), dtype=bool), k=1)
        if use_upper_triangle_pairs
        else ~np.eye(n_st, dtype=bool)
    )

    g_rows = g_tilde_prime[pair_mask].reshape(-1, int(np.prod(model_shape)))
    r_vec = r_prime[pair_mask].reshape(-1)
    if g_rows.shape[0] == 0:
        raise ValueError("No station pairs available to solve tomography system")

    gg_t = g_rows @ g_rows.T
    gg_t_reg = gg_t + float(lambda_reg) * np.eye(gg_t.shape[0], dtype=np.float64)

    try:
        alpha = np.linalg.solve(gg_t_reg, r_vec)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(gg_t_reg, r_vec, rcond=None)[0]

    return (g_rows.T @ alpha).reshape(model_shape)


def _select_top_n_weights(weights_matrix, n: int, normalize: bool = False):
    w = np.asarray(weights_matrix, dtype=np.float64)
    if w.ndim != 3:
        raise ValueError("weights_matrix must be a 3-D array")
    if not isinstance(n, (int, np.integer)):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("n must be >= 0")

    out = np.zeros_like(w)
    if n == 0:
        return out
    if n >= w.size:
        out = w.copy()
    else:
        flat = w.ravel()
        top_idx = np.argpartition(flat, -n)[-n:]
        out.ravel()[top_idx] = flat[top_idx]

    if normalize:
        s = out.sum()
        if s > 0:
            out /= s
    return out


def _calculate_G(station_field, origin_loc, station_loc, geo_shape, voxel_size, gradT=None):
    path = trace_ray_from_timefield(
        T=station_field,
        station_xyz=station_loc,
        epic_xyz=origin_loc,
        spacing_xyz=(1.0, 1.0, 1.0),
        gradT=gradT,
    )
    return rasterize_path_lengths(
        path_xyz=path, shape=geo_shape, voxel_size=voxel_size, dtype=np.float64
    )
