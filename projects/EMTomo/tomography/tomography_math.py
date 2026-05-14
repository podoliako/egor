from __future__ import annotations

from typing import Tuple

import numpy as np

from raytracing import rasterize_path_lengths, trace_ray_from_timefield


def _calculate_residuals(station_fields: np.ndarray, arrivals: np.ndarray, weight_idx):
    x, y, z = weight_idx
    predicted = station_fields[:, x, y, z]
    residual_vector = arrivals - predicted
    return residual_vector[:, np.newaxis] - residual_vector[np.newaxis, :]


def _solve_delta_s(g_tilde_prime, r_prime, model_shape, lambda_reg, use_upper_triangle_pairs):
    n_st = g_tilde_prime.shape[0]
    pair_mask = (
        np.triu(np.ones((n_st, n_st), dtype=bool), k=1)
        if use_upper_triangle_pairs
        else ~np.eye(n_st, dtype=bool)
    )

    n_vox = int(np.prod(model_shape))
    g_rows = g_tilde_prime[pair_mask].reshape(-1, n_vox)
    r_vec  = r_prime[pair_mask].reshape(-1)

    if g_rows.shape[0] == 0:
        raise ValueError("No station pairs available")

    gtg = g_rows.T @ g_rows                    # (n_vox, n_vox)
    gtr = g_rows.T @ r_vec                     # (n_vox,)

    # Нормируем λ на средний диагональный элемент GᵀG.
    # lambda_reg=1.0  → регуляризация = data term (сильно)
    # lambda_reg=0.01 → 1% от data term (слабо)
    # lambda_reg=0.0  → нет регуляризации
    scale = np.trace(gtg) / n_vox
    gtg_reg = gtg + float(lambda_reg) * scale * np.eye(n_vox, dtype=np.float64)

    try:
        delta_s = np.linalg.solve(gtg_reg, gtr)
    except np.linalg.LinAlgError:
        delta_s = np.linalg.lstsq(gtg_reg, gtr, rcond=None)[0]

    return delta_s.reshape(model_shape)


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
