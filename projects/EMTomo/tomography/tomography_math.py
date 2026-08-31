from __future__ import annotations

from typing import Tuple

import numpy as np

from raytracing import rasterize_path_lengths, trace_ray_from_timefield


def _calculate_residuals(station_fields: np.ndarray, arrivals: np.ndarray, weight_idx):
    x, y, z = weight_idx
    predicted = station_fields[:, x, y, z]
    residual_vector = arrivals - predicted
    return residual_vector[:, np.newaxis] - residual_vector[np.newaxis, :]


def _normal_equation_contribution(
    station_sensitivities: np.ndarray,
    station_residuals: np.ndarray,
    model_shape,
    weight: float,
    valid_stations: np.ndarray,
):
    """Return ``w GᵀG`` and ``w Gᵀr`` for all valid station pairs.

    For station rows ``X_i`` and residuals ``q_i``, the identity
    ``sum_{i<j}(X_i-X_j)ᵀ(X_i-X_j) = n X_cᵀX_c`` avoids explicitly
    constructing the quadratic number of pair rows.
    """
    n_vox = int(np.prod(model_shape))
    valid = np.asarray(valid_stations, dtype=bool)
    if np.count_nonzero(valid) < 2 or weight <= 0.0:
        return (
            np.zeros((n_vox, n_vox), dtype=np.float64),
            np.zeros(n_vox, dtype=np.float64),
        )

    rows = station_sensitivities[valid].reshape(-1, n_vox)
    residual = np.asarray(station_residuals, dtype=np.float64)[valid]
    rows_centered = rows - np.mean(rows, axis=0, keepdims=True)
    residual_centered = residual - np.mean(residual)
    n_valid = rows.shape[0]

    return (
        weight * n_valid * (rows_centered.T @ rows_centered),
        weight * n_valid * (rows_centered.T @ residual_centered),
    )


def _solve_delta_s(hessian, rhs, model_shape, lambda_reg):
    """Solve an already accumulated normal system for the slowness update."""
    n_vox = int(np.prod(model_shape))
    hessian = np.asarray(hessian, dtype=np.float64).reshape(n_vox, n_vox)
    rhs = np.asarray(rhs, dtype=np.float64).reshape(n_vox)

    # Нормируем λ на средний диагональный элемент GᵀG.
    # lambda_reg=1.0  → регуляризация = data term (сильно)
    # lambda_reg=0.01 → 1% от data term (слабо)
    # lambda_reg=0.0  → нет регуляризации
    scale = np.trace(hessian) / n_vox
    hessian_reg = hessian + float(lambda_reg) * scale * np.eye(n_vox, dtype=np.float64)

    try:
        delta_s = np.linalg.solve(hessian_reg, rhs)
    except np.linalg.LinAlgError:
        delta_s = np.linalg.lstsq(hessian_reg, rhs, rcond=None)[0]

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
    path, reached = trace_ray_from_timefield(
        T=station_field,
        station_xyz=station_loc,
        epic_xyz=origin_loc,
        spacing_xyz=(1.0, 1.0, 1.0),
        gradT=gradT,
        return_status=True,
    )
    if not reached:
        return np.zeros(geo_shape, dtype=np.float64)
    return rasterize_path_lengths(
        path_xyz=path, shape=geo_shape, voxel_size=voxel_size, dtype=np.float64
    )
