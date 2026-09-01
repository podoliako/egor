from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from instruments.instruments_coords import (
    cell_coord_bounds,
    sample_cell_centered_trilinear_batch,
)
from raytracing import rasterize_path_lengths, trace_ray_from_timefield


def _station_residuals_at_coord(
    station_fields: np.ndarray,
    arrivals: np.ndarray,
    cell_coord,
) -> np.ndarray:
    predicted = sample_cell_centered_trilinear_batch(station_fields, cell_coord)
    return np.asarray(arrivals, dtype=np.float64) - predicted


def _pairwise_misfit_from_station_residuals(residuals: np.ndarray) -> float:
    residuals = np.asarray(residuals, dtype=np.float64)
    n_stations = residuals.size
    value = n_stations * np.dot(residuals, residuals) - np.sum(residuals) ** 2
    return max(float(value), 0.0)


def _calculate_residuals(station_fields: np.ndarray, arrivals: np.ndarray, cell_coord):
    residual_vector = _station_residuals_at_coord(station_fields, arrivals, cell_coord)
    return residual_vector[:, np.newaxis] - residual_vector[np.newaxis, :]


def _refine_epicenter_in_cell(
    station_fields: np.ndarray,
    arrivals: np.ndarray,
    cell_index,
):
    """Refine one event hypothesis continuously, constrained to its cell."""
    start = np.asarray(cell_index, dtype=np.float64)
    bounds = cell_coord_bounds(tuple(int(v) for v in cell_index), station_fields.shape[1:])

    def objective(coord):
        residuals = _station_residuals_at_coord(station_fields, arrivals, coord)
        return _pairwise_misfit_from_station_residuals(residuals)

    start_misfit = objective(start)
    result = minimize(
        objective,
        start,
        method="Powell",
        bounds=bounds,
        options={"xtol": 1e-3, "ftol": 1e-8, "maxiter": 50},
    )
    refined = np.asarray(result.x, dtype=np.float64)
    refined_misfit = objective(refined)
    if not np.all(np.isfinite(refined)) or not np.isfinite(refined_misfit):
        return start, start_misfit
    if refined_misfit > start_misfit:
        return start, start_misfit
    return refined, refined_misfit


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


def _solve_delta_s(
    hessian,
    rhs,
    model_shape,
    lambda_reg,
    coverage_damping_power: float = 0.0,
    coverage_floor: float = 0.05,
    coverage_reference_percentile: float = 75.0,
    return_diagnostics: bool = False,
):
    """Solve the normal system with optional coverage-aware damping.

    ``diag(G.T @ W @ G)`` measures differential sensitivity, which is more
    informative than raw ray counts. When ``coverage_damping_power`` is
    positive, poorly constrained cells receive a stronger zero-update prior.
    No coupling between neighbouring cells is introduced.
    """
    n_vox = int(np.prod(model_shape))
    hessian = np.asarray(hessian, dtype=np.float64).reshape(n_vox, n_vox)
    rhs = np.asarray(rhs, dtype=np.float64).reshape(n_vox)
    if coverage_damping_power < 0.0:
        raise ValueError("coverage_damping_power must be >= 0")
    if not 0.0 < coverage_floor <= 1.0:
        raise ValueError("coverage_floor must be in (0, 1]")
    if not 0.0 <= coverage_reference_percentile <= 100.0:
        raise ValueError("coverage_reference_percentile must be in [0, 100]")

    sensitivity = np.maximum(np.diag(hessian), 0.0)
    positive = sensitivity[sensitivity > 0.0]
    reference = (
        float(np.percentile(positive, coverage_reference_percentile))
        if positive.size
        else 1.0
    )
    confidence = np.clip(sensitivity / max(reference, np.finfo(float).tiny), 0.0, 1.0)

    # lambda_reg is relative to the mean diagonal data sensitivity. With
    # power=0 this is exactly the previous uniform ridge regularization.
    scale = np.trace(hessian) / n_vox
    safe_confidence = np.maximum(confidence, coverage_floor)
    regularization_diagonal = (
        float(lambda_reg)
        * scale
        / np.power(safe_confidence, float(coverage_damping_power))
    )
    hessian_reg = hessian + np.diag(regularization_diagonal)

    try:
        delta_s = np.linalg.solve(hessian_reg, rhs)
    except np.linalg.LinAlgError:
        delta_s = np.linalg.lstsq(hessian_reg, rhs, rcond=None)[0]

    delta_s = delta_s.reshape(model_shape)
    if not return_diagnostics:
        return delta_s
    return delta_s, sensitivity.reshape(model_shape), confidence.reshape(model_shape)


def _select_top_n_cells_by_misfit(
    misfit: np.ndarray,
    n: int,
    min_distance: int = 2,
) -> np.ndarray:
    """Select separated low-misfit cells using Chebyshev index distance."""
    values = np.asarray(misfit, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("misfit must be a 3-D array")
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be an integer >= 1")
    if not isinstance(min_distance, (int, np.integer)) or min_distance < 1:
        raise ValueError("min_distance must be an integer >= 1")

    selected = []
    for flat_index in np.argsort(values, axis=None, kind="stable"):
        index = np.asarray(np.unravel_index(int(flat_index), values.shape), dtype=np.int64)
        if not np.isfinite(values[tuple(index)]):
            continue
        if any(np.max(np.abs(index - previous)) < min_distance for previous in selected):
            continue
        selected.append(index)
        if len(selected) == n:
            break

    if not selected:
        raise ValueError("No finite epicenter candidates available")
    return np.asarray(selected, dtype=np.int64)


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
