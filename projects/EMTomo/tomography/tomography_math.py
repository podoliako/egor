from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from instruments.instruments_coords import cell_coord_bounds, sample_cell_centered_trilinear_batch
from instruments.likelihood import PickNoise, negative_log_likelihood


def select_candidate_cells(cost: np.ndarray, n: int, min_distance: int = 1) -> np.ndarray:
    """Lowest-cost cells, pairwise separated by at least ``min_distance`` (Chebyshev, in cells)."""
    values = np.asarray(cost, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("cost must be a 3-D array")
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be an integer >= 1")
    if not isinstance(min_distance, (int, np.integer)) or min_distance < 1:
        raise ValueError("min_distance must be an integer >= 1")

    flat = values.ravel()
    # Each selected cell can exclude at most (2d - 1)^3 better-ranked cells.
    n_considered = min(flat.size, n * (2 * min_distance - 1) ** 3 + n)
    while True:
        if n_considered < flat.size:
            order = np.argpartition(flat, n_considered - 1)[:n_considered]
            order = order[np.argsort(flat[order], kind="stable")]
        else:
            order = np.argsort(flat, kind="stable")
        selected = []
        for flat_index in order:
            if not np.isfinite(flat[flat_index]):
                continue
            index = np.asarray(np.unravel_index(int(flat_index), values.shape), dtype=np.int64)
            if any(np.max(np.abs(index - previous)) < min_distance for previous in selected):
                continue
            selected.append(index)
            if len(selected) == n:
                return np.asarray(selected, dtype=np.int64)
        if n_considered >= flat.size:
            break
        n_considered = flat.size
    if not selected:
        raise ValueError("No finite hypocentre candidates available")
    return np.asarray(selected, dtype=np.int64)


def refine_hypocentre_in_cell(
    station_fields: np.ndarray,
    observed: np.ndarray,
    cell_index,
    noise: PickNoise,
) -> tuple[np.ndarray, float]:
    """Continuous maximum-likelihood position of one hypothesis within its cell."""
    start = np.asarray(cell_index, dtype=np.float64)
    bounds = cell_coord_bounds(tuple(int(v) for v in cell_index), station_fields.shape[1:])

    def objective(coord):
        predicted = sample_cell_centered_trilinear_batch(station_fields, coord)
        return float(negative_log_likelihood(predicted, observed, noise))

    start_value = objective(start)
    result = minimize(
        objective,
        start,
        method="Powell",
        bounds=bounds,
        options={"xtol": 1e-3, "ftol": 1e-8, "maxiter": 50},
    )
    refined = np.asarray(result.x, dtype=np.float64)
    if not np.all(np.isfinite(refined)):
        return start, start_value
    refined_value = objective(refined)
    if not np.isfinite(refined_value) or refined_value > start_value:
        return start, start_value
    return refined, refined_value


def accumulate_normal_equations(
    hessian: np.ndarray,
    rhs: np.ndarray,
    station_sensitivities: np.ndarray,
    station_residuals: np.ndarray,
    valid_stations: np.ndarray,
    station_sigmas: np.ndarray,
    weight: float,
) -> None:
    """Add one weighted hypothesis to the normal equations in place.

    The unknown origin time is profiled out by centring station rows and
    residuals with precision weights ``1 / sigma_i²``; this equals station-pair
    differences weighted by ``p_i * p_j / sum(p)``.
    """
    valid = np.asarray(valid_stations, dtype=bool)
    if np.count_nonzero(valid) < 2 or weight <= 0.0:
        return
    n_vox = rhs.size
    rows = np.asarray(station_sensitivities)[valid].reshape(-1, n_vox)
    residual = np.asarray(station_residuals, dtype=np.float64)[valid]
    sigmas = np.asarray(station_sigmas, dtype=np.float64)
    if sigmas.shape != valid.shape:
        raise ValueError("station_sigmas must match valid_stations shape")
    sigmas = sigmas[valid]
    if not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0):
        raise ValueError("Valid station sigmas must be finite and positive")
    precision = 1.0 / np.square(sigmas)

    # Rays touch only a few cells and centring cannot fill an all-zero column,
    # so the Gram matrix is formed on the active columns only.
    active = np.flatnonzero(np.any(rows != 0.0, axis=0))
    if active.size == 0:
        return
    rows_centered = rows[:, active] - np.average(rows[:, active], axis=0, weights=precision)
    residual_centered = residual - np.average(residual, weights=precision)
    hessian[np.ix_(active, active)] += weight * (rows_centered.T @ (precision[:, None] * rows_centered))
    rhs[active] += weight * (rows_centered.T @ (precision * residual_centered))


def solve_slowness_update(
    hessian,
    rhs,
    model_shape,
    lambda_reg: float,
    coverage_damping_power: float = 0.0,
    coverage_floor: float = 0.05,
    coverage_reference_percentile: float = 75.0,
):
    """Solve the damped normal system for the slowness increment.

    Damping acts on the increment, not on the total model. ``lambda_reg`` is
    relative to the mean of ``diag(H)``; with ``coverage_damping_power > 0``
    cells with low ``diag(H)`` (relative to its ``coverage_reference_percentile``)
    are damped more strongly.

    Returns ``(delta_s, sensitivity_diagonal, coverage_confidence)``.
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
    reference = float(np.percentile(positive, coverage_reference_percentile)) if positive.size else 1.0
    confidence = np.clip(sensitivity / max(reference, np.finfo(float).tiny), 0.0, 1.0)

    scale = max(float(np.trace(hessian)) / n_vox, 0.0)
    damping = lambda_reg * scale / np.power(np.maximum(confidence, coverage_floor), coverage_damping_power)
    system = hessian + np.diag(damping)
    try:
        delta_s = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        delta_s = np.linalg.lstsq(system, rhs, rcond=None)[0]
    return (
        delta_s.reshape(model_shape),
        sensitivity.reshape(model_shape),
        confidence.reshape(model_shape),
    )
