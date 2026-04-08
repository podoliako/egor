from __future__ import annotations

from typing import Optional

import numpy as np


def compute_cellwise_pairwise_misfit(
    station_fields: np.ndarray,
    observed_arrivals: np.ndarray,
) -> np.ndarray:
    if not isinstance(station_fields, np.ndarray) or station_fields.ndim != 4:
        raise ValueError("station_fields must be a 4D numpy array (n_stations, n_x, n_y, n_z)")
    if not isinstance(observed_arrivals, np.ndarray) or observed_arrivals.ndim != 1:
        raise ValueError("observed_arrivals must be a 1D numpy array")
    if observed_arrivals.shape[0] != station_fields.shape[0]:
        raise ValueError("observed_arrivals length must equal number of station fields")
    if observed_arrivals.shape[0] < 2:
        raise ValueError("At least 2 stations are required")

    n_stations = station_fields.shape[0]
    sum_r = np.zeros(station_fields.shape[1:], dtype=np.float64)
    sum_r2 = np.zeros(station_fields.shape[1:], dtype=np.float64)

    for i in range(n_stations):
        residual = observed_arrivals[i] - station_fields[i].astype(np.float64)
        sum_r += residual
        sum_r2 += np.square(residual, dtype=np.float64)

    misfit = n_stations * sum_r2 - np.square(sum_r, dtype=np.float64)
    return np.maximum(misfit, 0.0)


def compute_epicenter_weight_matrix(
    station_fields: np.ndarray,
    observed: np.ndarray,
    abs_misfit_threshold: Optional[float] = None,
    temperature: float = 1.0,
    return_misfit: bool = False,
):
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if abs_misfit_threshold is not None and abs_misfit_threshold < 0:
        raise ValueError("abs_misfit_threshold must be >= 0 when provided")

    misfit = compute_cellwise_pairwise_misfit(station_fields, observed)
    weights = _weights_from_misfit(misfit, abs_misfit_threshold, temperature)

    if return_misfit:
        return weights, misfit
    return weights


def _weights_from_misfit(
    misfit: np.ndarray,
    abs_misfit_threshold: Optional[float],
    temperature: float,
) -> np.ndarray:
    if abs_misfit_threshold is None:
        mask = np.ones(misfit.shape, dtype=bool)
    else:
        mask = np.abs(misfit) <= abs_misfit_threshold

    if not np.any(mask):
        weights = np.zeros_like(misfit, dtype=np.float64)
        best = tuple(int(v) for v in np.unravel_index(np.argmin(misfit), misfit.shape))
        weights[best] = 1.0
        return weights

    selected = misfit[mask]
    delta = selected - float(np.min(selected))
    positive = delta[delta > 0]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, 1e-12)

    raw = np.exp(-delta / (temperature * scale))
    raw_sum = float(np.sum(raw, dtype=np.float64))

    weights = np.zeros_like(misfit, dtype=np.float64)
    if raw_sum <= 0:
        best_flat = int(np.argmin(selected))
        true_idx = np.argwhere(mask)[best_flat]
        weights[tuple(int(v) for v in true_idx)] = 1.0
        return weights

    weights[mask] = raw / raw_sum
    total = float(np.sum(weights, dtype=np.float64))
    if total > 0 and not np.isclose(total, 1.0):
        weights /= total
    return weights
