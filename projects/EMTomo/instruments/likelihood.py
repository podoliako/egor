"""Gaussian pick likelihood of hypocentre hypotheses with the origin time marginalized."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit


@dataclass(frozen=True)
class PickNoise:
    """Independent absolute-pick uncertainty for a station with predicted time ``T``:
    ``sigma² = (relative_sigma * T)² + absolute_sigma_s² + model_sigma_s²``."""

    relative_sigma: float = 0.0
    absolute_sigma_s: float = 0.0
    model_sigma_s: float = 0.2

    def __post_init__(self):
        for name in ("relative_sigma", "absolute_sigma_s", "model_sigma_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.absolute_sigma_s == 0 and self.model_sigma_s == 0:
            raise ValueError("absolute_sigma_s or model_sigma_s must be positive")

    @property
    def constant_variance(self) -> float:
        return self.absolute_sigma_s ** 2 + self.model_sigma_s ** 2

    def sigmas(self, predicted_times_s) -> np.ndarray:
        times = np.asarray(predicted_times_s, dtype=np.float64)
        return np.sqrt((self.relative_sigma * times) ** 2 + self.constant_variance)


@njit(cache=True)
def _half_chi2_kernel(predicted, observed, relative_sigma, constant_variance):
    n_stations, n_points = predicted.shape
    sum_p = np.zeros(n_points)
    sum_pr = np.zeros(n_points)
    sum_pr2 = np.zeros(n_points)
    for station in range(n_stations):
        observed_time = observed[station]
        for point in range(n_points):
            time = np.float64(predicted[station, point])
            precision = 1.0 / ((relative_sigma * time) ** 2 + constant_variance)
            residual = observed_time - time
            sum_p[point] += precision
            sum_pr[point] += precision * residual
            sum_pr2[point] += precision * residual * residual
    out = np.empty(n_points)
    for point in range(n_points):
        out[point] = 0.5 * max(sum_pr2[point] - sum_pr[point] ** 2 / sum_p[point], 0.0)
    return out


@njit(cache=True)
def _normalization_kernel(predicted, relative_sigma, constant_variance):
    n_stations, n_points = predicted.shape
    sum_p = np.zeros(n_points)
    sum_log_variance = np.zeros(n_points)
    for station in range(n_stations):
        for point in range(n_points):
            variance = (relative_sigma * np.float64(predicted[station, point])) ** 2 + constant_variance
            sum_p[point] += 1.0 / variance
            sum_log_variance[point] += np.log(variance)
    return 0.5 * sum_log_variance + 0.5 * np.log(sum_p)


def _station_major(predicted_times_s, n_observed=None):
    predicted = np.asarray(predicted_times_s)
    if predicted.ndim < 1 or predicted.shape[0] < 2:
        raise ValueError("Expected predicted times (n_stations >= 2, ...)")
    if n_observed is not None and n_observed != predicted.shape[0]:
        raise ValueError("Expected one observation per station")
    flat = np.ascontiguousarray(predicted.reshape(predicted.shape[0], -1))
    if flat.dtype not in (np.float32, np.float64):
        flat = flat.astype(np.float64)
    return flat, predicted.shape[1:]


def likelihood_normalization(predicted_times_s, noise: PickNoise):
    """Observation-independent part of ``-log L``: ``sum(log sigma) + log(sum precision) / 2``.

    It is constant when ``relative_sigma == 0`` (returned as 0.0) and otherwise
    depends only on the predicted times, so it can be reused for every event.
    """
    if noise.relative_sigma == 0:
        return 0.0
    flat, shape = _station_major(predicted_times_s)
    values = _normalization_kernel(flat, float(noise.relative_sigma), float(noise.constant_variance))
    return values.reshape(shape)


def negative_log_likelihood(predicted_times_s, observed_s, noise: PickNoise, normalization=None):
    """``-log L`` of each hypothesis, up to a constant shared by all hypotheses.

    ``predicted_times_s`` is station-major, ``(n_stations, ...)``. Observed times
    are relative to an unknown common origin; it is integrated out under a flat
    prior, which yields the precision-weighted chi² about the weighted mean
    residual plus ``likelihood_normalization`` (pass it to reuse a precomputed one).
    """
    observed = np.asarray(observed_s, dtype=np.float64)
    if observed.ndim != 1 or not np.all(np.isfinite(observed)):
        raise ValueError("Observed times must be a finite vector")
    flat, shape = _station_major(predicted_times_s, observed.size)
    half_chi2 = _half_chi2_kernel(
        flat, observed, float(noise.relative_sigma), float(noise.constant_variance)
    ).reshape(shape)
    if normalization is None:
        normalization = likelihood_normalization(predicted_times_s, noise)
    return half_chi2 + normalization


def candidate_weights(negative_log_likelihoods, temperature: float = 1.0) -> np.ndarray:
    """Normalized posterior weights for equally probable a priori hypotheses.

    ``temperature=1`` is the likelihood itself; other values temper it.
    """
    values = np.asarray(negative_log_likelihoods, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Expected a non-empty vector of finite values")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    weights = np.exp(-(values - values.min()) / temperature)
    return weights / weights.sum()
