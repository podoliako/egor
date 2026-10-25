"""Fused, allocation-free interpolation/likelihood for sub-cell refinement."""
from __future__ import annotations

import numpy as np
from numba import njit

from instruments.likelihood import PickNoise


@njit(cache=True, inline="always")
def _lerp(a, b, weight, float32_intermediates):
    if float32_intermediates:
        # NumPy 1.x casts the float64 scalar weights to the float32 array dtype.
        # Each multiply and add is a separate ufunc, hence a separate rounding.
        left = np.float32(np.float32(a) * np.float32(1.0 - weight))
        right = np.float32(np.float32(b) * np.float32(weight))
        return np.float64(np.float32(left + right))
    return np.float64(a) * (1.0 - weight) + np.float64(b) * weight


@njit(cache=True)
def _refinement_objective_kernel(
    fields, observed, coord, relative_sigma, constant_variance, float32_intermediates,
):
    for axis in range(3):
        if not np.isfinite(coord[axis]):
            raise ValueError("cell_coord must contain three finite coordinates")
    x = min(max(coord[0], 0.0), float(fields.shape[1] - 1))
    y = min(max(coord[1], 0.0), float(fields.shape[2] - 1))
    z = min(max(coord[2], 0.0), float(fields.shape[3] - 1))
    i0, j0, k0 = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
    i1 = min(i0 + 1, fields.shape[1] - 1)
    j1 = min(j0 + 1, fields.shape[2] - 1)
    k1 = min(k0 + 1, fields.shape[3] - 1)
    di, dj, dk = x - i0, y - j0, z - k0
    sum_p = 0.0
    sum_pr = 0.0
    sum_pr2 = 0.0
    sum_log_variance = 0.0
    for station in range(fields.shape[0]):
        c00 = _lerp(fields[station, i0, j0, k0], fields[station, i1, j0, k0], di, float32_intermediates)
        c10 = _lerp(fields[station, i0, j1, k0], fields[station, i1, j1, k0], di, float32_intermediates)
        c01 = _lerp(fields[station, i0, j0, k1], fields[station, i1, j0, k1], di, float32_intermediates)
        c11 = _lerp(fields[station, i0, j1, k1], fields[station, i1, j1, k1], di, float32_intermediates)
        c0 = _lerp(c00, c10, dj, float32_intermediates)
        c1 = _lerp(c01, c11, dj, float32_intermediates)
        time = _lerp(c0, c1, dk, float32_intermediates)
        variance = (relative_sigma * time) ** 2 + constant_variance
        precision = 1.0 / variance
        residual = observed[station] - time
        sum_p += precision
        sum_pr += precision * residual
        sum_pr2 += precision * residual * residual
        if relative_sigma != 0.0:
            sum_log_variance += np.log(variance)
    value = 0.5 * max(sum_pr2 - sum_pr ** 2 / sum_p, 0.0)
    if relative_sigma != 0.0:
        value += 0.5 * sum_log_variance + 0.5 * np.log(sum_p)
    return value


def make_refinement_objective(station_fields, observed, noise: PickNoise):
    """Return ``objective(cell_coord)`` for the existing bounded Powell search.

    Static inputs are validated once. Fields must be float32/64 ``(S, nx, ny,
    nz)``, with S >= 2 and nonempty spatial axes; strided arrays are supported
    without copying the field volume. Like the existing likelihood, field values
    are not scanned for finiteness. Observations must be a finite vector of S
    picks. The callback retains its input arrays; do not mutate them during use.

    The compiled path preserves coordinate clipping, staged x/y/z interpolation,
    NumPy's intermediate dtype promotion, station summation order, and the
    relative-noise normalization. No fastmath or optimizer settings are used.
    The first call compiles the relevant Numba specialization (call once before
    timing to warm up); subsequent calls allocate no station-sized temporaries.
    """
    fields = np.asarray(station_fields)
    if fields.ndim != 4 or fields.shape[0] < 2 or any(size == 0 for size in fields.shape[1:]):
        raise ValueError("station_fields must have shape (stations >= 2, nx > 0, ny > 0, nz > 0)")
    if fields.dtype not in (np.float32, np.float64):
        raise ValueError("station_fields must have float32 or float64 dtype")
    observed = np.asarray(observed, dtype=np.float64)
    if observed.ndim != 1 or observed.size != fields.shape[0] or not np.all(np.isfinite(observed)):
        raise ValueError("Expected one finite observation per station")
    relative_sigma = float(noise.relative_sigma)
    constant_variance = float(noise.constant_variance)
    # Probe the actual array/scalar promotion rule, rather than assuming a NumPy
    # version: 1.x keeps float32 intermediates, whereas 2.x promotes to float64.
    float32_intermediates = (
        fields.dtype == np.float32
        and (np.ones(1, dtype=np.float32) * np.float64(0.5)).dtype == np.float32
    )

    def objective(cell_coord):
        coord = np.asarray(cell_coord, dtype=np.float64)
        if coord.shape != (3,):
            raise ValueError("cell_coord must contain three finite coordinates")
        return _refinement_objective_kernel(
            fields, observed, coord, relative_sigma, constant_variance, float32_intermediates,
        )

    return objective
