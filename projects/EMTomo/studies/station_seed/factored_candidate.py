"""Experimental monotone factored Godunov solver; not a production backend.

T = distance * factor. First-order upwinding is applied to the smooth factor,
not to the point-source singularity. All directional affine branches are kept
in the Godunov maximum, avoiding second-order stencil switches at time ties.

The numerical grid includes the physical domain faces. This is important: a
source on a face is then inside the nodal domain, rather than half a cell beyond
it. Missing directions use the state-constraint (no incoming neighbor) closure.
Cell-centred output interpolates the factor, then restores the exact distance.

The homogeneous seed sphere and source-cell parameter remain an approximation.
Node slownesses outside the seed average neighboring fine-cell slownesses;
consequently this is not the same discrete forward operator as scikit-fmm.
"""
from __future__ import annotations

import time

import numpy as np
from numba import njit


@njit(cache=True)
def _axis(u, al, bl, ar, br):
    left, right = al * u - bl, ar * u - br
    # At a tie use the right derivative of the convex maximum.
    if left > right or (left == right and al >= ar):
        value, slope = left, al
    else:
        value, slope = right, ar
    if value <= 0.0:
        return 0.0, 0.0
    return value * value, value * slope


@njit(cache=True)
def _coefficients(factor, i, j, k, rr, px, py, pz):
    nx, ny, nz = factor.shape
    # Missing directions are absent, not artificial Neumann factor data.
    al = rr + px if i > 0 else 0.0
    bl = rr * factor[i - 1, j, k] if i > 0 else 0.0
    ar = rr - px if i + 1 < nx else 0.0
    br = rr * factor[i + 1, j, k] if i + 1 < nx else 0.0
    cl = rr + py if j > 0 else 0.0
    dl = rr * factor[i, j - 1, k] if j > 0 else 0.0
    cr = rr - py if j + 1 < ny else 0.0
    dr = rr * factor[i, j + 1, k] if j + 1 < ny else 0.0
    el = rr + pz if k > 0 else 0.0
    fl = rr * factor[i, j, k - 1] if k > 0 else 0.0
    er = rr - pz if k + 1 < nz else 0.0
    fr = rr * factor[i, j, k + 1] if k + 1 < nz else 0.0
    return al, bl, ar, br, cl, dl, cr, dr, el, fl, er, fr


@njit(cache=True)
def _hamiltonian(u, coefficients):
    al, bl, ar, br, cl, dl, cr, dr, el, fl, er, fr = coefficients
    hx, dx = _axis(u, al, bl, ar, br)
    hy, dy = _axis(u, cl, dl, cr, dr)
    hz, dz = _axis(u, el, fl, er, fr)
    return hx + hy + hz, 2.0 * (dx + dy + dz)


@njit(cache=True)
def _root(upper, slowness, coefficients):
    # Convex, increasing H: Newton from a supersolution remains above its root.
    u = upper
    target = slowness * slowness
    for _ in range(40):
        value, derivative = _hamiltonian(u, coefficients)
        defect = value - target
        if defect <= 0.0:
            return u
        step = defect / derivative
        next_u = u - step
        if next_u <= 0.0:
            next_u = 0.5 * u
        if u - next_u <= 2e-15 * u:
            return next_u
        u = next_u
    return u


@njit(cache=True)
def _sweep(factor, slowness, distance, known, ox, oy, oz, h, sx, sy, sz):
    nx, ny, nz = factor.shape
    change = 0.0
    for ii in range(nx):
        i = ii if sx > 0 else nx - 1 - ii
        for jj in range(ny):
            j = jj if sy > 0 else ny - 1 - jj
            for kk in range(nz):
                k = kk if sz > 0 else nz - 1 - kk
                if known[i, j, k]:
                    continue
                r = distance[i, j, k]
                coefficients = _coefficients(factor, i, j, k, r / h, ox[i] / r, oy[j] / r, oz[k] / r)
                previous = factor[i, j, k]
                value = min(previous, _root(previous, slowness[i, j, k], coefficients))
                change = max(change, (previous - value) / previous)
                factor[i, j, k] = value
    return change


@njit(cache=True)
def _solve(factor, slowness, distance, known, ox, oy, oz, h, tolerance, max_cycles):
    change = np.inf
    for cycle in range(max_cycles):
        change = 0.0
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    change = max(change, _sweep(factor, slowness, distance, known, ox, oy, oz, h, sx, sy, sz))
        if change <= tolerance:
            return cycle + 1, change
    return max_cycles, change


@njit(cache=True)
def _residual(factor, slowness, distance, known, ox, oy, oz, h):
    largest, smallest, square, count = -np.inf, np.inf, 0.0, 0
    nx, ny, nz = factor.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if known[i, j, k]:
                    continue
                r = distance[i, j, k]
                coefficients = _coefficients(factor, i, j, k, r / h, ox[i] / r, oy[j] / r, oz[k] / r)
                value, _ = _hamiltonian(factor[i, j, k], coefficients)
                relative = value / slowness[i, j, k] ** 2 - 1.0
                largest, smallest = max(largest, relative), min(smallest, relative)
                square += relative * relative
                count += 1
    if count == 0:
        return 0.0, 0.0, 0.0
    return smallest, largest, np.sqrt(square / count)


def _node_slowness(slowness):
    padded = np.pad(slowness, 1, mode="edge")
    nx, ny, nz = np.asarray(slowness.shape) + 1
    nodes = np.zeros((nx, ny, nz))
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                nodes += padded[a:a + nx, b:b + ny, c:c + nz] / 8.0
    return nodes


def travel_time_field(velocity, cell_size, source_m, *, radius_cells=3.0,
                      tolerance=1e-12, max_cycles=64, diagnostics=False, dtype=np.float32):
    """Cell-centred seconds from an off-grid source; study use only.

    Each solve starts from a fresh supersolution, not a previous model's times.
    Numerical convergence controls the forward solve, not inversion stopping.
    """
    started = time.perf_counter()
    velocity = np.asarray(velocity, dtype=np.float64)
    source = np.asarray(source_m, dtype=np.float64)
    if velocity.ndim != 3 or min(velocity.shape) < 1 or not np.all(np.isfinite(velocity)) or np.any(velocity <= 0):
        raise ValueError("velocity must be a nonempty positive finite 3D array")
    h = float(cell_size)
    extent = np.asarray(velocity.shape) * h
    if not np.isfinite(h) or h <= 0 or source.shape != (3,) or not np.all(np.isfinite(source)) or np.any(source < 0) or np.any(source > extent):
        raise ValueError("finite source must be inside the physical domain and cell_size positive")
    if (not np.isfinite(radius_cells) or radius_cells < 2.0
            or not np.isfinite(tolerance) or tolerance <= 0
            or isinstance(max_cycles, (bool, np.bool_))
            or not isinstance(max_cycles, (int, np.integer)) or max_cycles < 1):
        raise ValueError("finite radius_cells >= 2, finite tolerance > 0 and integer max_cycles >= 1 are required")
    output_dtype = np.dtype(dtype)
    if not np.issubdtype(output_dtype, np.floating):
        raise ValueError("dtype must be floating point")
    cell = tuple(np.minimum((source // h).astype(int), np.asarray(velocity.shape) - 1))
    source_slowness = 1.0 / velocity[cell]
    centre_offsets = [(np.arange(n) + 0.5) * h - s for n, s in zip(velocity.shape, source)]
    centre_distance = np.sqrt(centre_offsets[0][:, None, None] ** 2
                              + centre_offsets[1][None, :, None] ** 2
                              + centre_offsets[2][None, None, :] ** 2)
    radius = radius_cells * h
    slow = 1.0 / velocity
    slow[centre_distance <= radius] = source_slowness
    slowness = _node_slowness(slow)
    offsets = [np.arange(n + 1) * h - s for n, s in zip(velocity.shape, source)]
    distance = np.sqrt(offsets[0][:, None, None] ** 2
                       + offsets[1][None, :, None] ** 2
                       + offsets[2][None, None, :] ** 2)
    known = distance <= radius
    factor = np.full(distance.shape, max(float(slowness.max()), source_slowness))
    factor[known] = source_slowness
    cycles, change = _solve(factor, slowness, distance, known, *offsets, h, tolerance, max_cycles)
    residual = _residual(factor, slowness, distance, known, *offsets, h)
    if (not np.isfinite(change) or not np.all(np.isfinite(residual))
            or not np.all(np.isfinite(factor)) or np.any(factor <= 0)
            or change > tolerance or max(abs(residual[0]), abs(residual[1])) > 1e-7):
        raise RuntimeError(f"Factored solver did not converge: {cycles} cycles, change={change}, residual={residual}")
    nx, ny, nz = velocity.shape
    centre_factor = np.zeros(velocity.shape)
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                centre_factor += factor[a:a + nx, b:b + ny, c:c + nz] / 8.0
    result = centre_distance * centre_factor
    result[centre_distance <= radius] = centre_distance[centre_distance <= radius] * source_slowness
    result = result.astype(output_dtype)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("Factored solver returned nonfinite travel times")
    if not diagnostics:
        return result
    return result, {"cycles_of_8_sweeps": int(cycles), "max_relative_factor_change": float(change),
                    "relative_hamiltonian_residual": {"min": float(residual[0]), "max": float(residual[1]), "rms": float(residual[2])},
                    "seconds": time.perf_counter() - started,
                    "factor_min": float(factor.min()), "factor_max": float(factor.max())}
