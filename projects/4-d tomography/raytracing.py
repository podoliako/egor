"""
raytracing.py — Numba-accelerated ray tracing.

Public API (trace_ray_from_timefield, rasterize_path_lengths) is unchanged.

New exports
-----------
compute_G_all_stations        — Numba prange over stations; use with n_workers=1
compute_G_all_stations_serial — single-threaded Numba; use inside fork workers

JIT compilation is triggered on first call and cached to __pycache__ (~3–5 s
warm-up once; subsequent runs load from cache in < 0.1 s).
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange # pyright: ignore[reportMissingImports]


# ── Numba kernels ─────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def _trilinear_nb(v, i, j, k, di, dj, dk):
    nx, ny, nz = v.shape
    i1 = i + 1 if i + 1 < nx else i
    j1 = j + 1 if j + 1 < ny else j
    k1 = k + 1 if k + 1 < nz else k
    return (v[i , j , k ] * (1 - di) * (1 - dj) * (1 - dk)
          + v[i1, j , k ] *      di  * (1 - dj) * (1 - dk)
          + v[i , j1, k ] * (1 - di) *      dj  * (1 - dk)
          + v[i1, j1, k ] *      di  *      dj  * (1 - dk)
          + v[i , j , k1] * (1 - di) * (1 - dj) *      dk
          + v[i1, j , k1] *      di  * (1 - dj) *      dk
          + v[i , j1, k1] * (1 - di) *      dj  *      dk
          + v[i1, j1, k1] *      di  *      dj  *      dk)


@njit(cache=True, fastmath=True)
def _trace_ray_nb(gx, gy, gz, station, epic, step, tol_sq, max_steps, x_lo, x_hi):
    """
    Gradient-descent ray trace in index coordinates (spacing = 1 assumed).
    Receives tol_sq = tol**2 to avoid one sqrt per step.
    Returns (N, 3) contiguous float64 path.
    """
    x0 = min(max(epic[0], x_lo[0]), x_hi[0])
    x1 = min(max(epic[1], x_lo[1]), x_hi[1])
    x2 = min(max(epic[2], x_lo[2]), x_hi[2])

    buf = np.empty((max_steps + 1, 3), dtype=np.float64)
    buf[0, 0] = x0;  buf[0, 1] = x1;  buf[0, 2] = x2
    n = 1
    nx_, ny_, nz_ = gx.shape

    for _ in range(max_steps):
        dx = x0 - station[0];  dy = x1 - station[1];  dz = x2 - station[2]
        if dx * dx + dy * dy + dz * dz <= tol_sq:
            break

        cx = min(max(x0, x_lo[0]), x_hi[0])
        cy = min(max(x1, x_lo[1]), x_hi[1])
        cz = min(max(x2, x_lo[2]), x_hi[2])

        i = int(np.floor(cx));  j = int(np.floor(cy));  k = int(np.floor(cz))
        if not (0 <= i < nx_ and 0 <= j < ny_ and 0 <= k < nz_):
            break
        di = cx - i;  dj = cy - j;  dk = cz - k

        vx = _trilinear_nb(gx, i, j, k, di, dj, dk)
        vy = _trilinear_nb(gy, i, j, k, di, dj, dk)
        vz = _trilinear_nb(gz, i, j, k, di, dj, dk)

        ng2 = vx * vx + vy * vy + vz * vz
        if ng2 < 1e-24:
            break
        s = step / np.sqrt(ng2)

        x0 = min(max(cx - vx * s, x_lo[0]), x_hi[0])
        x1 = min(max(cy - vy * s, x_lo[1]), x_hi[1])
        x2 = min(max(cz - vz * s, x_lo[2]), x_hi[2])

        buf[n, 0] = x0;  buf[n, 1] = x1;  buf[n, 2] = x2
        n += 1

    return buf[:n]


@njit(cache=True, fastmath=True)
def _rasterize_nb(path, G, vsx, vsy, vsz, eps=1e-12):
    """DDA ray-length accumulation into G in-place (index coordinates)."""
    nx, ny, nz = G.shape
    hx = float(nx) - 1e-9
    hy = float(ny) - 1e-9
    hz = float(nz) - 1e-9

    for s in range(path.shape[0] - 1):
        a0 = path[s,   0];  a1 = path[s,   1];  a2 = path[s,   2]
        b0 = path[s+1, 0];  b1 = path[s+1, 1];  b2 = path[s+1, 2]
        d0 = b0 - a0;       d1 = b1 - a1;       d2 = b2 - a2

        # ── clip to grid ──────────────────────────────────────────────────
        t0c = 0.0;  t1c = 1.0;  skip = False
        for ax in range(3):
            aa = a0 if ax == 0 else (a1 if ax == 1 else a2)
            da = d0 if ax == 0 else (d1 if ax == 1 else d2)
            hi = hx if ax == 0 else (hy if ax == 1 else hz)
            if abs(da) < eps:
                if aa < 0.0 or aa > hi:
                    skip = True;  break
            else:
                tn = (0.0 - aa) / da;  tf = (hi - aa) / da
                if tn > tf:  tn, tf = tf, tn
                t0c = max(t0c, tn);  t1c = min(t1c, tf)
                if t0c > t1c:  skip = True;  break
        if skip:
            continue

        ca0 = a0 + t0c * d0;  ca1 = a1 + t0c * d1;  ca2 = a2 + t0c * d2
        cd0 = d0 * (t1c - t0c);  cd1 = d1 * (t1c - t0c);  cd2 = d2 * (t1c - t0c)
        seg_len = ((cd0 * vsx) ** 2 + (cd1 * vsy) ** 2 + (cd2 * vsz) ** 2) ** 0.5
        if seg_len < eps:
            continue

        # ── DDA traversal ─────────────────────────────────────────────────
        i = int(np.floor(ca0));  j = int(np.floor(ca1));  k = int(np.floor(ca2))
        si_ = 1 if cd0 > 0 else (-1 if cd0 < 0 else 0)
        sj_ = 1 if cd1 > 0 else (-1 if cd1 < 0 else 0)
        sk_ = 1 if cd2 > 0 else (-1 if cd2 < 0 else 0)

        tm0 = tm1 = tm2 = 2.0
        td0 = td1 = td2 = 2.0
        if abs(cd0) >= eps:
            nb = np.floor(ca0) + 1.0 if cd0 > 0 else np.floor(ca0)
            if cd0 < 0 and nb >= ca0 - eps:  nb -= 1.0
            tm0 = (nb - ca0) / cd0 if cd0 > 0 else (ca0 - nb) / (-cd0)
            td0 = 1.0 / abs(cd0)
        if abs(cd1) >= eps:
            nb = np.floor(ca1) + 1.0 if cd1 > 0 else np.floor(ca1)
            if cd1 < 0 and nb >= ca1 - eps:  nb -= 1.0
            tm1 = (nb - ca1) / cd1 if cd1 > 0 else (ca1 - nb) / (-cd1)
            td1 = 1.0 / abs(cd1)
        if abs(cd2) >= eps:
            nb = np.floor(ca2) + 1.0 if cd2 > 0 else np.floor(ca2)
            if cd2 < 0 and nb >= ca2 - eps:  nb -= 1.0
            tm2 = (nb - ca2) / cd2 if cd2 > 0 else (ca2 - nb) / (-cd2)
            td2 = 1.0 / abs(cd2)

        t = 0.0
        while True:
            if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):  break
            t_next = min(1.0, tm0, tm1, tm2);  dt = t_next - t
            if dt > 0.0:  G[i, j, k] += dt * seg_len
            if t_next >= 1.0 - 1e-15:  break
            if tm0 <= t_next + 1e-12:  i += si_;  tm0 += td0
            if tm1 <= t_next + 1e-12:  j += sj_;  tm1 += td1
            if tm2 <= t_next + 1e-12:  k += sk_;  tm2 += td2
            t = t_next


# ── High-level G-tensor builders ──────────────────────────────────────────────

@njit(parallel=True, cache=True, fastmath=True)
def compute_G_all_stations(
    gx, gy, gz,       # (n_st, nx, ny, nz) float64 — stacked gradient components
    sl,               # (n_st, 3)           float64 — station positions (index coords)
    epic,             # (3,)                float64 — epicenter (index coords)
    vsx, vsy, vsz,    # physical voxel sizes (scalars)
    step, tol,        # ray-trace step and tolerance (index units)
    max_steps,
    x_lo, x_hi,       # (3,) grid bounds (index coords)
):
    """
    Traces rays from *epic* to every station in parallel via Numba prange.
    Returns G tensor (n_st, nx, ny, nz) float64.

    Use this when n_workers = 1 (single process; Numba uses all available cores).
    With 40 stations on a 48-core machine this saturates ~40 cores.
    """
    n_st = gx.shape[0]
    nx_ = gx.shape[1];  ny_ = gx.shape[2];  nz_ = gx.shape[3]
    G_all = np.zeros((n_st, nx_, ny_, nz_), dtype=np.float64)
    tol_sq = tol * tol

    for si in prange(n_st):
        path = _trace_ray_nb(
            gx[si], gy[si], gz[si], sl[si], epic,
            step, tol_sq, max_steps, x_lo, x_hi,
        )
        _rasterize_nb(path, G_all[si], vsx, vsy, vsz)

    return G_all


@njit(cache=True, fastmath=True)
def compute_G_all_stations_serial(
    gx, gy, gz, sl, epic,
    vsx, vsy, vsz, step, tol, max_steps, x_lo, x_hi,
):
    """
    Same as compute_G_all_stations but without prange — for use inside
    multiprocessing worker processes (one process per core, Numba single-threaded).
    """
    n_st = gx.shape[0]
    nx_ = gx.shape[1];  ny_ = gx.shape[2];  nz_ = gx.shape[3]
    G_all = np.zeros((n_st, nx_, ny_, nz_), dtype=np.float64)
    tol_sq = tol * tol

    for si in range(n_st):
        path = _trace_ray_nb(
            gx[si], gy[si], gz[si], sl[si], epic,
            step, tol_sq, max_steps, x_lo, x_hi,
        )
        _rasterize_nb(path, G_all[si], vsx, vsy, vsz)

    return G_all


# ── Public backward-compatible API ────────────────────────────────────────────

def trace_ray_from_timefield(
    T, station_xyz, epic_xyz, spacing_xyz,
    step=None, tol=None, max_steps=50000, gradT=None,
):
    station = np.asarray(station_xyz, dtype=np.float64)
    epic    = np.asarray(epic_xyz,    dtype=np.float64)
    spacing = np.asarray(spacing_xyz, dtype=np.float64)

    if step is None:  step = float(0.5 * spacing.min())
    if tol  is None:  tol  = step

    x_lo = np.zeros(3, dtype=np.float64)
    x_hi = (np.asarray(T.shape, dtype=np.float64) - 1) * spacing

    if gradT is None:
        gradT = np.gradient(T, *spacing_xyz, edge_order=1)

    gx = np.ascontiguousarray(gradT[0], dtype=np.float64)
    gy = np.ascontiguousarray(gradT[1], dtype=np.float64)
    gz = np.ascontiguousarray(gradT[2], dtype=np.float64)

    return _trace_ray_nb(gx, gy, gz, station, epic, step, tol * tol, max_steps, x_lo, x_hi)


def rasterize_path_lengths(path_xyz, shape, voxel_size=(1.0, 1.0, 1.0), dtype=np.float32):
    path = np.ascontiguousarray(path_xyz, dtype=np.float64)
    G    = np.zeros(shape, dtype=np.float64)
    _rasterize_nb(path, G, float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2]))
    return G.astype(dtype, copy=False)