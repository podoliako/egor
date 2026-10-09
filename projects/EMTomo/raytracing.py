"""Numba ray tracing down travel-time gradients and ray-length rasterization.

compute_G_all_stations        — prange over stations; use in a single process
compute_G_all_stations_serial — single-threaded; use inside fork workers
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
    Gradient-descent ray trace in cell-centred index coordinates.

    Integer coordinates denote cell centres. Returns the path and a flag showing
    whether the ray reached the station. Unsuccessful partial paths must not be
    used to build sensitivity matrices.
    """
    x0 = min(max(epic[0], x_lo[0]), x_hi[0])
    x1 = min(max(epic[1], x_lo[1]), x_hi[1])
    x2 = min(max(epic[2], x_lo[2]), x_hi[2])

    # One extra slot is reserved for the exact station position.
    buf = np.empty((max_steps + 2, 3), dtype=np.float64)
    buf[0, 0] = x0;  buf[0, 1] = x1;  buf[0, 2] = x2
    n = 1
    nx_, ny_, nz_ = gx.shape

    for _ in range(max_steps):
        dx = x0 - station[0];  dy = x1 - station[1];  dz = x2 - station[2]
        # A surface station lies half a cell outside the cell-centre domain and
        # the gradient there points out of it, so clipping can stall before the
        # tolerance is met. Coming within half a cell is sufficient; the exact
        # station position is appended below.
        in_source_cell = abs(dx) <= 0.5 and abs(dy) <= 0.5 and abs(dz) <= 0.5
        if dx * dx + dy * dy + dz * dz <= tol_sq or in_source_cell:
            if dx * dx + dy * dy + dz * dz > 1e-24:
                buf[n, 0] = station[0]
                buf[n, 1] = station[1]
                buf[n, 2] = station[2]
                n += 1
            return buf[:n], True

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

    return buf[:n], False


@njit(cache=True, fastmath=True)
def _rasterize_nb(path, G, cell_len, ix, wx, iy, wy, iz, wz, eps=1e-12):
    """DDA ray-length accumulation over the fine grid, deposited into ``G``.

    The path is in fine cell-centred index coordinates. A length in fine cell
    ``(i, j, k)`` is added to ``G[ix[a, i], iy[b, j], iz[c, k]]`` with weight
    ``wx[a, i] * wy[b, j] * wz[c, k]`` (a, b, c in {0, 1}); identity tables give
    the fine grid itself.
    """
    nx = ix.shape[1];  ny = iy.shape[1];  nz = iz.shape[1]
    # Integer coordinate i is the centre of cell i, whose bounds are i ± 0.5.
    lx = ly = lz = -0.5
    hx = float(nx) - 0.5 - 1e-9
    hy = float(ny) - 0.5 - 1e-9
    hz = float(nz) - 0.5 - 1e-9

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
            lo = lx if ax == 0 else (ly if ax == 1 else lz)
            if abs(da) < eps:
                if aa < lo or aa > hi:
                    skip = True;  break
            else:
                tn = (lo - aa) / da;  tf = (hi - aa) / da
                if tn > tf:  tn, tf = tf, tn
                t0c = max(t0c, tn);  t1c = min(t1c, tf)
                if t0c > t1c:  skip = True;  break
        if skip:
            continue

        ca0 = a0 + t0c * d0;  ca1 = a1 + t0c * d1;  ca2 = a2 + t0c * d2
        cd0 = d0 * (t1c - t0c);  cd1 = d1 * (t1c - t0c);  cd2 = d2 * (t1c - t0c)
        seg_len = (cd0 * cd0 + cd1 * cd1 + cd2 * cd2) ** 0.5 * cell_len
        if seg_len < eps:
            continue

        # ── DDA traversal ─────────────────────────────────────────────────
        i = int(np.floor(ca0 + 0.5));  j = int(np.floor(ca1 + 0.5));  k = int(np.floor(ca2 + 0.5))
        si_ = 1 if cd0 > 0 else (-1 if cd0 < 0 else 0)
        sj_ = 1 if cd1 > 0 else (-1 if cd1 < 0 else 0)
        sk_ = 1 if cd2 > 0 else (-1 if cd2 < 0 else 0)

        # A segment starting exactly on a cell boundary belongs to the cell it
        # enters, not the one it leaves.
        if cd0 < 0 and abs(ca0 - (i - 0.5)) < eps:  i -= 1
        if cd1 < 0 and abs(ca1 - (j - 0.5)) < eps:  j -= 1
        if cd2 < 0 and abs(ca2 - (k - 0.5)) < eps:  k -= 1

        tm0 = tm1 = tm2 = 2.0
        td0 = td1 = td2 = 2.0
        if abs(cd0) >= eps:
            nb = i + 0.5 if cd0 > 0 else i - 0.5
            tm0 = (nb - ca0) / cd0
            td0 = 1.0 / abs(cd0)
        if abs(cd1) >= eps:
            nb = j + 0.5 if cd1 > 0 else j - 0.5
            tm1 = (nb - ca1) / cd1
            td1 = 1.0 / abs(cd1)
        if abs(cd2) >= eps:
            nb = k + 0.5 if cd2 > 0 else k - 0.5
            tm2 = (nb - ca2) / cd2
            td2 = 1.0 / abs(cd2)

        t = 0.0
        while True:
            if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):  break
            t_next = min(1.0, tm0, tm1, tm2);  dt = t_next - t
            if dt > 0.0:
                length = dt * seg_len
                for a in range(2):
                    wa = wx[a, i]
                    if wa == 0.0:
                        continue
                    for b in range(2):
                        wab = wa * wy[b, j]
                        if wab == 0.0:
                            continue
                        for c in range(2):
                            w = wab * wz[c, k]
                            if w != 0.0:
                                G[ix[a, i], iy[b, j], iz[c, k]] += length * w
            if t_next >= 1.0 - 1e-15:  break
            if tm0 <= t_next + 1e-12:  i += si_;  tm0 += td0
            if tm1 <= t_next + 1e-12:  j += sj_;  tm1 += td1
            if tm2 <= t_next + 1e-12:  k += sk_;  tm2 += td2
            t = t_next


# ── Sensitivity builders ─────────────────────────────────────────────────────
# Arguments: stacked travel-time gradients (n_st, nx, ny, nz) on the fine grid,
# station positions (n_st, 3) and the hypocentre (3,) in fine continuous index
# coordinates, fine cell size, ray step and tolerance (index units), step limit,
# clipping bounds, restriction tables (see instruments_ops) and the output shape.

@njit(parallel=True, cache=True, fastmath=True)
def compute_G_all_stations(gx, gy, gz, sl, epic, cell_len, step, tol, max_steps,
                           x_lo, x_hi, ix, wx, iy, wy, iz, wz, out_shape):
    """Ray lengths per output cell for every station, parallel over stations.
    Returns ``(G, reached)``; unreached stations get all-zero rows."""
    n_st = gx.shape[0]
    G_all = np.zeros((n_st, out_shape[0], out_shape[1], out_shape[2]), dtype=np.float64)
    reached = np.zeros(n_st, dtype=np.bool_)
    tol_sq = tol * tol
    for si in prange(n_st):
        path, ray_reached = _trace_ray_nb(
            gx[si], gy[si], gz[si], sl[si], epic, step, tol_sq, max_steps, x_lo, x_hi,
        )
        reached[si] = ray_reached
        if ray_reached:
            _rasterize_nb(path, G_all[si], cell_len, ix, wx, iy, wy, iz, wz)
    return G_all, reached


@njit(cache=True, fastmath=True)
def compute_G_all_stations_serial(gx, gy, gz, sl, epic, cell_len, step, tol, max_steps,
                                  x_lo, x_hi, ix, wx, iy, wy, iz, wz, out_shape):
    """Same as compute_G_all_stations without prange, for fork worker processes."""
    n_st = gx.shape[0]
    G_all = np.zeros((n_st, out_shape[0], out_shape[1], out_shape[2]), dtype=np.float64)
    reached = np.zeros(n_st, dtype=np.bool_)
    tol_sq = tol * tol
    for si in range(n_st):
        path, ray_reached = _trace_ray_nb(
            gx[si], gy[si], gz[si], sl[si], epic, step, tol_sq, max_steps, x_lo, x_hi,
        )
        reached[si] = ray_reached
        if ray_reached:
            _rasterize_nb(path, G_all[si], cell_len, ix, wx, iy, wy, iz, wz)
    return G_all, reached
