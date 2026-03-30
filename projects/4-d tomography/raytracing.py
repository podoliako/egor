import numpy as np
from components.utilities import trilinear_interpolation


def _sample_with_existing_trilinear(values, p_idx):
    nx, ny, nz = values.shape
    i0 = np.floor(p_idx).astype(int)
    di, dj, dk = p_idx - i0
    i, j, k = int(i0[0]), int(i0[1]), int(i0[2])
    if i < 0 or j < 0 or k < 0 or i >= nx or j >= ny or k >= nz:
        return None
    return trilinear_interpolation(values, i, j, k, float(di), float(dj), float(dk))


def _grad_at(gradT, p_idx):
    gx = _sample_with_existing_trilinear(gradT[0], p_idx)
    gy = _sample_with_existing_trilinear(gradT[1], p_idx)
    gz = _sample_with_existing_trilinear(gradT[2], p_idx)
    if gx is None or gy is None or gz is None:
        return None
    return np.array([gx, gy, gz], dtype=float)


def trace_ray_from_timefield(
    T, station_xyz, epic_xyz, spacing_xyz,
    step=None, tol=None, max_steps=50000
):
    """
    T shape: (nx, ny, nz), travel-time field for source at station.
    coords: same units as origin/spacing.

    After each gradient step the position is clamped to the grid bounds so
    that shallow rays near the surface boundary cannot escape the domain.
    Without clamping, np.gradient's one-sided edge derivative can have an
    upward component that kicks x.z below zero, causing _grad_at to return
    None and the trace to terminate before reaching the station.
    """
    station = np.asarray(station_xyz, dtype=float)
    x = np.asarray(epic_xyz, dtype=float).copy()
    origin = np.asarray((0, 0, 0), dtype=float)
    spacing = np.asarray(spacing_xyz, dtype=float)

    if step is None:
        step = 0.5 * spacing.min()
    if tol is None:
        tol = step

    # Grid bounds in the same index units as x
    shape = np.asarray(T.shape, dtype=float)
    x_lo = origin                           # [0, 0, 0]
    x_hi = origin + (shape - 1) * spacing  # [(nx-1)*s, ...]

    gradT = np.gradient(T, *spacing, edge_order=1)

    path = [x.copy()]
    for _ in range(max_steps):
        if np.linalg.norm(x - station) <= tol:
            break

        # Clamp before gradient lookup so we always stay inside the grid.
        # This handles the case where the previous step slightly overshot a
        # boundary (most commonly z < 0 for surface stations with shallow rays).
        x_clamped = np.clip(x, x_lo, x_hi)

        p_idx = (x_clamped - origin) / spacing
        g = _grad_at(gradT, p_idx)
        if g is None or not np.isfinite(g).all():
            break
        ng = np.linalg.norm(g)
        if ng < 1e-12:
            break

        x = x_clamped - step * (g / ng)  # move opposite gradient (toward source)
        # Clamp after step too — ensures the appended point is always valid
        x = np.clip(x, x_lo, x_hi)
        path.append(x.copy())
    return np.vstack(path)


def rasterize_path_binary(path_xyz, shape, origin_xyz, spacing_xyz):
    G3 = np.zeros(shape, dtype=np.uint8)
    origin = np.asarray(origin_xyz, dtype=float)
    spacing = np.asarray(spacing_xyz, dtype=float)

    for a, b in zip(path_xyz[:-1], path_xyz[1:]):
        seg = b - a
        L = np.linalg.norm(seg)
        n = max(1, int(np.ceil(2.0 * L / spacing.min())))
        t = np.linspace(0.0, 1.0, n + 1)[:, None]
        pts = a + t * seg
        ijk = np.floor((pts - origin) / spacing).astype(int)
        valid = (
            (ijk[:, 0] >= 0) & (ijk[:, 0] < shape[0]) &
            (ijk[:, 1] >= 0) & (ijk[:, 1] < shape[1]) &
            (ijk[:, 2] >= 0) & (ijk[:, 2] < shape[2])
        )
        ijk = ijk[valid]
        G3[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = 1
    return G3


def _clip_segment_to_grid(a, b, shape, eps=1e-9):
    lo = np.array([0.0, 0.0, 0.0], dtype=float)
    hi = np.array(shape, dtype=float) - eps
    d = b - a
    t0, t1 = 0.0, 1.0
    for ax in range(3):
        if abs(d[ax]) < eps:
            if a[ax] < lo[ax] or a[ax] > hi[ax]:
                return None, None
        else:
            t_near = (lo[ax] - a[ax]) / d[ax]
            t_far = (hi[ax] - a[ax]) / d[ax]
            if t_near > t_far:
                t_near, t_far = t_far, t_near
            t0 = max(t0, t_near)
            t1 = min(t1, t_far)
            if t0 > t1:
                return None, None
    return a + t0 * d, a + t1 * d


def _accumulate_segment_lengths(G, a, b, voxel_size=(1.0, 1.0, 1.0), eps=1e-12):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = _clip_segment_to_grid(a, b, G.shape)
    if a is None:
        return

    d = b - a
    if np.linalg.norm(d) < eps:
        return

    voxel_size = np.asarray(voxel_size, float)
    seg_len_phys = np.linalg.norm(d * voxel_size)
    if seg_len_phys < eps:
        return

    i, j, k = np.floor(a).astype(int)
    step = np.sign(d).astype(int)

    t_max = np.array([np.inf, np.inf, np.inf], dtype=float)
    t_delta = np.array([np.inf, np.inf, np.inf], dtype=float)

    for ax in range(3):
        if abs(d[ax]) < eps:
            continue
        if d[ax] > 0:
            next_boundary = np.floor(a[ax]) + 1.0
            t_max[ax] = (next_boundary - a[ax]) / d[ax]
        else:
            next_boundary = np.floor(a[ax])
            # If a[ax] is exactly on a cell boundary and moving negative,
            # t_max would be 0 — the DDA would skip the starting cell entirely.
            if next_boundary >= a[ax] - eps:
                next_boundary -= 1.0
            t_max[ax] = (a[ax] - next_boundary) / (-d[ax])
        t_delta[ax] = 1.0 / abs(d[ax])

    t = 0.0
    nx, ny, nz = G.shape

    while True:
        if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
            break

        t_next = min(1.0, t_max[0], t_max[1], t_max[2])
        dt = t_next - t
        if dt > 0:
            G[i, j, k] += dt * seg_len_phys

        if t_next >= 1.0 - 1e-15:
            break

        if t_max[0] <= t_next + 1e-12:
            i += step[0]
            t_max[0] += t_delta[0]
        if t_max[1] <= t_next + 1e-12:
            j += step[1]
            t_max[1] += t_delta[1]
        if t_max[2] <= t_next + 1e-12:
            k += step[2]
            t_max[2] += t_delta[2]

        t = t_next


def rasterize_path_lengths(path_xyz, shape, voxel_size=(1.0, 1.0, 1.0), dtype=np.float32):
    """
    Returns G_len where each voxel stores ray length inside it (not just 0/1).
    path_xyz: (N,3) polyline in index coordinates.
    """
    path = np.asarray(path_xyz, dtype=float)
    G = np.zeros(shape, dtype=np.float64)
    for a, b in zip(path[:-1], path[1:]):
        _accumulate_segment_lengths(G, a, b, voxel_size=voxel_size)
    return G.astype(dtype, copy=False)