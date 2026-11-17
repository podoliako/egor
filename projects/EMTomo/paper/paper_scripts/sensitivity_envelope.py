"""Display envelope only: convexify and widen a 2-D sensitivity selection.

Added pixels are NOT evidence of sensitivity or spatial resolution.
"""
import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from scipy.spatial import ConvexHull, Delaunay

THRESHOLD = 0.5
PADDING_M = 500.


def expanded_envelope(raw_mask, spacing_m, padding_m=PADDING_M):
    raw = np.asarray(raw_mask, dtype=bool)
    if raw.ndim != 2 or not raw.any():
        raise ValueError("Expected a nonempty 2-D mask")
    if not np.isfinite(spacing_m) or spacing_m <= 0 or not np.isfinite(padding_m) or padding_m < 0:
        raise ValueError("Spacing must be positive and padding nonnegative")
    points = np.column_stack(np.nonzero(raw))
    hull_mask = raw.copy()
    if len(points) >= 3 and np.linalg.matrix_rank(points - points[0]) == 2:
        hull = ConvexHull(points)
        domain = np.column_stack(np.indices(raw.shape).reshape(2, -1))
        hull_mask |= (Delaunay(points[hull.vertices]).find_simplex(domain, tol=1e-9) >= 0).reshape(raw.shape)
    envelope = distance_transform_edt(~hull_mask, sampling=spacing_m) <= padding_m
    envelope = binary_fill_holes(envelope | hull_mask)
    if not np.all(envelope[raw]):
        raise AssertionError("Envelope must contain the original selection")
    return envelope


def cell_boundary_segments(mask, cell_size, origin=(0., 0.)):
    """Axis-aligned exposed cell edges, including domain-edge boundaries.

    Rows increase along Y, columns along X. Units follow cell_size and origin.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not np.isfinite(cell_size) or cell_size <= 0:
        raise ValueError("Expected a 2-D mask and positive finite cell size")
    padded = np.pad(mask, 1)
    segments = []
    for dr, dc, edge in [(-1,0,((0,0),(1,0))), (1,0,((0,1),(1,1))),
                         (0,-1,((0,0),(0,1))), (0,1,((1,0),(1,1)))]:
        neighbour = padded[1+dr:1+dr+mask.shape[0], 1+dc:1+dc+mask.shape[1]]
        rows, cols = np.nonzero(mask & ~neighbour)
        if len(rows):
            starts = np.column_stack((cols, rows))
            segments.append((starts[:, None, :] + np.asarray(edge)[None, :, :]) * cell_size + origin)
    return np.concatenate(segments) if segments else np.empty((0, 2, 2))
