"""Independent, off-grid point-source FMM using PyKonal (float64).

A small source box is initialized with straight-segment integrals through the
input voxels. Its physical radius shrinks with the numerical grid spacing.
This is a local approximation, not an exact heterogeneous source solution.
Receiver sampling interpolates T / distance, then restores the exact distance;
this avoids interpolating the point-source cusp directly.
"""
from __future__ import annotations

from dataclasses import replace
from itertools import product

import numpy as np

from .model import Arrival, ForwardConfig, PointSet, VelocityGrid
from .noise import NoiseConfig, add_arrival_noise

__version__ = "1"


def _nodal_velocity(model: VelocityGrid, refinement: int) -> np.ndarray:
    # Preserve the input voxels rather than smoothing them over the coarse-grid
    # scale. Only the numerical interface nodes average adjacent slownesses.
    slow = 1.0 / model.velocity
    shape = tuple(n * refinement + 1 for n in slow.shape)
    node_slow = np.zeros(shape, dtype=np.float64)
    indices = []
    for n in slow.shape:
        nodes = np.arange(n * refinement + 1)
        indices.append((np.clip((nodes - 1) // refinement, 0, n - 1),
                        np.clip(nodes // refinement, 0, n - 1)))
    for sides in product((0, 1), repeat=3):
        ix = [indices[axis][sides[axis]] for axis in range(3)]
        node_slow += slow[np.ix_(*ix)] / 8.0
    return 1.0 / node_slow


def _segment_time(model: VelocityGrid, source: np.ndarray, end: np.ndarray) -> float:
    """Exact straight-segment integral through piecewise-constant input voxels.

    Both positions use local coordinates. This is not a bent-ray solver.
    """
    delta = end - source
    length = float(np.linalg.norm(delta))
    if length == 0:
        return 0.0
    cuts = [0.0, 1.0]
    for axis in range(3):
        if delta[axis] == 0:
            continue
        lo, hi = sorted((source[axis], end[axis]))
        first = int(np.floor(lo / model.cell_size_m)) + 1
        last = int(np.ceil(hi / model.cell_size_m))
        faces = np.arange(first, last, dtype=float) * model.cell_size_m
        cuts.extend(((faces - source[axis]) / delta[axis]).tolist())
    cuts = np.unique(cuts)
    mids = source + (0.5 * (cuts[:-1] + cuts[1:]))[:, None] * delta
    cells = np.floor(mids / model.cell_size_m).astype(np.intp)
    cells = np.clip(cells, 0, np.array(model.velocity.shape) - 1)
    velocity = model.velocity[tuple(cells.T)]
    return float(length * np.sum(np.diff(cuts) / velocity))


def _sample_factored(field, source, receivers, h, source_slowness):
    """Trilinear interpolation of the regular factor T(x)/|x-source|."""
    coords = receivers / h
    lower = np.minimum(np.floor(coords).astype(np.intp), np.array(field.shape) - 2)
    weights = coords - lower
    result = np.zeros(len(receivers), dtype=np.float64)
    for offset in product((0, 1), repeat=3):
        index = lower + offset
        distances = np.linalg.norm(index * h - source, axis=1)
        factors = np.full(len(receivers), source_slowness, dtype=np.float64)
        np.divide(field[tuple(index.T)], distances, out=factors, where=distances > 0)
        weight = np.prod(np.where(np.array(offset), weights, 1.0 - weights), axis=1)
        result += weight * factors
    return result * np.linalg.norm(receivers - source, axis=1)


def compute_travel_times(
    model: VelocityGrid,
    stations: PointSet,
    events: PointSet,
    config: ForwardConfig = ForwardConfig(),
) -> np.ndarray:
    """Return absolute propagation seconds shaped (events, stations).

    Sources are the events; stations are receivers. No swapping by reciprocity
    is performed, so numerical results do not change with catalogue size/order.
    One field is held at a time. Boundaries are included, never clipped or padded.
    """
    model.validate_points(stations, "stations")
    model.validate_points(events, "events")
    try:
        from pykonal import EikonalSolver
    except ImportError as exc:
        raise ImportError("Install pykonal==0.4.1 to calculate forward arrivals") from exc

    h = model.cell_size_m / config.refinement
    velocity = _nodal_velocity(model, config.refinement)
    shape = np.array(velocity.shape)
    origin = np.array(model.origin_m)
    receivers = stations.coordinates_m - origin
    times = np.empty((len(events.ids), len(stations.ids)), dtype=np.float64)
    for row, source in enumerate(events.coordinates_m - origin):
        solver = EikonalSolver(coord_sys="cartesian")
        solver.velocity.min_coords = (0.0, 0.0, 0.0)
        solver.velocity.node_intervals = (h, h, h)
        solver.velocity.npts = tuple(shape)
        solver.velocity.values = velocity
        base = np.floor(source / h).astype(np.intp)
        radius = config.source_radius_cells
        ranges = [range(max(0, int(i) - radius + 1), min(int(n), int(i) + radius + 1))
                  for i, n in zip(base, shape)]
        # Initial source-box values are local straight-ray upper bounds. The
        # approximation is tested by refining both the box and the global grid.
        for index in product(*ranges):
            solver.traveltime.values[index] = _segment_time(model, source, np.array(index) * h)
            solver.unknown[index] = False
            solver.trial.push(*index)
        if not solver.solve():
            raise RuntimeError(f"Forward FMM failed for event {events.ids[row]}")
        cell = np.clip(np.floor(source / model.cell_size_m).astype(np.intp),
                       0, np.array(model.velocity.shape) - 1)
        times[row] = _sample_factored(
            solver.traveltime.values, source, receivers, h,
            1.0 / model.velocity[tuple(cell)],
        )
        if not np.all(np.isfinite(times[row])) or np.any(times[row] < 0):
            raise RuntimeError(f"Invalid travel times for event {events.ids[row]}")
    return times


def compute_arrivals(
    model: VelocityGrid,
    stations: PointSet,
    events: PointSet,
    config: ForwardConfig = ForwardConfig(),
    *,
    noise: NoiseConfig | None = None,
) -> list[Arrival]:
    """All pairs, optionally noisy; each event's earliest observed pick is zero."""
    times = compute_travel_times(model, stations, events, config)
    if noise is not None:
        times = add_arrival_noise(times, events.ids, stations.ids, noise)
    times -= times.min(axis=1, keepdims=True)
    return [Arrival(station_id, event_id, float(times[e, s]))
            for e, event_id in enumerate(events.ids)
            for s, station_id in enumerate(stations.ids)]


def check_convergence(
    model: VelocityGrid,
    stations: PointSet,
    events: PointSet,
    config: ForwardConfig = ForwardConfig(),
) -> dict:
    """Compare absolute and relative times at r and 2r, not an error bound.

    Absolute differences also matter: subtracting the first arrival can hide a
    common source error. For reference datasets, check more than two resolutions.
    """
    coarse = compute_travel_times(model, stations, events, config)
    fine = compute_travel_times(model, stations, events, replace(config, refinement=2 * config.refinement))
    absolute_delta = fine - coarse
    coarse -= coarse.min(axis=1, keepdims=True)
    fine -= fine.min(axis=1, keepdims=True)
    delta = fine - coarse
    return {
        "coarse_refinement": config.refinement,
        "fine_refinement": 2 * config.refinement,
        "max_abs_difference_s": float(np.max(np.abs(delta))),
        "rms_difference_s": float(np.sqrt(np.mean(delta ** 2))),
        "max_absolute_time_difference_s": float(np.max(np.abs(absolute_delta))),
        "rms_absolute_time_difference_s": float(np.sqrt(np.mean(absolute_delta ** 2))),
    }
