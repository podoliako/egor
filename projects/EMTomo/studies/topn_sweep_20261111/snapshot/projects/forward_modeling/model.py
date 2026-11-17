"""SI-unit inputs for a single-wave, isotropic forward problem."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class VelocityGrid:
    """Piecewise-constant voxel velocities; origin is the lower domain corner."""

    velocity: np.ndarray
    cell_size_m: float
    origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        velocity = np.array(self.velocity, dtype=np.float64, copy=True)
        if velocity.ndim != 3 or any(n == 0 for n in velocity.shape):
            raise ValueError("velocity must be a nonempty 3-D array (nx, ny, nz)")
        if not np.all(np.isfinite(velocity)) or np.any(velocity <= 0):
            raise ValueError("velocity must contain finite positive values in m/s")
        h = float(self.cell_size_m)
        if not np.isfinite(h) or h <= 0:
            raise ValueError("cell_size_m must be finite and positive")
        origin = np.asarray(self.origin_m, dtype=np.float64)
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError("origin_m must contain three finite coordinates")
        if not np.all(np.isfinite(origin + h * np.array(velocity.shape))):
            raise ValueError("model bounds must be finite")
        velocity.flags.writeable = False
        object.__setattr__(self, "velocity", velocity)
        object.__setattr__(self, "cell_size_m", h)
        object.__setattr__(self, "origin_m", tuple(float(x) for x in origin))

    def validate_points(self, points: PointSet, label: str):
        local = points.coordinates_m - np.asarray(self.origin_m)
        extent = self.cell_size_m * np.asarray(self.velocity.shape)
        invalid = np.any((local < 0) | (local > extent), axis=1)
        if np.any(invalid):
            ids = [points.ids[i] for i in np.flatnonzero(invalid)]
            raise ValueError(f"{label} outside the closed model domain: {ids}")


@dataclass(frozen=True)
class PointSet:
    """Exact xyz coordinates in metres. Integer IDs are normalized to strings."""

    ids: tuple[str, ...]
    coordinates_m: np.ndarray

    def __post_init__(self):
        ids = tuple(self.ids)
        if not ids or any(isinstance(i, bool) or not isinstance(i, (str, Integral)) for i in ids):
            raise ValueError("ids must be a nonempty sequence of strings or integers")
        ids = tuple(str(i) for i in ids)
        if any(not i.strip() for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("ids must be nonempty and unique after string conversion")
        coords = np.array(self.coordinates_m, dtype=np.float64, copy=True)
        if coords.shape != (len(ids), 3) or not np.all(np.isfinite(coords)):
            raise ValueError("coordinates_m must have shape (len(ids), 3) and be finite")
        coords.flags.writeable = False
        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "coordinates_m", coords)


@dataclass(frozen=True)
class ForwardConfig:
    refinement: int = 2
    source_radius_cells: int = 2

    def __post_init__(self):
        for name in ("refinement", "source_radius_cells"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")
            object.__setattr__(self, name, int(value))


@dataclass(frozen=True)
class Arrival:
    station_id: str
    event_id: str
    arrival_time_s: float
