"""Station travel-time fields by fast marching (scikit-fmm) and reciprocity."""
from __future__ import annotations

import multiprocessing as mp

import numpy as np
import skfmm

from velocity_model import VelocityModel

# Radius, in cells, of the homogeneous sphere that seeds fast marching.
SOURCE_RADIUS_CELLS = 3.0


def travel_time_field(velocity: np.ndarray, cell_size: float, source_m) -> np.ndarray:
    """First-arrival times in seconds from an exact source point to every cell centre.

    Within ``SOURCE_RADIUS_CELLS`` of the source the medium is taken as
    homogeneous with the source cell velocity, and fast marching continues from
    that sphere. Seeding a single node instead would move the source to a cell
    centre and keep the large first-order FMM error of a point source.
    """
    velocity = np.asarray(velocity, dtype=np.float64)
    source = np.asarray(source_m, dtype=np.float64)
    extent = np.asarray(velocity.shape) * cell_size
    if source.shape != (3,) or np.any(source < 0) or np.any(source > extent):
        raise ValueError(f"source {source_m} is outside the model domain")

    offsets = [(np.arange(n) + 0.5) * cell_size - s for n, s in zip(velocity.shape, source)]
    distance = np.sqrt(
        offsets[0][:, None, None] ** 2
        + offsets[1][None, :, None] ** 2
        + offsets[2][None, None, :] ** 2
    )
    source_cell = tuple(min(int(s // cell_size), n - 1) for s, n in zip(source, velocity.shape))
    source_velocity = velocity[source_cell]
    radius = SOURCE_RADIUS_CELLS * cell_size
    phi = distance - radius
    inside = phi <= 0.0
    if inside.all():
        return (distance / source_velocity).astype(np.float32)
    times = np.asarray(skfmm.travel_time(phi, velocity, dx=cell_size, order=2), dtype=np.float64)
    times += radius / source_velocity
    times[inside] = distance[inside] / source_velocity
    return times.astype(np.float32)


_WORKER: dict = {}


def _init_worker(velocity: np.ndarray, cell_size: float) -> None:
    _WORKER["velocity"] = velocity
    _WORKER["cell_size"] = cell_size


def _solve_in_worker(source_m) -> np.ndarray:
    return travel_time_field(_WORKER["velocity"], _WORKER["cell_size"], source_m)


def station_travel_time_fields(model: VelocityModel, stations_m, n_workers: int = 1) -> np.ndarray:
    """Travel-time fields ``(n_stations, nx, ny, nz)``, one fast-marching solve per station."""
    stations = np.asarray(stations_m, dtype=np.float64).reshape(-1, 3)
    if len(stations) == 0:
        raise ValueError("At least one station is required")
    fields = np.empty((len(stations),) + model.shape, dtype=np.float32)
    if n_workers <= 1 or len(stations) == 1:
        for index, station in enumerate(stations):
            fields[index] = travel_time_field(model.velocity, model.cell_size, station)
        return fields

    context = mp.get_context("fork")
    with context.Pool(
        min(n_workers, len(stations)),
        initializer=_init_worker,
        initargs=(model.velocity, model.cell_size),
    ) as pool:
        for index, field in enumerate(pool.imap(_solve_in_worker, list(stations))):
            fields[index] = field
    return fields
