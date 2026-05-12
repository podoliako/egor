from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np

from wave_propagation import WavePropagator


def compute_station_travel_time_fields(
    grid,
    station_locs: Sequence[Tuple[int, int, int]],
    wave_type: str,
    solver: Union[str, object],
) -> np.ndarray:
    if len(station_locs) == 0:
        raise ValueError("At least one station is required")

    propagator = WavePropagator(solver=solver)
    fields = []
    for loc in station_locs:
        travel = propagator.compute_from_geo_grid(
            geo_grid=grid,
            source_idx=loc,
            wave_type=wave_type,
        )
        fields.append(travel.astype(np.float32))
    return np.stack(fields, axis=0)
