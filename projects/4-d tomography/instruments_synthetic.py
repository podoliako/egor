from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from instruments_coords import (
    MetricPoint,
    _resolve_event_locs_metric,
    _resolve_station_locs_metric,
)
from instruments_travel import compute_station_travel_time_fields


def generate_synthetic_arrivals_table(
    model,
    station_locs: Optional[Sequence[MetricPoint]] = None,
    event_locs: Optional[Sequence[MetricPoint]] = None,
    n_stations: Optional[int] = None,
    n_events: Optional[int] = None,
    wave_type: str = "P",
    solver: Union[str, object] = "skfmm",
    random_seed: Optional[int] = None,
    subdivision: Optional[int] = 1,
    depth_bias: float = 0.0,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
) -> Tuple[List[List[float]], List[MetricPoint]]:
    geo_grid = model.get_geo_grid(subdivision=subdivision)
    cell_size = float(geo_grid.cell_size)
    shape = tuple(int(v) for v in geo_grid.shape)
    rng = np.random.default_rng(seed=random_seed)

    _metric_stations, station_idx = _resolve_station_locs_metric(
        shape, cell_size, station_locs, n_stations, rng
    )
    metric_events, event_idx = _resolve_event_locs_metric(
        shape,
        cell_size,
        event_locs,
        n_events,
        rng,
        depth_bias=depth_bias,
        x_offset=x_offset,
        y_offset=y_offset,
    )

    fields = compute_station_travel_time_fields(
        grid=geo_grid,
        station_locs=station_idx,
        wave_type=wave_type,
        solver=solver,
    ).astype(np.float64, copy=False)

    synthetic: List[List[float]] = []

    for loc_idx, event_loc in enumerate(event_idx):
        i, j, k = event_loc
        arrivals_abs = fields[:, i, j, k]

        if not np.all(np.isfinite(arrivals_abs)):
            raise ValueError(
                f"Non-finite travel times for event {event_loc} "
                f"(metric: {metric_events[loc_idx]}). Check model/solver settings."
            )

        t_min = float(np.min(arrivals_abs))
        arrivals_rel = arrivals_abs - t_min

        event_arrivals = [float(arrivals_rel[station_idx_i]) for station_idx_i in range(len(station_idx))]
        synthetic.append(event_arrivals)

    return synthetic, metric_events
