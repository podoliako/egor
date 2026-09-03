from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .instruments_coords import (
    MetricPoint,
    _resolve_event_locs_metric,
    _resolve_station_locs_metric,
    metric_to_cell_coord,
    sample_cell_centered_trilinear,
)
from .instruments_travel import compute_station_travel_time_fields


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
    z_offset: float = 0.0,
    slowness_interpolation: str = "nearest",
    arrival_noise_std: float = 0.0,
) -> Tuple[List[List[float]], List[MetricPoint]]:
    """Generate relative synthetic arrivals, optionally with Gaussian pick noise.

    ``arrival_noise_std`` is the per-station standard deviation in seconds.
    Noise is applied to absolute travel times before the first arrival is used as
    the event reference, matching the relative-time format used by inversion.
    """
    if not np.isfinite(arrival_noise_std) or arrival_noise_std < 0.0:
        raise ValueError("arrival_noise_std must be a finite value >= 0")

    geo_grid = model.get_geo_grid(
        subdivision=subdivision,
        slowness_interpolation=slowness_interpolation,
    )
    cell_size = float(geo_grid.cell_size)
    shape = tuple(int(v) for v in geo_grid.shape)
    rng = np.random.default_rng(seed=random_seed)

    _metric_stations, station_idx = _resolve_station_locs_metric(
        shape, cell_size, station_locs, n_stations, rng
    )
    metric_events, _event_idx = _resolve_event_locs_metric(
        shape,
        cell_size,
        event_locs,
        n_events,
        rng,
        depth_bias=depth_bias,
        x_offset=x_offset,
        y_offset=y_offset,
        z_offset=z_offset,
    )

    fields = compute_station_travel_time_fields(
        grid=geo_grid,
        station_locs=station_idx,
        wave_type=wave_type,
        solver=solver,
    ).astype(np.float64, copy=False)

    synthetic: List[List[float]] = []

    for event_metric in metric_events:
        event_coord = metric_to_cell_coord(event_metric, cell_size, shape)
        arrivals_abs = np.asarray(
            [sample_cell_centered_trilinear(field, event_coord) for field in fields],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(arrivals_abs)):
            raise ValueError(
                f"Non-finite travel times for event at metric position {event_metric}. "
                "Check model/solver settings."
            )

        if arrival_noise_std > 0.0:
            arrivals_abs += rng.normal(
                loc=0.0,
                scale=arrival_noise_std,
                size=arrivals_abs.shape,
            )

        t_min = float(np.min(arrivals_abs))
        arrivals_rel = arrivals_abs - t_min

        event_arrivals = [float(arrivals_rel[station_idx_i]) for station_idx_i in range(len(station_idx))]
        synthetic.append(event_arrivals)

    return synthetic, metric_events
