"""
Local utilities (public facade).
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Union

from .instruments_coords import (
    CellCoord,
    CellIndex,
    GridPoint,
    MetricPoint,
    cell_coord_bounds,
    cell_coord_to_metric,
    cell_index_to_metric_center,
    metric_to_cell_coord,
    metric_to_cell_index,
    metric_to_index,
    sample_cell_centered_trilinear,
    sample_cell_centered_trilinear_batch,
    snap_metric_points_to_cell_centers,
)
from .instruments_ops import coarsen_G
from .instruments_synthetic import generate_synthetic_arrivals_table
from .instruments_travel import compute_station_travel_time_fields
from .instruments_weights import (
    compute_cellwise_pairwise_misfit,
    compute_epicenter_weight_matrix,
    compute_weights_from_misfit,
)

StationArrival = Dict[str, Union[Tuple[int, int, int], float, int]]
SyntheticEventArrivals = List[float]
ArrivalTable = List[SyntheticEventArrivals]

__all__ = [
    "ArrivalTable",
    "CellCoord",
    "CellIndex",
    "GridPoint",
    "MetricPoint",
    "StationArrival",
    "SyntheticEventArrivals",
    "cell_coord_bounds",
    "cell_coord_to_metric",
    "cell_index_to_metric_center",
    "coarsen_G",
    "compute_cellwise_pairwise_misfit",
    "compute_epicenter_weight_matrix",
    "compute_station_travel_time_fields",
    "compute_weights_from_misfit",
    "generate_synthetic_arrivals_table",
    "metric_to_cell_coord",
    "metric_to_cell_index",
    "metric_to_index",
    "sample_cell_centered_trilinear",
    "sample_cell_centered_trilinear_batch",
    "snap_metric_points_to_cell_centers",
]
