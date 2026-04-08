"""
Local utilities (public facade).
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Union

from instruments_coords import GridPoint, MetricPoint, metric_to_index
from instruments_ops import coarsen_G
from instruments_synthetic import generate_synthetic_arrivals_table
from instruments_travel import compute_station_travel_time_fields
from instruments_weights import (
    compute_cellwise_pairwise_misfit,
    compute_epicenter_weight_matrix,
)

StationArrival = Dict[str, Union[Tuple[int, int, int], float, int]]
SyntheticEventArrivals = List[float]
ArrivalTable = List[SyntheticEventArrivals]

__all__ = [
    "ArrivalTable",
    "GridPoint",
    "MetricPoint",
    "StationArrival",
    "SyntheticEventArrivals",
    "coarsen_G",
    "compute_cellwise_pairwise_misfit",
    "compute_epicenter_weight_matrix",
    "compute_station_travel_time_fields",
    "generate_synthetic_arrivals_table",
    "metric_to_index",
]
