"""P-wave checkerboard test using the real 2011 The Geysers geometry.

Run from the EMTomo directory:
    python geysers_checkerboard.py

The geometry is prepared by ``prepare_geysers_geometry.py``.  Arrival times are
synthetic: every selected real hypocentre is paired with every selected real BG
station, as required by EMTomo's dense arrival table.
"""
from dataclasses import replace

from main import CONFIG, main


GEYSERS_CONFIG = replace(
    CONFIG,
    # 20 x 7 x 7 km model, oriented along the compact Geysers footprint.
    cell_size=1_000.0,
    grid_shape=(20, 7, 7),
    lon=-122.77070,
    lat=38.80275,
    height=0.0,
    azimuth=135.0,
    station_locations_csv="data/geysers_2011/stations_metric.csv",
    event_locations_csv="data/geysers_2011/events_metric_sample_300.csv",
    # A 2 km checkerboard is compatible with the 1 km inversion cells.
    checkerboard_cell_size=2_000.0,
    checkerboard_rotation_degrees=0.0,
    run_name="geysers_2011_p_checkerboard",
    run_version="1.0",
    runs_dir="runs",
)


if __name__ == "__main__":
    main(GEYSERS_CONFIG)
