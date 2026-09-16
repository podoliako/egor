"""First synthetic test in the EMTomo checkerboard series.

Geometry
--------
- inversion grid: 24 x 12 x 12 coarse cells (10 km per cell);
- checkerboard blocks: 4 x 2 x 2 coarse cells;
- true Vp: 4.75/5.25 km/s, initial Vp: homogeneous 5.0 km/s;
- stations: 14 x 7 = 98, regularly spaced on the top surface;
- events: 24 x 12 x 6 = 1,728, regular horizontally and denser at depth;
- synthetic arrival noise: disabled.

Run from the EMTomo directory:
    python checkerboard_4x2x2.py
"""
from dataclasses import replace

from main import CONFIG, main


CHECKERBOARD_4X2X2_CONFIG = replace(
    CONFIG,
    cell_size=10_000.0,
    grid_shape=(24, 12, 12),
    station_grid_shape=(14, 7),
    background_vp=5_000.0,
    checkerboard_anomaly_fraction=0.05,
    checkerboard_block_shape=(4, 2, 2),
    checkerboard_rotation_degrees=0.0,
    event_grid_shape=(24, 12, 6),
    event_depth_bias=0.5,
    arrival_noise_std=0.0,
    run_name="checkerboard_4x2x2",
    run_version="1.0",
)


if __name__ == "__main__":
    main(CHECKERBOARD_4X2X2_CONFIG)
