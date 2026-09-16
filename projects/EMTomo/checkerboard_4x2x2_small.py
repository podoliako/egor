"""Reduced proportional version of the 4 x 2 x 2 checkerboard test.

The anomaly size in coarse cells and the event/station spatial densities are
kept close to the full 24 x 12 x 12 scenario, while the model volume is reduced
to make iteration times practical.

Geometry
--------
- inversion grid: 16 x 8 x 8 coarse cells (10 km per cell);
- checkerboard blocks: 4 x 2 x 2 coarse cells (4 x 4 x 4 blocks);
- true Vp: 4.75/5.25 km/s, initial Vp: homogeneous 5.0 km/s;
- stations: 10 x 5 = 50, regularly spaced on the top surface;
- events: 16 x 8 x 4 = 512, regular horizontally and denser at depth;
- forward/inversion subdivision: 2;
- synthetic arrival noise: disabled.

Run from the EMTomo directory:
    python checkerboard_4x2x2_small.py
"""
from dataclasses import replace

from checkerboard_4x2x2 import CHECKERBOARD_4X2X2_CONFIG
from main import main


CHECKERBOARD_4X2X2_SMALL_CONFIG = replace(
    CHECKERBOARD_4X2X2_CONFIG,
    grid_shape=(16, 8, 8),
    station_grid_shape=(10, 5),
    event_grid_shape=(16, 8, 4),
    subdivision=2,
    n_workers=24,
    run_name="checkerboard_4x2x2_small",
    run_version="1.0",
)


if __name__ == "__main__":
    main(CHECKERBOARD_4X2X2_SMALL_CONFIG)
