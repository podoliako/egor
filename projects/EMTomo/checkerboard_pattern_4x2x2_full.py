"""Full 24 x 12 x 12 model with a 4 x 2 x 2 checkerboard pattern.

Here ``4 x 2 x 2`` is the number of checkerboard blocks along x, y, and z,
not the number of parameterization cells inside one anomaly. Consequently each
checkerboard block spans 6 x 6 x 6 coarse cells (60 x 60 x 60 km).

Coverage is scaled from the successful reduced 16 x 8 x 8 experiment:
- stations: 14 x 7 = 98 on the top surface;
- events: 24 x 12 x 6 = 1,728, with the same downward depth bias;
- subdivision: 8;
- 14 EM cycles, 0.5% velocity trust region, lambda=0.05, coverage power=1.5;
- noise-free synthetic arrivals.

Run from the EMTomo directory:
    python -u checkerboard_pattern_4x2x2_full.py
"""
from dataclasses import replace

from checkerboard_4x2x2_sub8_final import (
    CHECKERBOARD_4X2X2_SUB8_FINAL_CONFIG,
)
from main import main


CHECKERBOARD_PATTERN_4X2X2_FULL_CONFIG = replace(
    CHECKERBOARD_4X2X2_SUB8_FINAL_CONFIG,
    grid_shape=(24, 12, 12),
    station_grid_shape=(14, 7),
    event_grid_shape=(24, 12, 6),
    checkerboard_block_shape=None,
    checkerboard_pattern_shape=(4, 2, 2),
    synthetic_arrivals_cache=(
        "runs/cache/checkerboard_pattern_4x2x2_full_sub8_no_noise.npz"
    ),
    n_workers=24,
    run_name="checkerboard_pattern_4x2x2_full_sub8_cov15_14cycles",
    run_version="1.0",
)


if __name__ == "__main__":
    main(CHECKERBOARD_PATTERN_4X2X2_FULL_CONFIG)
