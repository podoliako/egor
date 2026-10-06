"""Full checkerboard test with an independent structured hypocentre grid.

The velocity parameterization remains 24 x 12 x 12 and the true model contains
4 x 2 x 2 checkerboard blocks. Hypocentres do not follow the parameterization
cell counts: they occupy their own deterministic 20 x 10 x 7 grid (1,400
events). Horizontal spacing is 12 km along both axes; the seven depth levels
retain the configured downward density bias.

All inversion parameters match ``checkerboard_pattern_4x2x2_full.py``.

Run from the EMTomo directory:
    python -u -m archive.legacy_synthetics.checkerboard_pattern_4x2x2_event_grid
"""
from dataclasses import replace

from .checkerboard_pattern_4x2x2_full import (
    CHECKERBOARD_PATTERN_4X2X2_FULL_CONFIG,
)
from .runner import main


CHECKERBOARD_PATTERN_4X2X2_EVENT_GRID_CONFIG = replace(
    CHECKERBOARD_PATTERN_4X2X2_FULL_CONFIG,
    event_grid_shape=(20, 10, 7),
    synthetic_arrivals_cache=(
        "runs/cache/checkerboard_pattern_4x2x2_event_grid_20x10x7_sub8.npz"
    ),
    run_name="checkerboard_pattern_4x2x2_event_grid_20x10x7",
    run_version="1.0",
)


if __name__ == "__main__":
    main(CHECKERBOARD_PATTERN_4X2X2_EVENT_GRID_CONFIG)
