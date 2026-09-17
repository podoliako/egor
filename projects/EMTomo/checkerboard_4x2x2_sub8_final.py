"""Final 14-cycle subdivision-8 checkerboard experiment.

This extends the conservative suite configuration with moderately reduced
coverage damping to observe longer-term convergence without regenerating the
shared noise-free synthetic arrivals.

Run from the EMTomo directory:
    python -u checkerboard_4x2x2_sub8_final.py
"""
from dataclasses import replace

from checkerboard_4x2x2_sub8_suite import SUB8_BASE_CONFIG
from main import main


CHECKERBOARD_4X2X2_SUB8_FINAL_CONFIG = replace(
    SUB8_BASE_CONFIG,
    n_cycles=14,
    max_velocity_step_fraction=0.005,
    lambda_reg=0.05,
    coverage_damping_power=1.5,
    run_name="checkerboard_4x2x2_small_sub8_cov15_14cycles",
    run_version="1.0",
)


if __name__ == "__main__":
    main(CHECKERBOARD_4X2X2_SUB8_FINAL_CONFIG)
