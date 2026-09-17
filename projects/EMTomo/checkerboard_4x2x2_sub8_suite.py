"""Unattended subdivision-8 parameter suite for the reduced checkerboard model.

The first run keeps the previous inversion parameters. The following runs test
progressively more conservative velocity updates and regularization because the
subdivision-2 experiment started diverging after its second iteration.

Run from the EMTomo directory:
    python -u checkerboard_4x2x2_sub8_suite.py
"""
from dataclasses import replace

from checkerboard_4x2x2_small import CHECKERBOARD_4X2X2_SMALL_CONFIG
from main import main


ARRIVALS_CACHE = "runs/cache/checkerboard_4x2x2_small_sub8_no_noise.npz"

SUB8_BASE_CONFIG = replace(
    CHECKERBOARD_4X2X2_SMALL_CONFIG,
    subdivision=8,
    synthetic_arrivals_cache=ARRIVALS_CACHE,
    run_name="checkerboard_4x2x2_small_sub8_baseline",
    run_version="1.0",
)

EXPERIMENTS = (
    SUB8_BASE_CONFIG,
    replace(
        SUB8_BASE_CONFIG,
        max_velocity_step_fraction=0.01,
        run_name="checkerboard_4x2x2_small_sub8_step01",
    ),
    replace(
        SUB8_BASE_CONFIG,
        max_velocity_step_fraction=0.01,
        lambda_reg=0.1,
        run_name="checkerboard_4x2x2_small_sub8_step01_lambda01",
    ),
    replace(
        SUB8_BASE_CONFIG,
        max_velocity_step_fraction=0.005,
        lambda_reg=0.05,
        coverage_damping_power=2.0,
        run_name="checkerboard_4x2x2_small_sub8_conservative",
    ),
)


if __name__ == "__main__":
    for index, config in enumerate(EXPERIMENTS, start=1):
        print(
            f"=== Experiment {index}/{len(EXPERIMENTS)}: {config.run_name} ===",
            flush=True,
        )
        main(config)
