"""Create small matched checkerboard/homogeneous inputs for an exploratory K8 study.

This module only writes forward inputs; solve arrivals and run inversions separately.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .generator import CheckerboardConfig, save_checkerboard


CHECKERBOARD_ID = "ambiguity24x18x12_8st_80ev_seed17"
HOMOGENEOUS_ID = "ambiguity_homogeneous24x18x12_8st_80ev_seed17"
CONFIG = CheckerboardConfig(
    lengths_km=(24., 18., 12.), blocks=(4, 3, 2),
    velocities_m_s=(4500., 5500.), surface_stations=(4, 2),
    n_events=80, event_grid_shape=(4, 4, 5), min_depth_km=2.,
    bottom_bias=0., seed=17,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id", nargs="?", help="Override the default ID for this case")
    parser.add_argument("--homogeneous", action="store_true", help="Use a 5000 m/s negative control")
    parser.add_argument("--root", type=Path, default=Path(__file__).parent / "experiments")
    args = parser.parse_args(argv)
    experiment_id = args.experiment_id or (HOMOGENEOUS_ID if args.homogeneous else CHECKERBOARD_ID)
    config = replace(CONFIG, velocities_m_s=(5000., 5000.)) if args.homogeneous else CONFIG
    try:
        destination = save_checkerboard(args.root, experiment_id, config)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
