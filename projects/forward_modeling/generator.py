"""Generate reproducible checkerboard forward-modeling experiment inputs.

This module only creates inputs. The independent solver reads those inputs later.
Coordinates are Cartesian x, y, z (z positive down), metres internally.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from .experiments import save_inputs
from .model import PointSet, VelocityGrid


@dataclass(frozen=True)
class CheckerboardConfig:
    lengths_km: tuple[float, float, float] = (240.0, 120.0, 120.0)
    blocks: tuple[int, int, int] = (4, 2, 2)
    velocities_m_s: tuple[float, float] = (4750.0, 5250.0)
    surface_stations: tuple[int, int] = (14, 7)
    n_events: int = 1000
    event_grid_shape: tuple[int, int, int] | None = None
    min_depth_km: float = 10.0
    bottom_bias: float = 0.15
    seed: int = 42

    def __post_init__(self):
        if len(self.lengths_km) != 3 or not np.all(np.isfinite(self.lengths_km)) or any(
            value <= 0 for value in self.lengths_km
        ):
            raise ValueError("lengths_km must be three finite positive dimensions")
        if len(self.blocks) != 3 or any(type(value) is not int or value < 1 for value in self.blocks):
            raise ValueError("blocks must contain three positive integers")
        sides = np.asarray(self.lengths_km) / self.blocks
        if not np.allclose(sides, sides[0], rtol=1e-12, atol=0):
            raise ValueError("lengths_km / blocks must define cubic velocity cells")
        if len(self.velocities_m_s) != 2 or not np.all(np.isfinite(self.velocities_m_s)) or any(
            value <= 0 for value in self.velocities_m_s
        ):
            raise ValueError("velocities_m_s must contain two finite positive speeds")
        if len(self.surface_stations) != 2 or any(
            type(value) is not int or value < 1 for value in self.surface_stations
        ):
            raise ValueError("surface_stations must contain two positive integers")
        if type(self.n_events) is not int or self.n_events < 1:
            raise ValueError("n_events must be a positive integer")
        if self.event_grid_shape is not None:
            if len(self.event_grid_shape) != 3 or any(
                type(value) is not int or value < 1 for value in self.event_grid_shape
            ):
                raise ValueError("event_grid_shape must contain three positive integers")
            if int(np.prod(self.event_grid_shape)) != self.n_events:
                raise ValueError("n_events must equal the product of event_grid_shape")
        if not np.isfinite(self.min_depth_km) or not 0 < self.min_depth_km < self.lengths_km[2]:
            raise ValueError("min_depth_km must be strictly inside (0, depth)")
        if not np.isfinite(self.bottom_bias) or not 0 <= self.bottom_bias < 1:
            raise ValueError("bottom_bias must be finite and in [0, 1)")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")


def generate_checkerboard(config: CheckerboardConfig = CheckerboardConfig()) -> tuple[VelocityGrid, PointSet, PointSet]:
    """Return a single-wave checkerboard, surface stations and interior events.

    By default, a scrambled Sobol sequence distributes events quasi-uniformly.
    When event_grid_shape is given, hypocentres form an independent Cartesian
    product of cell-centre coordinates in x/y/z (not tied to velocity or FMM
    cells). The optional depth bias warps only the z levels toward the bottom.
    """
    indices = np.indices(config.blocks)
    parity = np.sum(indices, axis=0) % 2
    speed = np.where(parity == 0, config.velocities_m_s[0], config.velocities_m_s[1])
    cell_size_m = config.lengths_km[0] * 1000.0 / config.blocks[0]
    model = VelocityGrid(speed, cell_size_m)

    count_x, count_y = config.surface_stations
    station_x = (np.arange(count_x) + 0.5) * config.lengths_km[0] * 1000 / count_x
    station_y = (np.arange(count_y) + 0.5) * config.lengths_km[1] * 1000 / count_y
    x, y = np.meshgrid(station_x, station_y, indexing="ij")
    station_xyz = np.column_stack((x.ravel(), y.ravel(), np.zeros(count_x * count_y)))
    stations = PointSet(
        tuple(f"STA_{i:0{len(str(count_x * count_y))}d}" for i in range(1, count_x * count_y + 1)),
        station_xyz,
    )

    if config.event_grid_shape is None:
        # Power-of-two Sobol draw retains balanced coverage; truncation only
        # removes the final low-discrepancy samples.
        sampler = qmc.Sobol(d=3, scramble=True, seed=config.seed)
        unit = sampler.random_base2(m=(config.n_events - 1).bit_length())[:config.n_events]
    else:
        levels = [(np.arange(count) + 0.5) / count for count in config.event_grid_shape]
        x_unit, y_unit, z_unit = np.meshgrid(*levels, indexing="ij")
        unit = np.column_stack((x_unit.ravel(), y_unit.ravel(), z_unit.ravel()))
    bias = config.bottom_bias
    # Invert F(t) = (1-b)t + b*t², density p(t) = 1-b+2bt.
    # Rationalized form avoids cancellation as b approaches zero.
    depth_fraction = 2 * unit[:, 2] / (
        1 - bias + np.sqrt((1 - bias) ** 2 + 4 * bias * unit[:, 2])
    )
    event_xyz = np.column_stack((
        unit[:, 0] * config.lengths_km[0] * 1000,
        unit[:, 1] * config.lengths_km[1] * 1000,
        (config.min_depth_km + depth_fraction * (config.lengths_km[2] - config.min_depth_km)) * 1000,
    ))
    events = PointSet(
        tuple(f"EVT_{i:0{len(str(config.n_events))}d}" for i in range(1, config.n_events + 1)),
        event_xyz,
    )
    return model, stations, events


def save_checkerboard(root, experiment_id, config: CheckerboardConfig = CheckerboardConfig()) -> Path:
    """Publish inputs and generator parameters; does not compute any arrivals."""
    model, stations, events = generate_checkerboard(config)
    return save_inputs(root, experiment_id, model, stations, events,
                       generation={"generator": "checkerboard", "parameters": asdict(config)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id", metavar="EXPERIMENT_ID")
    parser.add_argument("--root", type=Path, default=Path(__file__).parent / "experiments")
    parser.add_argument("--lengths-km", type=float, nargs=3, default=(240., 120., 120.), metavar=("X", "Y", "Z"))
    parser.add_argument("--blocks", type=int, nargs=3, default=(4, 2, 2), metavar=("NX", "NY", "NZ"))
    parser.add_argument("--velocities-m-s", type=float, nargs=2, default=(4750., 5250.), metavar=("V1", "V2"))
    parser.add_argument("--surface-stations", type=int, nargs=2, default=(14, 7), metavar=("NX", "NY"))
    parser.add_argument("--events", type=int, default=None,
                        help="Event count (in grid mode defaults to the product of grid dimensions)")
    parser.add_argument("--event-grid-shape", type=int, nargs=3, metavar=("NX", "NY", "NZ"),
                        help="Independent structured hypocentre grid instead of Sobol events")
    parser.add_argument("--min-depth-km", type=float, default=10.)
    parser.add_argument("--bottom-bias", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        config = CheckerboardConfig(
            lengths_km=tuple(args.lengths_km), blocks=tuple(args.blocks),
            velocities_m_s=tuple(args.velocities_m_s), surface_stations=tuple(args.surface_stations),
            n_events=(args.events if args.events is not None else
                      int(np.prod(args.event_grid_shape)) if args.event_grid_shape is not None else 1000),
            event_grid_shape=(tuple(args.event_grid_shape) if args.event_grid_shape is not None else None),
            min_depth_km=args.min_depth_km, bottom_bias=args.bottom_bias, seed=args.seed,
        )
        destination = save_checkerboard(args.root, args.experiment_id, config)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
