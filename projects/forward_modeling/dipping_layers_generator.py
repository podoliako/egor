"""Generate horizontal layers over a dipping lower interface, with compact spheres.

Stations form a regular surface grid and hypocentres are scrambled-Sobol
uniform in the volume. Only experiment inputs are written; arrivals are
computed by the separate solver. Coordinates are x/y/z in km, z positive down.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .experiments import save_inputs
from .generator import events_from_unit_points, sobol_unit_points, surface_station_grid
from .model import PointSet, VelocityGrid
from .spherical_generator import SphereAnomaly, add_sphere_anomalies, validate_anomalies_inside

DEFAULT_ANOMALIES = (
    SphereAnomaly((40., 30., 24.), 18., +700., 0.5),
    SphereAnomaly((105., 85., 30.), 20., -700., 0.5),
    SphereAnomaly((170., 35., 45.), 22., +750., 0.5),
    SphereAnomaly((205., 90., 70.), 22., -750., 0.5),
    SphereAnomaly((70., 80., 75.), 20., +800., 0.5),
    SphereAnomaly((135., 55., 90.), 18., -700., 0.5),
)


@dataclass(frozen=True)
class DippingLayersConfig:
    lengths_km: tuple[float, float, float] = (240., 120., 120.)
    cell_size_km: float = 2.5
    horizontal_boundaries_km: tuple[float, ...] = (20., 45.)
    # Depth of the lowest interface at x = 0 and at x = lengths_km[0].
    dipping_boundary_km: tuple[float, float] = (65., 95.)
    layer_velocities_m_s: tuple[float, ...] = (4600., 4900., 5200., 5600.)
    anomalies: tuple[SphereAnomaly, ...] = DEFAULT_ANOMALIES
    surface_stations: tuple[int, int] = (12, 6)
    n_events: int = 1000
    event_depth_km: tuple[float, float] = (5., 115.)
    seed: int = 42

    def __post_init__(self):
        lengths = np.asarray(self.lengths_km, dtype=float)
        if lengths.shape != (3,) or not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
            raise ValueError("lengths_km must contain three finite positive dimensions")
        shape = lengths / self.cell_size_km
        if not np.isfinite(self.cell_size_km) or self.cell_size_km <= 0 or not np.allclose(
            shape, np.rint(shape), rtol=1e-12, atol=0
        ):
            raise ValueError("cell_size_km must be positive and tile all domain dimensions")
        boundaries = np.asarray(self.horizontal_boundaries_km, dtype=float)
        dip = np.asarray(self.dipping_boundary_km, dtype=float)
        if dip.shape != (2,) or boundaries.ndim != 1 or not np.all(np.isfinite(boundaries)) \
                or not np.all(np.isfinite(dip)):
            raise ValueError("boundaries must be finite; dipping_boundary_km has two depths")
        if np.any(np.diff(boundaries) <= 0) or (boundaries.size and boundaries[0] <= 0) \
                or (boundaries.size and dip.min() <= boundaries[-1]) or dip.max() >= lengths[2]:
            raise ValueError("horizontal boundaries must increase and lie above the dipping one")
        velocities = np.asarray(self.layer_velocities_m_s, dtype=float)
        if velocities.shape != (boundaries.size + 2,) or not np.all(np.isfinite(velocities)) \
                or np.any(velocities <= 0):
            raise ValueError("layer_velocities_m_s needs one positive speed per layer")
        validate_anomalies_inside(self.anomalies, self.lengths_km)
        if len(self.surface_stations) != 2 or any(
            type(value) is not int or value < 1 for value in self.surface_stations
        ):
            raise ValueError("surface_stations must contain two positive integers")
        if type(self.n_events) is not int or self.n_events < 1:
            raise ValueError("n_events must be a positive integer")
        top, bottom = self.event_depth_km
        if not (np.isfinite(top) and np.isfinite(bottom) and 0 < top < bottom < lengths[2]):
            raise ValueError("event_depth_km must satisfy 0 < top < bottom < depth")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")


def dipping_boundary_depth_km(config: DippingLayersConfig, x_km):
    top, bottom = config.dipping_boundary_km
    return top + (bottom - top) * np.asarray(x_km) / config.lengths_km[0]


def generate_dipping_layers_model(config: DippingLayersConfig = DippingLayersConfig()) -> VelocityGrid:
    """Layer velocities sampled at voxel centres, plus the anomalies."""
    shape = tuple(int(round(length / config.cell_size_km)) for length in config.lengths_km)
    x, y, z = [(np.arange(n) + 0.5) * config.cell_size_km for n in shape]
    speeds = config.layer_velocities_m_s
    layer = np.searchsorted(np.asarray(config.horizontal_boundaries_km), z, side="right")
    velocity = np.broadcast_to(np.asarray(speeds)[layer], shape).copy()
    below_dip = z[None, None, :] >= dipping_boundary_depth_km(config, x)[:, None, None]
    velocity = np.where(below_dip, speeds[-1], velocity)
    velocity = add_sphere_anomalies(velocity, x, y, z, config.anomalies)
    return VelocityGrid(velocity, config.cell_size_km * 1000.)


def generate_dipping_layers(
    config: DippingLayersConfig = DippingLayersConfig(),
) -> tuple[VelocityGrid, PointSet, PointSet]:
    stations = surface_station_grid(config.lengths_km, config.surface_stations)
    events = events_from_unit_points(
        sobol_unit_points(config.n_events, config.seed), config.lengths_km, config.event_depth_km,
    )
    return generate_dipping_layers_model(config), stations, events


def save_dipping_layers(root, experiment_id: str,
                        config: DippingLayersConfig = DippingLayersConfig()) -> Path:
    """Publish inputs and generator parameters; does not compute any arrivals."""
    model, stations, events = generate_dipping_layers(config)
    top, bottom = config.dipping_boundary_km
    return save_inputs(root, experiment_id, model, stations, events, generation={
        "generator": "dipping_layers",
        "parameters": asdict(config),
        "dipping_boundary_km": f"{top:g} + {bottom - top:g} * x / {config.lengths_km[0]:g}",
        "anomaly_profile": "A within plateau_fraction*R, (1+cos(pi*(r-pR)/(R-pR)))/2 taper to R",
    })


def main(argv=None):
    defaults = DippingLayersConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id", metavar="EXPERIMENT_ID")
    parser.add_argument("--root", type=Path, default=Path(__file__).parent / "experiments")
    parser.add_argument("--surface-stations", type=int, nargs=2, default=defaults.surface_stations,
                        metavar=("NX", "NY"))
    parser.add_argument("--events", type=int, default=defaults.n_events)
    parser.add_argument("--event-depth-km", type=float, nargs=2, default=defaults.event_depth_km,
                        metavar=("TOP", "BOTTOM"))
    parser.add_argument("--seed", type=int, default=defaults.seed)
    args = parser.parse_args(argv)
    try:
        config = DippingLayersConfig(
            surface_stations=tuple(args.surface_stations), n_events=args.events,
            event_depth_km=tuple(args.event_depth_km), seed=args.seed,
        )
        destination = save_dipping_layers(args.root, args.experiment_id, config)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
