"""Generate horizontal layers over a dipping lower interface, with compact spheres.

Stations form a regular surface grid and hypocentres are scrambled-Sobol
uniform in the volume. Only experiment inputs are written; arrivals are
computed by the separate solver. Coordinates are x/y/z in km, z positive down.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
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
    # Regular hypocentre lattice (cell-centre fractions of x, y and the depth
    # range) instead of Sobol points; n_events must equal its size.
    event_grid_shape: tuple[int, int, int] | None = None

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
        if self.event_grid_shape is not None and (
            len(self.event_grid_shape) != 3
            or any(type(n) is not int or n < 1 for n in self.event_grid_shape)
            or int(np.prod(self.event_grid_shape)) != self.n_events
        ):
            raise ValueError("event_grid_shape must hold three positive integers whose product is n_events")


# Same structure at the scale of the Geysers BG network box (20 x 7 x 7 km, see
# EMTomo/data/geysers_2011), so that real stations and events can later be used
# with an identical model. The dipping interface lies within the well-sampled
# 0-5 km depth range of the catalogue.
GEYSERS_SCALE_ANOMALIES = (
    SphereAnomaly((3.5, 2.5, 1.2), 0.9, +600., 0.5),
    SphereAnomaly((8.0, 4.8, 2.0), 1.1, -650., 0.5),
    SphereAnomaly((12.5, 2.4, 2.6), 1.2, +700., 0.5),
    SphereAnomaly((16.5, 4.5, 1.6), 1.0, -600., 0.5),
    SphereAnomaly((6.0, 2.2, 3.6), 1.1, +700., 0.5),
    SphereAnomaly((11.0, 5.0, 4.0), 1.0, -650., 0.5),
)

# Four flat layers (3800/4600/5200/5700 m/s; interfaces 0.7, 1.9 and 3.4 km,
# off the faces of 1, 0.5 and 1/3 km inversion cells), the same spheres, a
# 13 x 4 station grid and a 24 x 8 x 10 hypocentre lattice whose coordinates
# avoid those faces as well.
GEYSERS_FLAT = DippingLayersConfig(
    lengths_km=(20., 7., 7.),
    cell_size_km=0.125,
    horizontal_boundaries_km=(0.7, 1.9),
    dipping_boundary_km=(3.4, 3.4),
    layer_velocities_m_s=(3800., 4600., 5200., 5700.),
    anomalies=GEYSERS_SCALE_ANOMALIES,
    surface_stations=(13, 4),
    n_events=1920,
    event_depth_km=(0.5, 5.5),
    seed=42,
    event_grid_shape=(24, 8, 10),
)

PRESETS = {
    "deep240": DippingLayersConfig(),
    "geysers20": DippingLayersConfig(
        lengths_km=(20., 7., 7.),
        cell_size_km=0.125,
        horizontal_boundaries_km=(0.6, 1.3),
        dipping_boundary_km=(2.0, 3.5),
        layer_velocities_m_s=(4000., 4500., 5000., 5500.),
        anomalies=GEYSERS_SCALE_ANOMALIES,
        surface_stations=(9, 3),
        n_events=1000,
        event_depth_km=(0.5, 5.5),
        seed=42,
    ),
    "geysers20flat": GEYSERS_FLAT,
}


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
    if config.event_grid_shape is None:
        unit = sobol_unit_points(config.n_events, config.seed)
    else:
        levels = [(np.arange(n) + 0.5) / n for n in config.event_grid_shape]
        unit = np.column_stack([axis.ravel() for axis in np.meshgrid(*levels, indexing="ij")])
    events = events_from_unit_points(unit, config.lengths_km, config.event_depth_km)
    return generate_dipping_layers_model(config), stations, events


def save_dipping_layers(root, experiment_id: str,
                        config: DippingLayersConfig = DippingLayersConfig(),
                        stations: PointSet | None = None, events: PointSet | None = None,
                        geometry_source: dict | None = None) -> Path:
    """Publish inputs and generator parameters; does not compute any arrivals.

    ``stations``/``events`` replace the generated uniform geometry (for example a
    real network); ``geometry_source`` is then recorded in generation.json.
    """
    model, uniform_stations, uniform_events = generate_dipping_layers(config)
    top, bottom = config.dipping_boundary_km
    generation = {
        "generator": "dipping_layers",
        "parameters": asdict(config),
        "dipping_boundary_km": f"{top:g} + {bottom - top:g} * x / {config.lengths_km[0]:g}",
        "anomaly_profile": "A within plateau_fraction*R, (1+cos(pi*(r-pR)/(R-pR)))/2 taper to R",
    }
    if stations is not None or events is not None:
        generation["geometry_source"] = geometry_source or "external"
    return save_inputs(root, experiment_id, model,
                       uniform_stations if stations is None else stations,
                       uniform_events if events is None else events, generation=generation)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id", metavar="EXPERIMENT_ID")
    parser.add_argument("--root", type=Path, default=Path(__file__).parent / "experiments")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="deep240")
    parser.add_argument("--surface-stations", type=int, nargs=2, metavar=("NX", "NY"))
    parser.add_argument("--events", type=int)
    parser.add_argument("--event-depth-km", type=float, nargs=2, metavar=("TOP", "BOTTOM"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    try:
        overrides = {name: value for name, value in (
            ("surface_stations", args.surface_stations and tuple(args.surface_stations)),
            ("n_events", args.events),
            ("event_depth_km", args.event_depth_km and tuple(args.event_depth_km)),
            ("seed", args.seed),
        ) if value is not None}
        config = replace(PRESETS[args.preset], **overrides)
        destination = save_dipping_layers(args.root, args.experiment_id, config)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
