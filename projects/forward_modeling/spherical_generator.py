"""Generate a 1-D gradient with compact spherical velocity perturbations.

Only experiment inputs are generated here; the independent forward solver reads
and computes their arrivals separately. Coordinates are x/y/z, z positive down.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import numpy as np

from .experiments import load_inputs, save_inputs
from .model import VelocityGrid


@dataclass(frozen=True)
class SphereAnomaly:
    center_km: tuple[float, float, float]
    radius_km: float
    peak_delta_m_s: float

    def __post_init__(self):
        if len(self.center_km) != 3 or not np.all(np.isfinite(self.center_km)):
            raise ValueError("center_km must contain three finite coordinates")
        if not np.isfinite(self.radius_km) or self.radius_km <= 0:
            raise ValueError("radius_km must be finite and positive")
        if not np.isfinite(self.peak_delta_m_s) or self.peak_delta_m_s == 0:
            raise ValueError("peak_delta_m_s must be finite and nonzero")


DEFAULT_ANOMALIES = (
    SphereAnomaly((34., 29., 36.), 14., +360.),
    SphereAnomaly((76., 82., 67.), 21., -320.),
    SphereAnomaly((134., 37., 92.), 18., +520.),
    SphereAnomaly((185., 84., 41.), 26., -440.),
    SphereAnomaly((212., 32., 84.), 12., +270.),
    SphereAnomaly((114., 98., 30.), 16., -260.),
)


@dataclass(frozen=True)
class SphericalConfig:
    lengths_km: tuple[float, float, float] = (240., 120., 120.)
    cell_size_km: float = 2.5
    surface_velocity_m_s: float = 4600.
    bottom_velocity_m_s: float = 5600.
    anomalies: tuple[SphereAnomaly, ...] = DEFAULT_ANOMALIES

    def __post_init__(self):
        if len(self.lengths_km) != 3 or not np.all(np.isfinite(self.lengths_km)) or any(
            value <= 0 for value in self.lengths_km
        ):
            raise ValueError("lengths_km must contain three finite positive dimensions")
        if not np.isfinite(self.cell_size_km) or self.cell_size_km <= 0:
            raise ValueError("cell_size_km must be finite and positive")
        shape = np.asarray(self.lengths_km, dtype=float) / self.cell_size_km
        if not np.allclose(shape, np.rint(shape), rtol=1e-12, atol=0):
            raise ValueError("cell_size_km must tile all domain dimensions")
        if not np.all(np.isfinite((self.surface_velocity_m_s, self.bottom_velocity_m_s))) or min(
            self.surface_velocity_m_s, self.bottom_velocity_m_s
        ) <= 0:
            raise ValueError("background velocities must be finite and positive")
        if not self.anomalies or any(not isinstance(a, SphereAnomaly) for a in self.anomalies):
            raise ValueError("anomalies must be a nonempty sequence of SphereAnomaly")
        for anomaly in self.anomalies:
            center = np.asarray(anomaly.center_km)
            if np.any(center - anomaly.radius_km <= 0) or np.any(
                center + anomaly.radius_km >= self.lengths_km
            ):
                raise ValueError("every sphere must lie strictly within the domain")


def generate_spherical_model(config: SphericalConfig = SphericalConfig()) -> VelocityGrid:
    """Sample a depth gradient plus C1-continuous compact spheres at voxel centres.

    Within radius R, each sphere adds A * (1 + cos(pi * distance/R)) / 2;
    outside R it adds zero. A is the peak perturbation, not a hard step.
    """
    shape = tuple(int(round(length / config.cell_size_km)) for length in config.lengths_km)
    x, y, z = [(np.arange(n) + 0.5) * config.cell_size_km for n in shape]
    background = config.surface_velocity_m_s + (
        (config.bottom_velocity_m_s - config.surface_velocity_m_s) * z / config.lengths_km[2]
    )
    velocity = np.broadcast_to(background, shape).copy()
    for anomaly in config.anomalies:
        cx, cy, cz = anomaly.center_km
        distance_sq = ((x[:, None, None] - cx) ** 2 +
                       (y[None, :, None] - cy) ** 2 +
                       (z[None, None, :] - cz) ** 2)
        inside = distance_sq < anomaly.radius_km ** 2
        velocity[inside] += anomaly.peak_delta_m_s * (
            1 + np.cos(np.pi * np.sqrt(distance_sq[inside]) / anomaly.radius_km)
        ) / 2
    return VelocityGrid(velocity, config.cell_size_km * 1000.)


def save_spherical_experiment(root, experiment_id: str, geometry_experiment_id: str,
                              config: SphericalConfig = SphericalConfig()) -> Path:
    """Copy exact station/event geometry by ID; do not calculate arrivals."""
    root = Path(root)
    reference, stations, events = load_inputs(root, geometry_experiment_id)
    extent_m = np.asarray(reference.velocity.shape) * reference.cell_size_m
    if not np.allclose(extent_m, np.asarray(config.lengths_km) * 1000., rtol=1e-12, atol=0):
        raise ValueError("geometry experiment must have the same domain dimensions")
    if reference.origin_m != (0., 0., 0.):
        raise ValueError("geometry experiment must have origin (0, 0, 0)")
    source = root / "input" / geometry_experiment_id
    geometry_hashes = {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest()
        for name in ("stations.csv", "events.csv")
    }
    return save_inputs(root, experiment_id, generate_spherical_model(config), stations, events,
                       generation={"generator": "spherical_gradient",
                                   "parameters": asdict(config),
                                   "geometry_source": {"id": geometry_experiment_id,
                                                       "sha256": geometry_hashes}})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id")
    parser.add_argument("--geometry-experiment", required=True,
                        help="Existing experiment whose station and event IDs/coordinates to reuse")
    parser.add_argument("--root", type=Path, default=Path(__file__).parent / "experiments")
    parser.add_argument("--cell-size-km", type=float, default=2.5)
    args = parser.parse_args(argv)
    try:
        destination = save_spherical_experiment(
            args.root, args.experiment_id, args.geometry_experiment,
            SphericalConfig(cell_size_km=args.cell_size_km),
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
