"""Generate four wavy velocity layers with six compact cosine bubbles.

Only experiment inputs are generated; arrivals are computed by the separate solver.
Coordinates are x/y/z in km, with z positive down.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import numpy as np

from .experiments import load_inputs, save_inputs
from .model import VelocityGrid
from .spherical_generator import DEFAULT_ANOMALIES, SphereAnomaly


DEFAULT_BUBBLES = tuple(
    SphereAnomaly(
        (120., 95., 30.) if index == 5 else
        (anomaly.center_km[0], 80., anomaly.center_km[2]) if index == 3 else
        anomaly.center_km,
        anomaly.radius_km * 1.5,
        anomaly.peak_delta_m_s,
    )
    for index, anomaly in enumerate(DEFAULT_ANOMALIES)
)


@dataclass(frozen=True)
class LayeredBubblesConfig:
    lengths_km: tuple[float, float, float] = (240., 120., 120.)
    cell_size_km: float = 2.5
    layer_velocities_m_s: tuple[float, float, float, float] = (4600., 4850., 5150., 5400.)
    anomalies: tuple[SphereAnomaly, ...] = DEFAULT_BUBBLES

    def __post_init__(self):
        if len(self.lengths_km) != 3 or not np.all(np.isfinite(self.lengths_km)) or any(
            value <= 0 for value in self.lengths_km
        ):
            raise ValueError("lengths_km must contain three finite positive dimensions")
        if not np.isfinite(self.cell_size_km) or self.cell_size_km <= 0:
            raise ValueError("cell_size_km must be finite and positive")
        shape = np.asarray(self.lengths_km) / self.cell_size_km
        if not np.allclose(shape, np.rint(shape), rtol=1e-12, atol=0):
            raise ValueError("cell_size_km must tile all domain dimensions")
        if len(self.layer_velocities_m_s) != 4 or not np.all(np.isfinite(self.layer_velocities_m_s)) or any(
            value <= 0 for value in self.layer_velocities_m_s
        ):
            raise ValueError("layer_velocities_m_s must contain four finite positive speeds")
        if not self.anomalies or any(not isinstance(a, SphereAnomaly) for a in self.anomalies):
            raise ValueError("anomalies must be a nonempty sequence of SphereAnomaly")
        for anomaly in self.anomalies:
            center = np.asarray(anomaly.center_km)
            if np.any(center - anomaly.radius_km <= 0) or np.any(
                center + anomaly.radius_km >= self.lengths_km
            ):
                raise ValueError("every bubble must lie strictly within the domain")
        x, y = [(np.arange(int(round(n))) + 0.5) * self.cell_size_km for n in shape[:2]]
        b1, b2, b3 = layer_boundaries_km(x[:, None], y[None, :])
        if not np.all((0 < b1) & (b1 < b2) & (b2 < b3) & (b3 < self.lengths_km[2])):
            raise ValueError("layer boundaries must be ordered and inside the domain at voxel centres")


def layer_boundaries_km(x, y):
    """Depths of the three interfaces at horizontal coordinates in km."""
    b1 = 30 + 6 * np.sin(2 * np.pi * x / 240) + 4 * np.cos(2 * np.pi * y / 120)
    b2 = 60 + 8 * np.sin(2 * np.pi * x / 240 + 0.8) + 5 * np.sin(2 * np.pi * y / 120 + 0.3)
    b3 = 90 + 6 * np.cos(2 * np.pi * x / 240 - 0.4) - 4 * np.sin(2 * np.pi * y / 120)
    return b1, b2, b3


def generate_layered_bubbles_model(config: LayeredBubblesConfig = LayeredBubblesConfig()) -> VelocityGrid:
    """Sample layers and C1 bubbles at voxel centres (z positive down).

    Each bubble adds A * (1 + cos(pi * r/R)) / 2 for r < R, zero otherwise.
    """
    shape = tuple(int(round(length / config.cell_size_km)) for length in config.lengths_km)
    x, y, z = [(np.arange(n) + 0.5) * config.cell_size_km for n in shape]
    boundaries = layer_boundaries_km(x[:, None], y[None, :])
    velocity = np.full(shape, config.layer_velocities_m_s[0])
    for boundary, speed in zip(boundaries, config.layer_velocities_m_s[1:]):
        velocity = np.where(z[None, None, :] >= boundary[:, :, None], speed, velocity)
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


def save_layered_bubbles_experiment(root, experiment_id: str, geometry_experiment_id: str,
                                    config: LayeredBubblesConfig = LayeredBubblesConfig()) -> Path:
    """Reuse station/event geometry by ID without calculating arrivals."""
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
    return save_inputs(root, experiment_id, generate_layered_bubbles_model(config), stations, events,
                       generation={"generator": "layered_bubbles",
                                   "parameters": asdict(config),
                                   "layer_boundaries_km": [
                                       "30+6*sin(2*pi*x/240)+4*cos(2*pi*y/120)",
                                       "60+8*sin(2*pi*x/240+0.8)+5*sin(2*pi*y/120+0.3)",
                                       "90+6*cos(2*pi*x/240-0.4)-4*sin(2*pi*y/120)",
                                   ],
                                   "geometry_source": {"id": geometry_experiment_id,
                                                       "sha256": geometry_hashes}})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id")
    parser.add_argument("--geometry-experiment", required=True,
                        help="Existing experiment whose station and event IDs/coordinates to reuse")
    parser.add_argument("--root", type=Path, default=Path(__file__).parent / "experiments")
    args = parser.parse_args(argv)
    try:
        destination = save_layered_bubbles_experiment(
            args.root, args.experiment_id, args.geometry_experiment,
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
