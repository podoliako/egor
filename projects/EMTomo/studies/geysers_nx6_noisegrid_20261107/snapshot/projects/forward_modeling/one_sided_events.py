"""Generate a stratified event catalog and XY/XZ/YZ projections of all points.

Uses an existing stations.csv without changing it. Coordinates are metres,
with z positive down. Does not generate models, arrivals, or inversions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .model import PointSet


DOMAIN_KM = (24.0, 12.0, 12.0)
GROUPS = (
    {"name": "far", "lower_km": (10.0, 1.5, 3.0),
     "upper_km": (21.0, 10.5, 10.5), "grid_shape": (10, 10, 9)},
    {"name": "near", "lower_km": (3.0, 1.5, 2.0),
     "upper_km": (9.0, 10.5, 8.0), "grid_shape": (6, 10, 5)},
)
PROJECTIONS = (("xy", 0, 1, "Top view"),
               ("xz", 0, 2, "Side view along y"),
               ("yz", 1, 2, "Side view along x"))


def generate_events(seed: int = 42) -> PointSet:
    """One uniform random point per 3D stratum: 900 far and 300 near events."""
    rng = np.random.default_rng(seed)
    coordinates = []
    for group in GROUPS:
        shape = np.asarray(group["grid_shape"])
        tiles = np.indices(shape).reshape(3, -1).T
        unit = (tiles + rng.random(tiles.shape)) / shape
        lower = np.asarray(group["lower_km"])
        upper = np.asarray(group["upper_km"])
        coordinates.append((lower + unit * (upper - lower)) * 1000.0)
    xyz = np.concatenate(coordinates)
    return PointSet(tuple(f"EVT_{i:04d}" for i in range(1, len(xyz) + 1)), xyz)


def load_stations(path: Path) -> PointSet:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["station_id", "x_m", "y_m", "z_m"]:
            raise ValueError("Expected station_id,x_m,y_m,z_m CSV columns")
        rows = list(reader)
    stations = PointSet(tuple(row["station_id"] for row in rows),
                        [[float(row[key]) for key in ("x_m", "y_m", "z_m")]
                         for row in rows])
    xyz = stations.coordinates_m
    if np.any(xyz < 0) or np.any(xyz > np.asarray(DOMAIN_KM) * 1000):
        raise ValueError("Stations must be inside the 24 x 12 x 12 km domain")
    return stations


def plot_projection(events: PointSet, stations: PointSet, projection, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    _, horizontal, vertical, title = projection
    extent_x, extent_y = DOMAIN_KM[horizontal], DOMAIN_KM[vertical]
    xyz = events.coordinates_m / 1000.0
    station_xyz = stations.coordinates_m / 1000.0
    fig, ax = plt.subplots(figsize=(11 if extent_x == 24 else 7, 6.5),
                           constrained_layout=True)
    ax.add_patch(Rectangle((0, 0), extent_x, extent_y, facecolor="#fafafa",
                           edgecolor="0.25", linewidth=1.2))
    for x in np.arange(4, extent_x, 4):
        ax.plot([x, x], [0, extent_y], color="0.85", linewidth=0.8, zorder=1)
    for y in np.arange(4, extent_y, 4):
        ax.plot([0, extent_x], [y, y], color="0.85", linewidth=0.8, zorder=1)
    ax.scatter(xyz[:, horizontal], xyz[:, vertical], s=7, color="black",
               linewidths=0, zorder=2, label=f"{len(events.ids)} events")
    ax.scatter(station_xyz[:, horizontal], station_xyz[:, vertical], marker="^",
               s=65, color="#b23a35", edgecolors="white", linewidths=0.6,
               zorder=3, label=f"{len(stations.ids)} stations")
    labels = ("x (km)", "y (km)", "Depth z (km)")
    ax.set(xlim=(-0.4, extent_x + 0.4), ylim=(-0.5, extent_y + 0.4),
           xlabel=labels[horizontal], ylabel=labels[vertical],
           title=f"{title} — projection of all events\n900 far + 300 near; nx6 block boundaries every 4 km")
    if vertical == 2:
        ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(0, extent_x + 1, 2))
    ax.set_yticks(np.arange(0, extent_y + 1, 2))
    ax.legend(loc="lower right" if vertical == 2 else "upper right", fontsize=9)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stations-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    output = args.output_dir
    paths = [output / "events.csv", output / "events_generation.json"]
    paths += [output / f"geometry_{name}.png" for name, *_ in PROJECTIONS]
    if any(path.exists() for path in paths):
        parser.error("Output files already exist; choose a new --output-dir")
    stations = load_stations(args.stations_csv)
    events = generate_events(args.seed)
    output.mkdir(parents=True, exist_ok=True)
    with paths[0].open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("event_id", "x_m", "y_m", "z_m"))
        writer.writerows((identifier, *xyz) for identifier, xyz in
                         zip(events.ids, events.coordinates_m))
    groups = []
    start = 0
    for group in GROUPS:
        count = int(np.prod(group["grid_shape"]))
        groups.append({**group, "count": count, "first_event_id": events.ids[start],
                       "last_event_id": events.ids[start + count - 1]})
        start += count
    metadata = {"generator": "one_sided_events", "seed": args.seed,
                "domain_km": DOMAIN_KM, "coordinate_units": "m", "z_positive": "down",
                "sampling": "one independent uniform point per 3D stratum",
                "groups": groups, "stations_source": str(args.stations_csv),
                "stations_sha256": hashlib.sha256(args.stations_csv.read_bytes()).hexdigest(),
                "figures": "Orthographic projections of all points, not thin slices"}
    with paths[1].open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    for projection, path in zip(PROJECTIONS, paths[2:]):
        plot_projection(events, stations, projection, path)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
