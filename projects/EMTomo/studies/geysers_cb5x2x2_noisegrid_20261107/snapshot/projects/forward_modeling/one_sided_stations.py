"""Generate and preview a surface-only network for the 24 x 12 km nx6 domain.

No events, velocity models, arrival times, or inversions are generated.
Run from the repository root with:
    python -m projects.forward_modeling.one_sided_stations --output-dir OUTPUT_DIR
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .model import PointSet


DOMAIN_KM = (24.0, 12.0)
X_RANGE_KM = (1.0, 5.0)
Y_RANGE_KM = (1.0, 11.0)
GRID_SHAPE = (3, 6)
JITTER_FRACTION = 0.25


def generate_stations(seed: int = 42) -> PointSet:
    """One station per tile; jitter is at most 25% of each tile's width.

    Stratification preserves aperture and separation without a perfectly regular
    lattice. Coordinates use metres, with z positive down and surface z = 0.
    """
    rng = np.random.default_rng(seed)
    nx, ny = GRID_SHAPE
    lower = np.array([X_RANGE_KM[0], Y_RANGE_KM[0]])
    tile_size = (np.array([X_RANGE_KM[1], Y_RANGE_KM[1]]) - lower) / [nx, ny]
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    tiles = np.column_stack((ix.ravel(), iy.ravel()))
    jitter = rng.uniform(-JITTER_FRACTION, JITTER_FRACTION, size=tiles.shape)
    xy_km = lower + (tiles + 0.5 + jitter) * tile_size
    return PointSet(
        tuple(f"STA_{i:02d}" for i in range(1, nx * ny + 1)),
        np.column_stack((xy_km * 1000.0, np.zeros(nx * ny))),
    )


def plot_stations(stations: PointSet, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    xy = stations.coordinates_m[:, :2] / 1000.0
    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    ax.add_patch(Rectangle((0, 0), *DOMAIN_KM, facecolor="#f7f7f7",
                           edgecolor="0.25", linewidth=1.2))
    ax.add_patch(Rectangle((X_RANGE_KM[0], Y_RANGE_KM[0]),
                           X_RANGE_KM[1] - X_RANGE_KM[0],
                           Y_RANGE_KM[1] - Y_RANGE_KM[0],
                           facecolor="#dceaf6", edgecolor="#6787a3",
                           linestyle="--", label="Station placement window"))
    # Only block boundaries, not a depth slice of the velocity truth.
    for x in np.arange(4.0, DOMAIN_KM[0], 4.0):
        ax.plot([x, x], [0, DOMAIN_KM[1]], color="0.8", linewidth=0.8, zorder=1)
    for y in np.arange(4.0, DOMAIN_KM[1], 4.0):
        ax.plot([0, DOMAIN_KM[0]], [y, y], color="0.8", linewidth=0.8, zorder=1)
    ax.scatter(*xy.T, marker="^", s=75, color="#b23a35", edgecolors="white",
               linewidths=0.7, zorder=3, label=f"{len(stations.ids)} stations, z = 0")
    for identifier, (x, y) in zip(stations.ids, xy):
        ax.annotate(identifier, (x, y), xytext=(5, 5), textcoords="offset points",
                    fontsize=7, zorder=4)
    ax.set(xlim=(-0.3, DOMAIN_KM[0] + 0.3), ylim=(-0.3, DOMAIN_KM[1] + 0.3),
           xlabel="x (km)", ylabel="y (km)",
           title="One-sided station network — top view\n24 × 12 km domain; nx6 block boundaries every 4 km")
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(0, DOMAIN_KM[0] + 1, 2))
    ax.set_yticks(np.arange(0, DOMAIN_KM[1] + 1, 2))
    ax.legend(loc="upper right")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    paths = [args.output_dir / name for name in
             ("stations.csv", "stations_top_view.png", "generation.json")]
    if any(path.exists() for path in paths):
        parser.error("Output files already exist; choose a new --output-dir")
    stations = generate_stations(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with paths[0].open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("station_id", "x_m", "y_m", "z_m"))
        writer.writerows((identifier, *xyz) for identifier, xyz in
                         zip(stations.ids, stations.coordinates_m))
    plot_stations(stations, paths[1])
    with paths[2].open("x", encoding="utf-8") as stream:
        json.dump({"generator": "one_sided_stations", "seed": args.seed,
                   "domain_xy_km": DOMAIN_KM, "x_range_km": X_RANGE_KM,
                   "y_range_km": Y_RANGE_KM, "grid_shape": GRID_SHAPE,
                   "jitter_fraction": JITTER_FRACTION, "surface_z_m": 0.0,
                   "coordinate_units": "m"}, stream, indent=2)
        stream.write("\n")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
