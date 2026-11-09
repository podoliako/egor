"""Preview a cropped nx6 model with bottom events, two blobs and end stations.

Only generates geometry and projection figures, not arrivals or inversions.
Coordinates in saved inputs are metres, with z positive down.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .model import PointSet, VelocityGrid


DOMAIN_KM = (24.0, 8.0, 8.0)
BLOB_CENTERS_KM = ((6.0, 4.0, 4.0), (18.0, 4.0, 4.0))
BLOB_SIGMA_KM = 0.45
SOURCE_ID = "checkerboard_24x12x12_nx6_98st_1200ev_grid20x10x6_minz3_noise01_pick001_seed42"
VIEWS = (("xy", 0, 1, "Top view: XY"),
         ("xz", 0, 2, "Vertical view: XZ (looking along y)"),
         ("yz", 1, 2, "Vertical view: YZ (looking along x)"))


def generate_geometry(seed: int = 42) -> tuple[PointSet, PointSet]:
    rng = np.random.default_rng(seed)
    tiles = np.indices((50, 10)).reshape(2, -1).T
    bottom_xy = np.array([0.5, 0.5]) + (tiles + rng.random(tiles.shape)) / [50, 10] * [23, 7]
    bottom = np.column_stack((bottom_xy, rng.uniform(7.2, 7.8, 500)))
    clouds = [rng.normal(center, BLOB_SIGMA_KM, (50, 3)) for center in BLOB_CENTERS_KM]
    xyz_km = np.concatenate((bottom, *clouds))
    if np.any(xyz_km <= 0) or np.any(xyz_km >= DOMAIN_KM):
        raise ValueError("Generated event outside domain; choose another seed")
    events = PointSet(tuple(f"EVT_{i:04d}" for i in range(1, 601)), xyz_km * 1000)

    x, y = np.meshgrid([0.5, 1.5], np.linspace(0.5, 7.5, 5), indexing="ij")
    left = np.column_stack((x.ravel(), y.ravel(), np.zeros(10)))
    right = left.copy()
    right[:, 0] = DOMAIN_KM[0] - right[:, 0]
    middle = np.array([[12.0, 2.0, 0.0], [12.0, 6.0, 0.0]])
    stations = PointSet(tuple(f"STA_{i:02d}" for i in range(1, 23)),
                        np.concatenate((left, right, middle)) * 1000)
    return stations, events


def plot_view(stations: PointSet, events: PointSet, view, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    _, horizontal, vertical, title = view
    width, height = DOMAIN_KM[horizontal], DOMAIN_KM[vertical]
    fig, ax = plt.subplots(figsize=(12, 4.8) if width == 24 else (6.5, 6.5),
                           constrained_layout=True)
    ax.add_patch(Rectangle((0, 0), width, height, facecolor="#fafafa",
                           edgecolor="0.25", linewidth=1.2))
    for coordinate in np.arange(4, width, 4):
        ax.plot([coordinate, coordinate], [0, height], color="0.8", linewidth=0.8, zorder=1)
    for coordinate in np.arange(4, height, 4):
        ax.plot([0, width], [coordinate, coordinate], color="0.8", linewidth=0.8, zorder=1)
    xyz = events.coordinates_m / 1000
    station_xyz = stations.coordinates_m / 1000
    ax.scatter(xyz[:, horizontal], xyz[:, vertical], color="black", s=9,
               linewidths=0, zorder=2, label="600 events (500 bottom + 50 + 50)")
    ax.scatter(station_xyz[:, horizontal], station_xyz[:, vertical], marker="^",
               s=65, color="#b23a35", edgecolors="white", linewidths=0.6,
               zorder=3, label="22 stations (10 + 10 + 2)")
    labels = ("x (km)", "y (km)", "Depth z (km)")
    ax.set(xlim=(-0.35, width + 0.35), ylim=(-0.5, height + 0.35),
           xlabel=labels[horizontal], ylabel=labels[vertical],
           title=f"{title}\n24 × 8 × 8 km; 4 km blocks; projection of all points")
    if vertical == 2:
        ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(0, width + 1, 2))
    ax.set_yticks(np.arange(0, height + 1, 2))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=2 if width == 24 else 1, fontsize=9)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-model", type=Path,
                        default=Path(__file__).parent / "experiments/input" / SOURCE_ID / "model.npz")
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    output = args.output_dir
    names = ["model.npz", "stations.csv", "events.csv", "generation.json"]
    names += [f"geometry_{view[0]}.png" for view in VIEWS]
    if any((output / name).exists() for name in names):
        parser.error("Output files already exist; choose a new --output-dir")
    with np.load(args.source_model, allow_pickle=False) as source:
        if (source["velocity"].shape != (6, 3, 3) or float(source["cell_size_m"]) != 4000
                or not np.array_equal(source["origin_m"], [0, 0, 0])):
            parser.error("Expected the zero-origin 24 x 12 x 12 km nx6 source model")
        model = VelocityGrid(source["velocity"][:, :2, :2], 4000.0)
    stations, events = generate_geometry(args.seed)
    model.validate_points(stations, "stations")
    model.validate_points(events, "events")
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "model.npz", velocity=model.velocity, cell_size_m=model.cell_size_m,
             origin_m=np.asarray(model.origin_m))
    for filename, id_column, points in (("stations.csv", "station_id", stations),
                                        ("events.csv", "event_id", events)):
        with (output / filename).open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow((id_column, "x_m", "y_m", "z_m"))
            writer.writerows((identifier, *xyz) for identifier, xyz in
                             zip(points.ids, points.coordinates_m))
    metadata = {"generator": "bottom_blobs", "seed": args.seed,
                "domain_km": DOMAIN_KM, "blocks_xyz": [6, 2, 2], "cell_size_m": 4000,
                "source_model": str(args.source_model),
                "source_model_sha256": hashlib.sha256(args.source_model.read_bytes()).hexdigest(),
                "crop": "velocity[:, :2, :2]; origin unchanged",
                "bottom": {"count": 500, "xy_range_km": [[0.5, 23.5], [0.5, 7.5]],
                           "depth_range_km": [7.2, 7.8], "xy_strata": [50, 10],
                           "first_id": "EVT_0001", "last_id": "EVT_0500"},
                "blobs": {"count_each": 50, "centers_km": BLOB_CENTERS_KM,
                          "isotropic_gaussian_sigma_km": BLOB_SIGMA_KM,
                          "first_ids": ["EVT_0501", "EVT_0551"]},
                "stations": {"left": 10, "right": 10, "middle": 2,
                             "left_x_km": [0.5, 1.5], "right_x_km": [22.5, 23.5],
                             "middle_xy_km": [[12, 2], [12, 6]], "z_m": 0},
                "coordinate_units": "m", "z_positive": "down",
                "figures": "Projections of all points, not thin slices; gray lines are block boundaries"}
    with (output / "generation.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    for view in VIEWS:
        plot_view(stations, events, view, output / f"geometry_{view[0]}.png")
    for name in names:
        print(output / name)


if __name__ == "__main__":
    main()
