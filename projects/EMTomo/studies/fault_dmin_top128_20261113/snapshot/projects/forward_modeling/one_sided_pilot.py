"""Prepare ten spatially spread events from the saved one-sided catalog.

Only saves forward inputs. Use the independent forward solver afterwards.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np

from .experiments import load_inputs, save_inputs
from .model import PointSet
from .one_sided_events import GROUPS, load_stations


SOURCE_EXPERIMENT = "checkerboard_24x12x12_nx6_98st_1200ev_grid20x10x6_minz3_noise01_pick001_seed42"
EXPERIMENT_ID = "onesided_nx6_24x12x12_18st_10ev_noise01_pick005_seed42"


def spread_indices(xyz: np.ndarray, count: int) -> list[int]:
    """Start near the centroid, then maximize distance to the nearest selection.

    Distances are Euclidean physical distances, not normalized per-axis values.
    Ties follow catalog order. No arrival times or inversion outcomes are used.
    """
    if not 1 <= count <= len(xyz):
        raise ValueError("count must be between one and the number of points")
    selected = [int(np.argmin(np.sum((xyz - xyz.mean(axis=0)) ** 2, axis=1)))]
    distance = np.full(len(xyz), np.inf)
    while len(selected) < count:
        distance = np.minimum(distance, np.sum((xyz - xyz[selected[-1]]) ** 2, axis=1))
        distance[selected] = -np.inf
        selected.append(int(np.argmax(distance)))
    return selected


def select_events(catalog: PointSet) -> tuple[PointSet, dict]:
    selected = []
    groups = {}
    for group, count in zip(GROUPS, (7, 3)):
        xyz_km = catalog.coordinates_m / 1000.0
        mask = np.all((xyz_km >= group["lower_km"]) & (xyz_km <= group["upper_km"]), axis=1)
        candidates = np.flatnonzero(mask)
        indices = candidates[spread_indices(catalog.coordinates_m[candidates], count)]
        selected.extend(indices.tolist())
        groups[group["name"]] = [catalog.ids[i] for i in indices]
    return PointSet(tuple(catalog.ids[i] for i in selected),
                    catalog.coordinates_m[selected]), groups


def main(argv=None):
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parent / "experiments")
    parser.add_argument("--catalog-dir", type=Path,
                        default=project_root / "EMTomo/data/one_sided_stations")
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    args = parser.parse_args(argv)
    model, _, _ = load_inputs(args.root, SOURCE_EXPERIMENT)
    if model.velocity.shape != (6, 3, 3):
        raise ValueError("Expected the existing nx6 model")
    if model.cell_size_m != 4000.0 or model.origin_m != (0.0, 0.0, 0.0):
        raise ValueError("Expected a zero-origin nx6 model with 4 km blocks")
    station_path = args.catalog_dir / "stations.csv"
    event_path = args.catalog_dir / "events.csv"
    stations = load_stations(station_path)
    with event_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["event_id", "x_m", "y_m", "z_m"]:
            raise ValueError("Expected event_id,x_m,y_m,z_m CSV columns")
        rows = list(reader)
    catalog = PointSet(tuple(row["event_id"] for row in rows),
                       [[float(row[key]) for key in ("x_m", "y_m", "z_m")] for row in rows])
    events, groups = select_events(catalog)
    source_paths = {"model": args.root / "input" / SOURCE_EXPERIMENT / "model.npz",
                    "stations": station_path, "catalog": event_path}
    generation = {"generator": "one_sided_pilot", "source_experiment": SOURCE_EXPERIMENT,
                  "selection": "Within each group: nearest to centroid, then farthest-point sampling in physical xyz",
                  "selected_event_groups": groups,
                  "sources": {name: {"path": str(path),
                                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                              for name, path in source_paths.items()},
                  "purpose": "Ten-event forward pilot for subsequent one-iteration weight inspection; not a tomography benchmark"}
    destination = save_inputs(args.root, args.experiment_id, model, stations, events,
                              generation=generation)
    print(destination)
    print("event_id,x_km,y_km,depth_km")
    for identifier, xyz in zip(events.ids, events.coordinates_m / 1000):
        print(f"{identifier},{xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f}")


if __name__ == "__main__":
    main()
