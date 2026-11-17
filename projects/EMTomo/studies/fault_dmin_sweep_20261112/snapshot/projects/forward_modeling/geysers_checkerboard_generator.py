"""Prepare synthetic nx6 inputs using the exact 31-station real-pick geometry.

Only IDs and station positions come from the training archive. Catalog positions
are used as synthetic source truth, not as known true real earthquake locations.
Real pick values and holdout files are not read or used as synthetic observations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .experiments import save_inputs
from .model import PointSet, VelocityGrid


MODEL_SOURCE_ID = "checkerboard_24x12x12_nx6_98st_1200ev_grid20x10x6_minz3_noise01_pick001_seed42"
EXPERIMENT_ID = "geysers_realgeom_nx6_24x12x12_31st_3471ev_noise01_pick005_seed42"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def catalog_event_points(rows, event_ids, geometry) -> PointSet:
    """Project the selected catalog in the existing real-data frame exactly once.

    geographic_reference_in_new_xy_m already includes the old-to-new translation.
    Catalog depth is relative to sea level; add the model top elevation once.
    """
    catalog = {row["event_id"]: row for row in rows}
    if len(catalog) != len(rows):
        raise ValueError("Duplicate catalog event IDs")
    if not set(event_ids) <= catalog.keys():
        raise ValueError("Catalog is missing selected real-inversion event IDs")
    reference_latitude = geometry["reference_latitude"]
    reference_longitude = geometry["reference_longitude"]
    expected_east_scale = f"111.320*cos(radians({reference_latitude}))"
    if geometry["east_km_per_longitude_degree"] != expected_east_scale:
        raise ValueError("Unsupported longitude scale in the saved geometry")
    if geometry["z_direction"] != "positive_down":
        raise ValueError("Expected positive-down coordinates")
    angle = math.radians(geometry["x_azimuth_degrees"])
    east_scale = 111.320 * math.cos(math.radians(reference_latitude))
    offset_x, offset_y = geometry["geographic_reference_in_new_xy_m"]
    xyz = []
    for event in event_ids:
        row = catalog[event]
        north = (float(row["latitude"]) - reference_latitude) * geometry["north_km_per_latitude_degree"]
        east = (float(row["longitude"]) - reference_longitude) * east_scale
        x = (north * math.cos(angle) + east * math.sin(angle)) * 1000 + offset_x
        y = (-north * math.sin(angle) + east * math.cos(angle)) * 1000 + offset_y
        z = geometry["top_elevation_m"] + float(row["depth_km"]) * 1000
        xyz.append((x, y, z))
    return PointSet(tuple(event_ids), xyz)


def prepare_geometry(real_dir: Path, model_path: Path):
    manifest = json.loads((real_dir / "manifest.json").read_text())
    names = ("emtomo/arrivals.npz", "emtomo/metadata.json", "geometry.json",
             "catalog.csv", "stations.csv")
    for name in names:
        if sha256(real_dir / name) != manifest["output_sha256"][name]:
            raise ValueError(f"Real geometry source changed: {name}")
    geometry = json.loads((real_dir / "geometry.json").read_text())
    with np.load(real_dir / "emtomo/arrivals.npz", allow_pickle=False) as archive:
        event_ids = tuple(map(str, archive["event_ids"]))
        stations = PointSet(tuple(map(str, archive["station_ids"])), archive["station_coordinates_m"])
    if len(event_ids) != 3471 or len(stations.ids) != 31:
        raise ValueError("Expected the approved 3471-event, 31-station real-inversion selection")
    with (real_dir / "stations.csv").open(newline="", encoding="utf-8") as stream:
        station_rows = {r["station_id"]: r for r in csv.DictReader(stream)}
    np.testing.assert_array_equal(
        stations.coordinates_m,
        [[float(station_rows[i][k]) for k in ("x_m", "y_m", "z_m")] for i in stations.ids],
    )
    with (real_dir / "catalog.csv").open(newline="", encoding="utf-8") as stream:
        events = catalog_event_points(list(csv.DictReader(stream)), event_ids, geometry)
    with np.load(model_path, allow_pickle=False) as archive:
        model = VelocityGrid(archive["velocity"], float(archive["cell_size_m"]),
                             tuple(archive["origin_m"]))
    if (model.velocity.shape != (6, 3, 3) or model.cell_size_m != 4000
            or model.origin_m != (0.0, 0.0, 0.0)):
        raise ValueError("Expected the zero-origin nx6 checkerboard with 4 km cubic blocks")
    model.validate_points(stations, "stations")
    model.validate_points(events, "events")
    provenance = {
        "generator": "geysers_real_geometry_checkerboard",
        "domain_km": [24, 12, 12], "blocks_xyz": [6, 3, 3], "cell_size_m": 4000,
        "original_real_geometry": geometry,
        "embedding": "Keep real metric coordinates and origin unchanged; extend upper domain bounds only",
        "event_selection": "Exactly the 3471 event IDs in the approved real-inversion training archive, in its order",
        "source_position_policy": "Catalog coordinates are synthetic source truth only, not real-data ground truth",
        "station_policy": "All 31 real-inversion stations, unchanged xyz including elevation-derived z; not flattened",
        "observations_policy": "Fresh synthetic arrivals for every event/station pair; no real observed times or real missing-pick mask",
        "datum_caveat": geometry["datum_caveat"],
        "sources": {name: {"path": str(real_dir / name), "sha256": sha256(real_dir / name)}
                    for name in (*names, "manifest.json")},
        "model_source": {"path": str(model_path), "sha256": sha256(model_path)},
    }
    return model, stations, events, provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parent / "experiments")
    parser.add_argument("--real-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "EMTomo/data/geysers_2011/real_experiment_v1")
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    args = parser.parse_args(argv)
    model_path = args.root / "input" / MODEL_SOURCE_ID / "model.npz"
    model, stations, events, provenance = prepare_geometry(args.real_dir, model_path)
    destination = save_inputs(args.root, args.experiment_id, model, stations, events, generation=provenance)
    print(destination)
    print(f"{len(events.ids)} events x {len(stations.ids)} stations = {len(events.ids) * len(stations.ids)} pairs")
    print("Station xyz range (m):", stations.coordinates_m.min(axis=0), stations.coordinates_m.max(axis=0))
    print("Event xyz range (m):", events.coordinates_m.min(axis=0), events.coordinates_m.max(axis=0))


if __name__ == "__main__":
    main()
