"""Immutable real-geometry inputs for the Geysers April 2011 synthetic study.

No arrivals are required or computed here. Catalog locations are not sampled,
perturbed, or generated; the old flat4 true velocity grid is copied unchanged.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np

# Support both direct CLI execution and import by the study queue.
_PROJECTS = Path(__file__).resolve().parents[3]
if str(_PROJECTS.parent) not in sys.path:
    sys.path.insert(0, str(_PROJECTS.parent))

from projects.forward_modeling.experiments import load_inputs, save_inputs
from projects.forward_modeling.model import PointSet


EXPERIMENT_ID = "geysers11_20110401-20110420_flat4_26st_3780ev_noise01_pick001"
TRUE_MODEL_SOURCE = "flat4_20x7x7km_52st_1920ev_grid24x8x10_z0.5-5.5_noise01_pick001"
DEFAULT_EXPERIMENTS_ROOT = _PROJECTS / "forward_modeling" / "experiments"
DEFAULT_DATA_DIR = _PROJECTS / "EMTomo" / "data" / "geysers_2011"
START_UTC = "2011-04-01T00:00:00Z"
END_UTC_EXCLUSIVE = "2011-04-21T00:00:00Z"
EXPECTED_EVENT_COUNT = 3780
EXPECTED_STATION_COUNT = 26
PREVIOUS_PAIR_COUNT = 52 * 1920
_EXTENT_M = (20000.0, 7000.0, 7000.0)
_SOURCE_FILES = ("stations_metric.csv", "events_metric_all.csv", "geometry_metadata.json")


def _utc(value: str) -> datetime:
    # NCEDC's slash-formatted catalog timestamps are documented UTC.
    parsed = datetime.fromisoformat(value.strip().replace("/", "-").replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _utc_text(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path, required: set[str], id_column: str) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path.name}: missing required columns {sorted(required)}")
        rows = list(reader)
    seen = set()
    for row in rows:
        if None in row or any(row.get(key) is None for key in required):
            raise ValueError(f"{path.name}: malformed CSV row")
        identifier = row[id_column]
        if not identifier.strip() or identifier != identifier.strip() or identifier in seen:
            raise ValueError(f"{path.name}: empty, noncanonical or duplicate {id_column}: {identifier!r}")
        seen.add(identifier)
    return rows


def _number(row: dict, key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite {key}: {row[key]!r}")
    return value


def _check_projection(row: dict, geometry: dict, *, station: bool) -> tuple[float, float, float]:
    latitude, longitude = _number(row, "latitude"), _number(row, "longitude")
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError("Invalid latitude/longitude")
    north = (latitude - geometry["top_face_center_latitude"]) * 110.574
    east = (longitude - geometry["top_face_center_longitude"]) * (
        111.320 * math.cos(math.radians(geometry["top_face_center_latitude"]))
    )
    azimuth = math.radians(geometry["azimuth_degrees"])
    projected = (
        (north * math.cos(azimuth) + east * math.sin(azimuth) + 10.0) * 1000.0,
        (-north * math.sin(azimuth) + east * math.cos(azimuth) + 3.5) * 1000.0,
        0.0 if station else _number(row, "depth_km") * 1000.0,
    )
    coordinates = (_number(row, "x_m"), _number(row, "y_m"), _number(row, "z_m"))
    if not np.allclose(coordinates, projected, rtol=0.0, atol=1e-7):
        raise ValueError("Raw-to-metric local projection/depth mismatch")
    if station and coordinates[2] != 0.0:
        raise ValueError("Stations must remain on the flat z=0 boundary")
    if any(value < 0 or value > extent for value, extent in zip(coordinates, _EXTENT_M)):
        raise ValueError("Coordinates outside the closed 20x7x7 km model domain")
    return coordinates


def _select_geometry(data_dir: Path):
    metadata = json.loads((data_dir / "geometry_metadata.json").read_text(encoding="utf-8"))
    geometry = metadata["geometry"]
    for key, expected in zip(("width_x_m", "width_y_m", "depth_m"), _EXTENT_M):
        if geometry[key] != expected:
            raise ValueError(f"Unexpected geometry {key}: {geometry[key]}")
    if geometry["station_z_m"] != 0.0:
        raise ValueError("Geometry must use flat station z=0")
    for key in ("top_face_center_latitude", "top_face_center_longitude", "azimuth_degrees"):
        if not math.isfinite(float(geometry[key])):
            raise ValueError(f"Nonfinite geometry {key}")
    common = {"latitude", "longitude", "x_m", "y_m", "z_m"}
    stations = _rows(data_dir / "stations_metric.csv", common | {
        "network", "station", "elevation_m", "start_time", "end_time",
    }, "station")
    events = _rows(data_dir / "events_metric_all.csv", common | {
        "event_id", "origin_time", "depth_km",
    }, "event_id")
    start, end = _utc(START_UTC), _utc(END_UTC_EXCLUSIVE)
    selected_stations = []
    station_metadata = []
    for row in sorted(stations, key=lambda row: row["station"]):
        epoch_start = _utc(row["start_time"])
        epoch_end = _utc(row["end_time"]) if row["end_time"].strip() else None
        if epoch_end is not None and epoch_end <= epoch_start:
            raise ValueError(f"Invalid station epoch: {row['station']}")
        if epoch_start > start or (epoch_end is not None and epoch_end < end):
            continue
        coordinates = _check_projection(row, geometry, station=True)
        selected_stations.append((row["station"], coordinates))
        station_metadata.append({
            "station_id": row["station"], "network": row["network"],
            "latitude": _number(row, "latitude"), "longitude": _number(row, "longitude"),
            "elevation_m": _number(row, "elevation_m"),
            "site_name": row.get("site_name", ""),
            "start_utc": _utc_text(epoch_start),
            "end_utc": _utc_text(epoch_end) if epoch_end else None,
            "x_m": coordinates[0], "y_m": coordinates[1], "z_m": coordinates[2],
        })
    selected_events = sorted(
        ((time, row["event_id"], row) for row in events
         if start <= (time := _utc(row["origin_time"])) < end),
        key=lambda item: (item[0], item[1]),
    )
    if not selected_stations or not selected_events:
        raise ValueError("Selection contains no stations or events")
    station_points = PointSet(
        tuple(identifier for identifier, _ in selected_stations),
        np.asarray([coordinates for _, coordinates in selected_stations]),
    )
    event_points = PointSet(
        tuple(identifier for _, identifier, _ in selected_events),
        np.asarray([_check_projection(row, geometry, station=False)
                    for _, _, row in selected_events]),
    )
    return station_points, event_points, {
        "geometry": geometry,
        "catalog_source": metadata.get("source", {}),
        "station_metadata": station_metadata,
        "event_origin_times_utc": {
            identifier: _utc_text(time) for time, identifier, _ in selected_events
        },
    }


def _same_model(left, right) -> bool:
    return (
        left.velocity.shape == right.velocity.shape
        and left.velocity.dtype == right.velocity.dtype
        and left.velocity.tobytes() == right.velocity.tobytes()
        and np.asarray((left.cell_size_m, *left.origin_m)).tobytes()
        == np.asarray((right.cell_size_m, *right.origin_m)).tobytes()
    )


def _verify_existing(root, experiment_id, model, stations, events, generation):
    directory = root / "input" / experiment_id
    actual_model, actual_stations, actual_events = load_inputs(root, experiment_id)
    actual_generation = json.loads((directory / "generation.json").read_text(encoding="utf-8"))
    if not _same_model(actual_model, model):
        raise ValueError(f"Existing true model differs; refusing overwrite: {directory}")
    for name, actual, expected in (("stations", actual_stations, stations),
                                   ("events", actual_events, events)):
        if actual.ids != expected.ids or not np.array_equal(actual.coordinates_m, expected.coordinates_m):
            raise ValueError(f"Existing {name} differ; refusing overwrite: {directory}")
    if _json_bytes(actual_generation) != _json_bytes(generation):
        raise ValueError(f"Existing generation/source hashes differ; refusing overwrite: {directory}")


def prepare_experiment(
    experiments_root=DEFAULT_EXPERIMENTS_ROOT,
    data_dir=DEFAULT_DATA_DIR,
    experiment_id=EXPERIMENT_ID,
    *,
    validate_only=False,
) -> dict:
    """Validate sources, create native inputs once, or verify exact existing inputs.

    ``validate_only=True`` performs the same validation without creating anything.
    An existing different dataset always raises; no arrivals/output are read.
    The returned dictionary is JSON-serializable and includes full provenance.
    """
    if not isinstance(experiment_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", experiment_id):
        raise ValueError("Invalid experiment_id")
    root, data_dir = Path(experiments_root).resolve(), Path(data_dir).resolve()
    model_archive = root / "input" / TRUE_MODEL_SOURCE / "model.npz"
    paths = {name: data_dir / name for name in _SOURCE_FILES}
    paths["true_model.npz"] = model_archive
    source_hashes = {name: _sha256(path) for name, path in paths.items()}
    model, _, _ = load_inputs(root, TRUE_MODEL_SOURCE)
    if model.origin_m != (0.0, 0.0, 0.0) or not np.array_equal(
        np.asarray(model.velocity.shape) * model.cell_size_m, _EXTENT_M
    ):
        raise ValueError("Source model must cover [0,20000] x [0,7000] x [0,7000] m")
    stations, events, provenance = _select_geometry(data_dir)
    if len(stations.ids) != EXPECTED_STATION_COUNT or len(events.ids) != EXPECTED_EVENT_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_STATION_COUNT} stations/{EXPECTED_EVENT_COUNT} events; "
            f"got {len(stations.ids)} stations/{len(events.ids)} events"
        )
    model.validate_points(stations, "stations")
    model.validate_points(events, "events")
    if source_hashes != {name: _sha256(path) for name, path in paths.items()}:
        raise ValueError("Source files changed during preparation")
    pair_count = len(stations.ids) * len(events.ids)
    generation = {
        "schema_version": 1,
        "generator": "geysers2011.prepare",
        "experiment_id": experiment_id,
        "source_hashes": source_hashes,
        "true_model_source": TRUE_MODEL_SOURCE,
        "true_model_archive_sha256": source_hashes["true_model.npz"],
        "true_velocity_sha256": hashlib.sha256(model.velocity.tobytes()).hexdigest(),
        "true_grid": {"shape": list(model.velocity.shape), "cell_size_m": model.cell_size_m,
                      "origin_m": list(model.origin_m)},
        "start_utc": START_UTC,
        "end_utc_exclusive": END_UTC_EXCLUSIVE,
        "event_count": len(events.ids), "station_count": len(stations.ids),
        "pair_count": pair_count,
        "previous_pair_count": PREVIOUS_PAIR_COUNT,
        "pair_count_change_percent": 100.0 * (pair_count - PREVIOUS_PAIR_COUNT) / PREVIOUS_PAIR_COUNT,
        "selection": "all catalog events in [start_utc,end_utc_exclusive); no thinning or generated locations",
        "ordering": "events: UTC origin time then string event ID; stations: string station ID",
        "station_epoch_policy": "start <= interval start and end >= interval end (blank end is open)",
        "arrival_kind": "synthetic arrivals with real catalog/station geometry; no local real picks",
        "station_z_policy": "flat boundary z=0; original elevation_m preserved in station_metadata",
        "depth_datum_approximation": (
            "Catalog depth_km * 1000 is used unchanged as positive-down z from the flat model top. "
            "Catalog depth datum is approximated by that top; no elevation, topographic or datum correction."
        ),
        **provenance,
    }
    directory = root / "input" / experiment_id
    existed = os.path.lexists(directory)
    if existed:
        _verify_existing(root, experiment_id, model, stations, events, generation)
    elif not validate_only:
        try:
            save_inputs(root, experiment_id, model, stations, events, generation)
        except FileExistsError:
            # A concurrent cooperating preparer may have published the same data.
            if not os.path.lexists(directory):
                raise
            _verify_existing(root, experiment_id, model, stations, events, generation)
            existed = True
    return {
        "experiment_id": experiment_id, "input_dir": str(directory),
        "event_count": len(events.ids), "station_count": len(stations.ids),
        "pair_count": pair_count, "start_utc": START_UTC,
        "end_utc_exclusive": END_UTC_EXCLUSIVE, "true_model_source": TRUE_MODEL_SOURCE,
        "true_model_source_sha256": source_hashes["true_model.npz"],
        "source_hashes": source_hashes,
        "generation_sha256": hashlib.sha256(_json_bytes(generation)).hexdigest(),
        "generation": generation,
        "status": "validated" if validate_only else ("reused" if existed else "created"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    parser.add_argument("--prepare", action="store_true", help="Prepare inputs (the default); never run forward")
    parser.add_argument("--validate-only", action="store_true", help="Validate without writing inputs")
    args = parser.parse_args(argv)
    try:
        result = prepare_experiment(args.experiments_root, args.data_dir, args.experiment_id,
                                    validate_only=args.validate_only)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"prepare: {error}\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
