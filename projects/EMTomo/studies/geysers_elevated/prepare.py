"""Prepare elevated Geysers native inputs, without terrain, air, or any solves.

The unchanged velocity cube is physically translated upward to H=1166 m.
Native z is positive down from that flat top: stations use H-elevation and
catalogue events use H+old_depth. The shared vertical datum is NOT verified.
Only the immutable native source dataset is used; raw data are never resampled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

_PROJECTS = Path(__file__).resolve().parents[3]
if str(_PROJECTS.parent) not in sys.path:
    sys.path.insert(0, str(_PROJECTS.parent))

from projects.forward_modeling.experiments import load_inputs, save_inputs
from projects.forward_modeling.model import PointSet

SOURCE_EXPERIMENT_ID = "geysers11_20110401-20110420_flat4_26st_3780ev_noise01_pick001"
EXPERIMENT_ID = "geysers11_20110401-20110420_flat4_elev1166_26st_3776ev_noise01_pick001"
DEFAULT_EXPERIMENTS_ROOT = _PROJECTS / "forward_modeling" / "experiments"
TOP_ELEVATION_M = 1166.0
START_UTC = "2011-04-01T00:00:00Z"
END_UTC_EXCLUSIVE = "2011-04-21T00:00:00Z"
EXPECTED_SOURCE_EVENT_COUNT = 3780
EXPECTED_EVENT_COUNT = 3776
EXPECTED_STATION_COUNT = 26
EXPECTED_DROPPED_COUNT = 4
_EXTENT_M = (20000.0, 7000.0, 7000.0)
_SOURCE_FILES = ("model.npz", "stations.csv", "events.csv", "generation.json")


def _json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _read_json(path):
    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid_constant,
                      object_pairs_hook=unique_keys)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes(directory):
    return {name: _sha256(directory / name) for name in _SOURCE_FILES}


def _utc(value):
    parsed = datetime.fromisoformat(value)
    offset = parsed.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise ValueError(f"Expected an explicit UTC timestamp: {value!r}")
    return parsed.astimezone(UTC)


def _transform(model, stations, events, source):
    """Validate the fixed source selection, transform z, and drop (never clip)."""
    if (model.origin_m != (0.0, 0.0, 0.0)
            or not np.array_equal(np.asarray(model.velocity.shape) * model.cell_size_m, _EXTENT_M)):
        raise ValueError("Source model must cover [0,20000] x [0,7000] x [0,7000] m")
    if len(stations.ids) != EXPECTED_STATION_COUNT or len(events.ids) != EXPECTED_SOURCE_EVENT_COUNT:
        raise ValueError("Unexpected source station/event counts; refusing to reselect raw data")
    for key, expected in (("experiment_id", SOURCE_EXPERIMENT_ID), ("start_utc", START_UTC),
                          ("end_utc_exclusive", END_UTC_EXCLUSIVE),
                          ("station_count", len(stations.ids)), ("event_count", len(events.ids)),
                          ("pair_count", len(stations.ids) * len(events.ids))):
        if source.get(key) != expected:
            raise ValueError(f"Source generation {key} mismatch")
    if np.any(stations.coordinates_m[:, 2] != 0.0):
        raise ValueError("Source stations must use the old flat z=0 boundary")

    metadata = source["station_metadata"]
    by_id = {row["station_id"]: row for row in metadata}
    if len(by_id) != len(metadata) or set(by_id) != set(stations.ids):
        raise ValueError("Source station_metadata IDs must match stations exactly, without duplicates")
    for key, ids in (("station_ids", stations.ids), ("event_ids", events.ids)):
        if key in source and source[key] != list(ids):
            raise ValueError(f"Source generation {key} mismatch")
    rows = [by_id[identifier] for identifier in stations.ids]
    elevations = np.asarray([row["elevation_m"] for row in rows], dtype=float)
    if not np.all(np.isfinite(elevations)) or float(elevations.max()) != TOP_ELEVATION_M:
        raise ValueError("Station elevations must be finite with highest station H=1166 m")
    for row, xyz in zip(rows, stations.coordinates_m):
        if not np.array_equal([row[key] for key in ("x_m", "y_m", "z_m")], xyz):
            raise ValueError("Source station_metadata XYZ mismatch")

    times = source["event_origin_times_utc"]
    if set(times) != set(events.ids):
        raise ValueError("Source UTC time IDs must match events exactly")
    start, end = _utc(START_UTC), _utc(END_UTC_EXCLUSIVE)
    if any(not start <= _utc(times[identifier]) < end for identifier in events.ids):
        raise ValueError("Source event UTC time outside the original selection interval")
    velocity_sha256 = hashlib.sha256(model.velocity.tobytes()).hexdigest()
    if source.get("true_velocity_sha256", velocity_sha256) != velocity_sha256:
        raise ValueError("Source true_velocity_sha256 mismatch")

    station_xyz = stations.coordinates_m.copy()
    station_xyz[:, 2] = TOP_ELEVATION_M - elevations
    elevated_stations = PointSet(stations.ids, station_xyz)
    model.validate_points(elevated_stations, "elevated stations")
    event_xyz = events.coordinates_m.copy()
    event_xyz[:, 2] += TOP_ELEVATION_M
    lower = np.asarray(model.origin_m)
    upper = lower + model.cell_size_m * np.asarray(model.velocity.shape)
    keep = np.all((event_xyz >= lower) & (event_xyz <= upper), axis=1)
    dropped = [
        {"event_id": events.ids[i], "reason": "outside_closed_model_domain_after_elevation_transform",
         "source_coordinates_m": events.coordinates_m[i].tolist(),
         "transformed_coordinates_m": event_xyz[i].tolist(),
         "origin_time_utc": times[events.ids[i]]}
        for i in np.flatnonzero(~keep)
    ]
    if int(keep.sum()) != EXPECTED_EVENT_COUNT or len(dropped) != EXPECTED_DROPPED_COUNT:
        raise ValueError(f"Expected {EXPECTED_EVENT_COUNT} retained/{EXPECTED_DROPPED_COUNT} dropped events; "
                         f"got {int(keep.sum())} retained/{len(dropped)} dropped")
    elevated_events = PointSet(tuple(identifier for identifier, retained in zip(events.ids, keep)
                                     if retained), event_xyz[keep])
    model.validate_points(elevated_events, "elevated events")
    return elevated_stations, elevated_events, rows, dropped


def _same_model(left, right):
    return (left.velocity.shape == right.velocity.shape
            and left.velocity.dtype == right.velocity.dtype
            and left.velocity.tobytes() == right.velocity.tobytes()
            and np.asarray((left.cell_size_m, *left.origin_m)).tobytes()
            == np.asarray((right.cell_size_m, *right.origin_m)).tobytes())


def _verify_existing(root, model, stations, events, generation):
    directory = root / "input" / EXPERIMENT_ID
    try:
        actual_model, actual_stations, actual_events = load_inputs(root, EXPERIMENT_ID)
        if not _same_model(actual_model, model):
            raise ValueError("true model differs")
        for name, actual, expected in (("stations", actual_stations, stations),
                                       ("events", actual_events, events)):
            if (actual.ids != expected.ids
                    or actual.coordinates_m.tobytes() != expected.coordinates_m.tobytes()):
                raise ValueError(f"{name} differ")
        if _json_bytes(_read_json(directory / "generation.json")) != _json_bytes(generation):
            raise ValueError("generation/provenance differs")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"Existing inputs invalid or different; refusing overwrite: {directory}: {error}") from error


def prepare_experiment(experiments_root=DEFAULT_EXPERIMENTS_ROOT, validate_only=False) -> dict:
    """Create native inputs once, or reuse only exact semantics AND provenance.

    ``validate_only=True`` never writes, including when a destination exists
    (it still verifies it). Returns a JSON-safe summary with full generation
    metadata. No arrivals, forward modeling, or inversion are read or run.
    """
    root = Path(experiments_root).resolve()
    source_dir = root / "input" / SOURCE_EXPERIMENT_ID
    hashes = _source_hashes(source_dir)
    model, source_stations, source_events = load_inputs(root, SOURCE_EXPERIMENT_ID)
    source = _read_json(source_dir / "generation.json")
    stations, events, original_rows, dropped = _transform(model, source_stations, source_events, source)
    generation = {
        "schema_version": 1,
        "generator": "geysers_elevated.prepare",
        "experiment_id": EXPERIMENT_ID,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "source_hashes": hashes,
        "source_generation_metadata": {key: value for key, value in source.items()
                                       if key not in ("station_metadata", "event_origin_times_utc",
                                                      "station_ids", "event_ids")},
        "start_utc": source["start_utc"],
        "end_utc_exclusive": source["end_utc_exclusive"],
        "station_count": len(stations.ids), "event_count": len(events.ids),
        "pair_count": len(stations.ids) * len(events.ids),
        "source_event_count": len(source_events.ids),
        "station_ids": list(stations.ids), "event_ids": list(events.ids),
        "selection": "reuse immutable native source selection/order; drop only transformed out-of-domain events; no clipping or resampling",
        "top_elevation_m": TOP_ELEVATION_M,
        "bottom_elevation_m": TOP_ELEVATION_M - _EXTENT_M[2],
        "vertical_coordinates": {
            "positive_direction": "down", "origin": "flat top at H=1166 m",
            "station_z": "H - source station_metadata elevation_m",
            "event_z": "H + source events.csv z_m (old catalogue depth)",
            "xy": "unchanged for all retained IDs",
        },
        "shared_datum_assumption": {
            "verified": False,
            "statement": "Catalogue depth and station elevation are assumed to share a vertical datum; this has NOT been verified or corrected.",
            "physical_station_height": "source elevation_m",
            "physical_event_height": "negative old catalogue depth_m",
        },
        "terrain": False, "air": False,
        "true_model_source": SOURCE_EXPERIMENT_ID,
        "true_model_archive_sha256": hashes["model.npz"],
        "true_velocity_sha256": hashlib.sha256(model.velocity.tobytes()).hexdigest(),
        "true_velocity_policy": "bitwise unchanged physical velocity array; cube translated upward by H; no padding, stretching, regridding, terrain or air",
        "true_grid": {"shape": list(model.velocity.shape), "cell_size_m": model.cell_size_m,
                      "origin_m": list(model.origin_m)},
        "source_station_metadata": original_rows,
        "station_metadata": [{**row, "z_m": float(xyz[2])}
                             for row, xyz in zip(original_rows, stations.coordinates_m)],
        "event_origin_times_utc": {identifier: source["event_origin_times_utc"][identifier]
                                   for identifier in events.ids},
        "dropped_event_count": len(dropped),
        "dropped_event_ids": [row["event_id"] for row in dropped],
        "dropped_events": dropped,
        "domain_m": {"lower": list(model.origin_m), "upper": list(_EXTENT_M),
                     "boundary": "closed; equality retained; outside excluded, never clipped"},
    }
    # Check the same immutable snapshot used for both geometry and provenance.
    if hashes != _source_hashes(source_dir):
        raise ValueError("Source files changed during preparation")
    directory = root / "input" / EXPERIMENT_ID
    existed = os.path.lexists(directory)
    if existed:
        _verify_existing(root, model, stations, events, generation)
    elif not validate_only:
        try:
            save_inputs(root, EXPERIMENT_ID, model, stations, events, generation)
        except FileExistsError:
            if not os.path.lexists(directory):
                raise
            _verify_existing(root, model, stations, events, generation)
            existed = True
    result = {
        "experiment_id": EXPERIMENT_ID, "input_dir": str(directory),
        "source_experiment_id": SOURCE_EXPERIMENT_ID, "source_input_dir": str(source_dir),
        "station_count": len(stations.ids), "event_count": len(events.ids),
        "pair_count": len(stations.ids) * len(events.ids), "dropped_event_count": len(dropped),
        "start_utc": generation["start_utc"], "end_utc_exclusive": generation["end_utc_exclusive"],
        "source_hashes": hashes,
        "generation_sha256": hashlib.sha256(_json_bytes(generation)).hexdigest(),
        "generation": generation,
        "status": "validated" if validate_only else ("reused" if existed else "created"),
    }
    # Enforce a strict JSON-safe public result for pipeline callers.
    return json.loads(_json_bytes(result))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--validate-only", action="store_true", help="Check without writing inputs")
    args = parser.parse_args(argv)
    try:
        result = prepare_experiment(args.experiments_root, validate_only=args.validate_only)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"prepare: {error}\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
