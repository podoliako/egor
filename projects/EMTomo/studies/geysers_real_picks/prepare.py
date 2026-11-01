"""Prepare real BG picks only; no model, forward solve, or inversion.

Seeded SHA-256 rankings are independent of input row order. Holdout quotas start
at floor(n/10), capped at n-6; remaining seats go to the largest quota deficits
(integer tenths), with seeded ties. Repeated allocation handles binding caps.
The global target is round-half-up(sum(n)/10) AFTER excluding n<6 events.
Six-pick events have no holdout and cannot contribute a per-event CV score.
Only emtomo/ is the EMTomo input: root CSVs contain audit/CV/tomoDD information.
All source event IDs (including excluded events) remain in the separate catalog;
tomoDD must select its events from train.csv, never from the catalog alone.
Existing output directories are always refused, including incomplete preparations.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import heapq
import importlib.util
import io
import json
from pathlib import Path
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/geysers_2011"
OUTPUT = DATA / "real_experiment_v1"
SEED = 42
MIN_TRAIN = 6
EXTENT = (23000, 11000, 7000)
REASON = "fewer_than_6_available_P_picks"
DATUM_CAVEAT = (
    "Station elevations and catalog depths are assumed to share a sea-level "
    "vertical datum; compatibility has not been independently established."
)
_spec = importlib.util.spec_from_file_location(
    "geysers_geometry", ROOT / "prepare_geysers_geometry.py"
)
_geometry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_geometry)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timestamp needs an explicit timezone")
    return parsed.astimezone(timezone.utc)


def iso(value):
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def rank(seed, kind, *parts):
    return hashlib.sha256(json.dumps([seed, kind, *parts]).encode()).digest()


def pick_key(row):
    return row["event_id"], row["network"], row["station"]


def allocate_quotas(counts, seed=SEED):
    """Constrained largest-remainder allocation, with no floating rounding ties."""
    if not counts or any(n < MIN_TRAIN for n in counts.values()):
        raise ValueError("Quota inputs must be nonempty and already filtered to n>=6")
    target = (sum(counts.values()) + 5) // 10
    if target > sum(n - MIN_TRAIN for n in counts.values()):
        raise ValueError("Exact 10% holdout is infeasible while keeping six training picks")
    quotas = {event: min(n // 10, n - MIN_TRAIN) for event, n in counts.items()}
    heap = [(-(n - 10 * quotas[e]), rank(seed, "quota", e), e)
            for e, n in counts.items() if quotas[e] < n - MIN_TRAIN]
    heapq.heapify(heap)
    for _ in range(target - sum(quotas.values())):
        _, tie, event = heapq.heappop(heap)
        quotas[event] += 1
        if quotas[event] < counts[event] - MIN_TRAIN:
            heapq.heappush(heap, (-(counts[event] - 10 * quotas[event]), tie, event))
    return quotas


def geometry_metadata():
    if (_geometry.CENTER_LAT, _geometry.CENTER_LON, _geometry.AZIMUTH_DEGREES,
            _geometry.WIDTH_X_KM, _geometry.WIDTH_Y_KM) != (
                38.80275, -122.77070, 135.0, 20.0, 7.0):
        raise ValueError("Original XY reference changed; v1 geometry must not move")
    return {
        "reference_latitude": 38.80275, "reference_longitude": -122.77070,
        "x_azimuth_degrees": 135.0, "y_azimuth_degrees": 225.0,
        "north_km_per_latitude_degree": 110.574,
        "east_km_per_longitude_degree": "111.320*cos(radians(38.80275))",
        "projection": "prepare_geysers_geometry.local_xy_km",
        "local_xy_to_old_xy_offset_m": [10000, 3500],
        "old_to_new_translation_m": [3000, 0, 0],
        "geographic_reference_in_new_xy_m": [13000, 3500],
        "domain_bounds_in_old_coordinates_m": [[-3000, 20000], [0, 11000], [0, 7000]],
        "domain_extent_m": list(EXTENT), "domain_origin_m": [0, 0, 0],
        "top_elevation_m": 1166, "z_direction": "positive_down",
        "station_z_formula": "1166 - elevation_m",
        "catalog_z_formula_for_tomoDD_only": "1166 + 1000*depth_km",
        "datum_caveat": DATUM_CAVEAT,
    }


def build_bundle(picks, catalog_rows, station_epochs, seed=SEED):
    """Build deterministic tables and training-only arrays without writing files."""
    if not picks:
        raise ValueError("No input picks")
    fields = list(picks[0])
    groups, seen = {}, set()
    start, end = utc("2011-04-01T00:00:00Z"), utc("2011-04-21T00:00:00Z")
    for row in sorted(picks, key=pick_key):
        key = pick_key(row)
        if key in seen:
            raise ValueError(f"Duplicate event/network/station: {key}")
        if set(row) != set(fields) or row["network"] != "BG" or row["phase"] != "P":
            raise ValueError("Expected uniform source columns and BG P picks only")
        seen.add(key)
        if not start <= utc(row["origin_time_utc"]) < end:
            raise ValueError("Origin outside selected April 1-20 period")
        utc(row["pick_time_utc"])
        groups.setdefault(row["event_id"], []).append(row)
    origins = {}
    for event, rows in groups.items():
        values = {r["origin_time_utc"] for r in rows}
        if len(values) != 1:
            raise ValueError(f"Inconsistent source pick origins: {event}")
        origins[event] = values.pop()
    selected = {}
    for row in catalog_rows:
        event = row["event_id"]
        if event in groups:
            if event in selected:
                raise ValueError(f"Duplicate catalog event: {event}")
            selected[event] = row
    if selected.keys() != groups.keys():
        raise ValueError("Catalog is missing source pick event IDs")
    catalog = [{"event_id": e, "origin_time_utc": origins[e],
                **{k: selected[e][k] for k in (
                    "latitude", "longitude", "depth_km", "magnitude", "magnitude_type")}}
               for e in sorted(groups)]
    geometry = geometry_metadata()
    stations = []
    for network, code in sorted({(r["network"], r["station"]) for r in picks}):
        matches = [r for r in station_epochs if r["network"] == network
                   and r["station"] == code
                   and utc(r["start_time"] + "Z") <= start
                   and utc(r["end_time"] + "Z") >= end]
        if len(matches) != 1:
            raise ValueError(f"Need exactly one station epoch covering April 1-20: {code}")
        station = dict(matches[0])
        x, y = _geometry.local_xy_km(float(station["latitude"]), float(station["longitude"]))
        station.update(station_id=f"{network}.{code}", x_old_m=(x + 10) * 1000,
                       y_old_m=(y + 3.5) * 1000,
                       x_m=(x + 10) * 1000 + 3000, y_m=(y + 3.5) * 1000,
                       z_m=1166 - float(station["elevation_m"]))
        xyz = [station[k] for k in ("x_m", "y_m", "z_m")]
        if not all(np.isfinite(v) and 0 <= v <= bound for v, bound in zip(xyz, EXTENT)):
            raise ValueError(f"Station outside fixed domain: {code}")
        stations.append(station)
    station_map = {s["station_id"]: s for s in stations}
    for row in picks:
        s = station_map[f'{row["network"]}.{row["station"]}']
        if not utc(s["start_time"] + "Z") <= utc(row["pick_time_utc"]) < utc(s["end_time"] + "Z"):
            raise ValueError("Pick outside station epoch")
    counts = {e: len(rows) for e, rows in groups.items() if len(rows) >= MIN_TRAIN}
    quotas = allocate_quotas(counts, seed)
    train, holdout, excluded_picks, excluded_events, references, split_counts = [], [], [], [], [], []
    clean_fields = [k for k in fields if k != "seconds_from_first_P"]
    for event, rows in sorted(groups.items()):
        clean = [{k: row[k] for k in clean_fields} for row in rows]
        if event not in quotas:
            excluded_events.append(dict(event_id=event, available_picks=len(rows), reason=REASON))
            excluded_picks.extend(dict(row, reason=REASON) for row in clean)
            continue
        chosen = sorted(rows, key=lambda r: rank(seed, "pick", *pick_key(r)))[:quotas[event]]
        held_keys = {pick_key(r) for r in chosen}
        reference = min(utc(r["pick_time_utc"]) for r in rows if pick_key(r) not in held_keys)
        references.append(dict(event_id=event, training_reference_utc=iso(reference)))
        split_counts.append(dict(event_id=event, available_picks=len(rows),
                                 training_picks=len(rows) - quotas[event], holdout_picks=quotas[event],
                                 note="six_pick_event_no_holdout" if len(rows) == 6 else ""))
        for row in clean:
            row["seconds_from_train_reference"] = (utc(row["pick_time_utc"]) - reference).total_seconds()
            (holdout if pick_key(row) in held_keys else train).append(row)
    event_ids = sorted(counts)
    station_ids = [s["station_id"] for s in stations]
    ei, si = {e: i for i, e in enumerate(event_ids)}, {s: i for i, s in enumerate(station_ids)}
    times = np.full((len(event_ids), len(station_ids)), np.nan, dtype="<f8")
    for row in train:
        times[ei[row["event_id"]], si[f'{row["network"]}.{row["station"]}']] = row["seconds_from_train_reference"]
    arrays = dict(event_ids=np.asarray(event_ids, dtype=str), station_ids=np.asarray(station_ids, dtype=str),
                  station_coordinates_m=np.asarray([[s[k] for k in ("x_m", "y_m", "z_m")]
                                                    for s in stations], dtype="<f8"),
                  arrival_times_s=times)
    tables = {
        "train.csv": (clean_fields + ["seconds_from_train_reference"], train),
        "holdout.csv": (clean_fields + ["seconds_from_train_reference"], holdout),
        "catalog.csv": (list(catalog[0]), catalog),
        "stations.csv": (list(stations[0]), stations),
        "event_time_references.csv": (["event_id", "training_reference_utc"], references),
        "excluded_events.csv": (["event_id", "available_picks", "reason"], excluded_events),
        "excluded_picks.csv": (clean_fields + ["reason"], excluded_picks),
        "split_counts.csv": (list(split_counts[0]), split_counts),
    }
    return dict(tables=tables, arrays=arrays, geometry=geometry, seed=seed,
                counts=dict(source_events=len(groups), source_picks=len(picks), stations=len(stations),
                            retained_events=len(counts), retained_picks=sum(counts.values()),
                            training_picks=len(train), holdout_picks=len(holdout),
                            excluded_events=len(excluded_events), excluded_picks=len(excluded_picks),
                            six_pick_events_without_holdout=sum(n == 6 for n in counts.values()),
                            events_without_holdout=sum(q == 0 for q in quotas.values())))


def write_bundle(output, bundle, source_hashes):
    """Exclusively create a new directory. Manifest is written last; no resume."""
    output = Path(output)
    output.mkdir(exist_ok=False)
    (output / "emtomo").mkdir()
    for name, (fields, rows) in bundle["tables"].items():
        with (output / name).open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    geometry = bundle["geometry"]
    (output / "geometry.json").write_bytes(json_bytes(geometry))
    # Fixed ZIP timestamps and uncompressed NPY entries give byte-identical NPZs.
    archive = output / "emtomo/arrivals.npz"
    with archive.open("xb") as stream, zipfile.ZipFile(stream, "w") as zipped:
        for key, array in bundle["arrays"].items():
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, array, allow_pickle=False)
            zipped.writestr(zipfile.ZipInfo(key + ".npy", (1980, 1, 1, 0, 0, 0)), buffer.getvalue())
    metadata = dict(schema="emtomo.real_picks.v1", domain_extent_m=list(EXTENT),
                    time_reference="earliest_training_pick_per_event",
                    units={"coordinates": "m", "arrival_time": "s"},
                    minimum_training_picks=MIN_TRAIN, arrays_sha256=sha256(archive.read_bytes()),
                    geometry=geometry)
    (output / "emtomo/metadata.json").write_bytes(json_bytes(metadata))
    manifest = dict(
        schema="geysers.real_experiment.v1", counts=bundle["counts"], source_sha256=source_hashes,
        output_sha256={str(p.relative_to(output)): sha256(p.read_bytes())
                       for p in sorted(output.rglob("*")) if p.is_file()},
        split_policy=dict(seed=bundle["seed"], unit="individual_event_station_P_pick", fraction=0.1,
                          target="round_half_up(0.10 * picks_after_n_lt_6_exclusion)",
                          quotas="floor(n/10), then largest n-10*q deficit; cap n-6; seeded SHA256 ties",
                          selection="smallest SHA256(seed, pick, event_id, network, station)",
                          minimum_training_picks=6, six_pick_events="all train; no holdout or per-event CV",
                          input_order_independent=True),
        exclusions={REASON: bundle["counts"]["excluded_events"]},
        noise={"status": "unknown", "sigma_s": None},
        velocity_model={"status": "pending_user_clarification", "chosen": None},
        hypocenter_truth={"available": False}, datum_caveat=DATUM_CAVEAT,
        catalog_policy="tomoDD ONLY; all original source event IDs including exclusions; pick origins authoritative",
        emtomo_input="emtomo/ ONLY; no catalog, UTC timestamps, event coordinates, truth, or holdout",
        time_policy="UTC differences from earliest TRAIN pick; held-out relative times may be negative",
        quality_policy="No additional QC or weight-code mapping; negative origin-relative times retained",
        geometry_policy="Fixed reference/rotation; x_new=x_old+3000; y unchanged; H=1166; never recentered",
        execution={"forward_modeling": False, "inversion": False},
    )
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    return manifest


def prepare():
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError(f"Refusing existing output: {OUTPUT}")
    paths = {"picks": DATA / "real_picks/P_picks.csv", "catalog": DATA / "events_metric_all.csv",
             "stations": DATA / "bg_stations_2011.txt"}
    raw = {k: p.read_bytes() for k, p in paths.items()}
    picks = list(csv.DictReader(io.StringIO(raw["picks"].decode())))
    catalog = list(csv.DictReader(io.StringIO(raw["catalog"].decode())))
    station_fields = ["network", "station", "latitude", "longitude", "elevation_m",
                      "site_name", "start_time", "end_time"]
    epochs = [dict(zip(station_fields, (v.strip() for v in row)))
              for row in csv.reader(io.StringIO(raw["stations"].decode()), delimiter="|")
              if row and not row[0].startswith("#")]
    bundle = build_bundle(picks, catalog, epochs)
    if (bundle["counts"]["source_picks"], bundle["counts"]["source_events"],
            bundle["counts"]["stations"]) != (49158, 3776, 31):
        raise ValueError("Source selection differs from the confirmed v1 dataset")
    hashes = {str(paths[k].relative_to(ROOT)): sha256(data) for k, data in raw.items()}
    for path in (Path(__file__).resolve(), ROOT / "prepare_geysers_geometry.py"):
        hashes[str(path.relative_to(ROOT))] = sha256(path.read_bytes())
    manifest = write_bundle(OUTPUT, bundle, hashes)
    print(json.dumps({"output": str(OUTPUT), **manifest["counts"],
                      "velocity_model": "pending; no forward/inversion run"}, indent=2))
    return manifest


if __name__ == "__main__":
    prepare()
