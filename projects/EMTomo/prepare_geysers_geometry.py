"""Prepare a real 2011 The Geysers geometry for EMTomo synthetic tests.

Input files are the monthly EGS CSV catalog downloads and the NCEDC FDSN BG
station response in ``data/geysers_2011``.  The resulting metric coordinates
use EMTomo's convention: x follows azimuth 135 degrees from North, y is to its
right, and z is positive downward from the model top face.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data" / "geysers_2011"
STATION_SOURCE = DATA_DIR / "bg_stations_2011.txt"
EVENT_SOURCES = sorted(DATA_DIR.glob("events_2011_*.csv"))

# Selected by a grid search that maximizes the number of working BG stations
# within the compact 20 x 7 km horizontal test window.
CENTER_LAT = 38.80275
CENTER_LON = -122.77070
AZIMUTH_DEGREES = 135.0
WIDTH_X_KM = 20.0
WIDTH_Y_KM = 7.0
MAX_DEPTH_KM = 7.0
SAMPLE_SIZE = 300
RANDOM_SEED = 20260912


def local_xy_km(latitude: float, longitude: float) -> tuple[float, float]:
    """Return EMTomo local x/y coordinates relative to the window top centre."""
    north = (latitude - CENTER_LAT) * 110.574
    east = (longitude - CENTER_LON) * (
        111.320 * math.cos(math.radians(CENTER_LAT))
    )
    azimuth = math.radians(AZIMUTH_DEGREES)
    # x is azimuth clockwise from North; y is 90 degrees to its right.
    return (
        north * math.cos(azimuth) + east * math.sin(azimuth),
        -north * math.sin(azimuth) + east * math.cos(azimuth),
    )


def inside_window(x_km: float, y_km: float, depth_km: float) -> bool:
    return (
        abs(x_km) <= WIDTH_X_KM / 2.0
        and abs(y_km) <= WIDTH_Y_KM / 2.0
        and 0.0 <= depth_km <= MAX_DEPTH_KM
    )


def load_stations() -> list[dict[str, object]]:
    stations: list[dict[str, object]] = []
    with STATION_SOURCE.open(newline="") as source:
        for row in csv.reader(source, delimiter="|"):
            if not row or row[0].startswith("#"):
                continue
            network, code, lat, lon, elevation, site, start, end = (
                value.strip() for value in row
            )
            x_km, y_km = local_xy_km(float(lat), float(lon))
            if inside_window(x_km, y_km, 0.0):
                stations.append(
                    {
                        "network": network,
                        "station": code,
                        "latitude": float(lat),
                        "longitude": float(lon),
                        "elevation_m": float(elevation),
                        "site_name": site,
                        "start_time": start,
                        "end_time": end,
                        "x_m": (x_km + WIDTH_X_KM / 2.0) * 1000.0,
                        "y_m": (y_km + WIDTH_Y_KM / 2.0) * 1000.0,
                        # EMTomo currently models a flat top boundary. Preserve
                        # real elevation above, but place receivers at z=0.
                        "z_m": 0.0,
                    }
                )
    return sorted(stations, key=lambda station: str(station["station"]))


def load_events() -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for source_path in EVENT_SOURCES:
        with source_path.open(newline="") as source:
            for row in csv.reader(source):
                if not row or not row[0].startswith("2011/"):
                    continue
                origin_time, lat, lon, depth, magnitude, magnitude_type, *_rest, event_id = row
                x_km, y_km = local_xy_km(float(lat), float(lon))
                depth_km = float(depth)
                if inside_window(x_km, y_km, depth_km):
                    events.append(
                        {
                            "event_id": event_id,
                            "origin_time": origin_time,
                            "latitude": float(lat),
                            "longitude": float(lon),
                            "depth_km": depth_km,
                            "magnitude": float(magnitude),
                            "magnitude_type": magnitude_type,
                            "x_m": (x_km + WIDTH_X_KM / 2.0) * 1000.0,
                            "y_m": (y_km + WIDTH_Y_KM / 2.0) * 1000.0,
                            "z_m": depth_km * 1000.0,
                        }
                    )
    return events


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not EVENT_SOURCES:
        raise FileNotFoundError(f"No monthly EGS event files found in {DATA_DIR}")
    stations = load_stations()
    events = load_events()
    if len(events) < SAMPLE_SIZE:
        raise ValueError(f"Only {len(events)} events available for sample size {SAMPLE_SIZE}")

    # A random, fixed-seed subset preserves the observed hypocentre density.
    indices = np.random.default_rng(RANDOM_SEED).choice(
        len(events), size=SAMPLE_SIZE, replace=False
    )
    sampled_events = [events[index] for index in sorted(indices)]

    write_csv(DATA_DIR / "stations_metric.csv", stations)
    write_csv(DATA_DIR / "events_metric_all.csv", events)
    write_csv(DATA_DIR / f"events_metric_sample_{SAMPLE_SIZE}.csv", sampled_events)
    metadata = {
        "source": {
            "catalog": "NCEDC Enhanced Geothermal Systems catalog, BG network",
            "station_service": "NCEDC FDSN Station service",
            "event_period": "2011-01-01T00:00:00 to 2011-12-31T23:59:59",
        },
        "geometry": {
            "top_face_center_latitude": CENTER_LAT,
            "top_face_center_longitude": CENTER_LON,
            "azimuth_degrees": AZIMUTH_DEGREES,
            "width_x_m": WIDTH_X_KM * 1000.0,
            "width_y_m": WIDTH_Y_KM * 1000.0,
            "depth_m": MAX_DEPTH_KM * 1000.0,
            "station_z_m": 0.0,
        },
        "counts": {
            "stations": len(stations),
            "events_in_window": len(events),
            "events_sampled": len(sampled_events),
        },
        "sampling": {"method": "uniform random without replacement", "seed": RANDOM_SEED},
    }
    (DATA_DIR / "geometry_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"Wrote {len(stations)} stations, {len(events)} events, and {len(sampled_events)} sampled events.")


if __name__ == "__main__":
    main()
