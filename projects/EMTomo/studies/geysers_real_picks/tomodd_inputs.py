"""Solver-free, sparse real-pick export for the corrected dd1-d40-s5 tomoDD build.

Contract
--------
``build_inputs(training_picks, events, stations, domain=Domain(...),
model=LayerModel(...), initial_xyz_km={event_id: (x,y,z), ...})`` or explicitly
``sigma_km=..., seed=...`` returns an in-memory Bundle. No input discovery, pick
loading, synthesis, subprocess, or inversion occurs. ``write_bundle`` exclusively
creates a NEW output directory; it never overwrites downloaded sources. The CLI
accepts one caller-prepared JSON object with these same keys (no split discovery).

Rows are mappings, with these required columns:
* training_picks: event_id, network, station, phase='P', pick_time_utc (timezone-
  aware ISO UTC or datetime); optional weight in (0,1], default 1. Raw weight_code
  is NOT interpreted. Duplicate event/network/station/P rows are errors, including
  multiple channels/locations: the caller must resolve those before export.
* events: event_id (positive INTEGER*4), latitude, longitude, depth_km (positive
  down from sea level), origin_time_utc; optional magnitude/horizontal_error_km/
  vertical_error_km/rms_s (zero if absent, not estimated).
* stations: network, station, latitude, longitude, elevation_m (above sea level).
Catalog/station supersets are allowed, but only IDs present in training picks
enter the export, noise draw, or graph. Unknown IDs and invalid rows fail closed.
Explicit initial_xyz_km must cover exactly the training event IDs. No held-out
rows or held-out paths are accepted. The caller owns splitting and authorization.

Geometry
--------
Domain uses the prepare_geysers_geometry convention: center_latitude/longitude
name the geographic REFERENCE point, whose local position is horizontal_offset_km.
Omitting that optional offset preserves (Lx/2,Ly/2). The real 23x11x7-km domain
uses reference 38.80275/-122.77070 and horizontal_offset_km=(13,3.5): x=u+13,
y=v+3.5 (old x+3), NOT the new box midpoint (11.5,5.5). x follows azimuth clockwise
from north, y points 90 degrees to its right; north/latitude=110.574 km/degree,
east/longitude=111.320*cos(center_lat). This is a local planar approximation,
not a global geodesic projection; antimeridian/polar-crossing grids are rejected.
Local z is positive down from top_elevation_m; catalog z=depth_km+top/1000 and
station z=(top-elevation)/1000. Perturb in this frame, then invert the rotation
back to REAL geography (never the synthetic adapter's 30.3/103.3 embedding).
Native event/MOD depths are sea-level depths and native station elevations remain
meters above sea level. MOD is a geographic lon/lat tensor grid enclosing the
rotated domain: 1-km cell-center nodes centered on its translated east/north
bounding box (rounded outward to whole-km widths), one fixed face >=12 km beyond
each main face, no free buffer. grid.dat/control retain the geographic reference;
MOD node coordinates, rather than that reference, carry the box translation. A rotated domain therefore
has a different main-node count from the old artificial-axis benchmark. Initial
1-D boundaries are LOCAL depths below the top; each node samples the layer on
its deeper side, as in external_checkerboard.build_tomodd_geometry. MOD is nodal,
so the solver interpolates between sampled layers, not a discontinuous model.

Time and formats
----------------
Absolute UTC observations are authoritative. Export origin = catalog origin
minus origin_shift_s, floored to centiseconds (native event.dat precision).
Relative seconds are recomputed from the unchanged UTC pick, including negatives.
Both inspected ph2dtN3.f and getdata_SPDR.f accept signed times. For a different
parser requiring nonnegative times, explicitly request ONE common backward shift
large enough for all training picks (e.g. 1 s covers -0.13/-0.01 s). Never clip,
replace with model times, or normalize each event to its first pick. A common
shift preserves all differential times; centisecond origin quantization is
recorded per event and UTC reconstruction remains accurate to a microsecond.

Native, whitespace-delimited files (station aliases are unique <=7 characters):
* phase.dat: '# Y M D h m sec lat lon depth mag herr zerr rms ID', then
  'STA relative_seconds weight P'. ph2dt.inp names station.dat and phase.dat,
  then MINWGHT MAXDIST MAXSEP MAXNGH MINLNK MINOBS MAXOBS MINSEP.
* event.dat: YYYYMMDD HHMMSScc lat lon depth mag herr zerr rms ID type=0.
  The type column is REQUIRED by this tomoDD fork; ph2dt's event output omits it.
* station.dat: STA latitude longitude elevation_m.
* absolute.dat: '# ID', then 'STA relative_seconds weight P'.
* dt.ct: '# ID1 ID2', then 'STA seconds1 seconds2 mean_weight P'; these are two
  relative times, NOT one pre-subtracted difference. Only common training P
  stations are included. dt.cc is empty (IDAT=2, IPHA=1).
* MOD: bld nx ny nz; longitude nodes; latitude nodes; sea-level depth nodes;
  Vp rows (x fastest, then y, then z); then Vp/Vs rows in the same order.
* grid.dat and tomoDD-SE.inp follow the snapshot renderer; schedule is
  '3 -9 -9 -9 -9 1 .08 -9 -9 1 40 1 0', smoothing 5/5/5, ray step .05 km,
  free hypocenters, Air_dep=-4 km, catalog starts, LSQR. Control data <=80 columns.

Provenance/dependencies
-----------------------
Python >=3.10 + NumPy only. Native serializers/geometry/noise/greedy pair algorithm
are adapted from runs/matched_geometry_20261021/snapshot/benchmark/
tomodd_benchmark/{external_checkerboard,synthetic,runner}.py; importing that
synthetic adapter would also import plotting/forward-data machinery, so we do not.
Live tools: /home/egor/tomodd_benchmark/tomodd_benchmark. Parser source:
/home/egor/tomodd_pipeline/USTC-tomoDD-SE-training2020/Src/ph2dt/ph2dtN3.f.
Required eventual solver: runs/tomodd_guard_fix_20261027/build/tomoDD-SE, with
coordinate-roundoff + 0.05-km segment fixes and corrected horizontal RaySPDR2
lookup guard (NOT the old frozen binary). The bottom guard remains unchanged.
Support files ak135.15.SKS and layer-16.dat are in the matched snapshot/support;
write_bundle can copy explicitly supplied support bytes. No binary is executed or
certified here. Historical sigma=1 km/seed=42 is evidence, NOT a default grant.
ph2dt.inp is interoperable preprocessing input, not a reproduction of our strict
undirected degree-4 graph. Do not overwrite the adapter's native files with its
outputs. A future run must separately verify solver/support hashes and retention.
Static limits checked here are maxnx/ny/nz=90/80/35 (including fixed faces),
4000 events, 150 stations, 60 picks/event, 1100000 observations, and 200000-entry
axis lookup tables. Native REAL*4 axes must remain finite and strictly increasing.
These checks do not certify runtime ray/sparse-memory capacity: MAXNODE=300
per-ray derivative storage and MAXSEG=10000 still require native preflight.
The requested paper model and sigma=1-km choice remain unresolved/unauthorized;
this module selects neither, and no real-data bundle has been generated.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable, Mapping

import numpy as np

SUPPORT_FILES = ("ak135.15.SKS", "layer-16.dat")
SCHEDULE = "3 -9 -9 -9 -9 1 0.08 -9 -9 1 40 1 0"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value):
    value = float(value)
    _require(math.isfinite(value), "Numeric inputs must be finite")
    return value


def _utc(value):
    if not isinstance(value, datetime):
        value = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    _require(value.tzinfo is not None and value.utcoffset() is not None,
             "UTC timestamps must have an explicit timezone")
    return value.astimezone(timezone.utc)


def _iso(value):
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _event_id(value):
    text = str(value)
    _require(re.fullmatch(r"[0-9]+", text) is not None, "event_id must be a positive integer")
    value = int(text)
    _require(0 < value <= 2147483647, "event_id exceeds positive INTEGER*4 range")
    return value


def _station_key(row):
    key = (str(row["network"]), str(row["station"]))
    _require(all(re.fullmatch(r"[A-Za-z0-9_-]+", part) for part in key),
             "network/station must be nonempty safe identifiers")
    return key


@dataclass(frozen=True)
class Domain:
    center_latitude: float
    center_longitude: float
    azimuth_degrees: float
    lengths_km: tuple[float, float, float]
    top_elevation_m: float
    horizontal_offset_km: tuple[float, float] | None = None

    def __post_init__(self):
        _require(-89 < _number(self.center_latitude) < 89, "Invalid center latitude")
        _require(-180 < _number(self.center_longitude) < 180, "Invalid center longitude")
        _number(self.azimuth_degrees)
        _require(len(self.lengths_km) == 3 and all(_number(v) > 0 for v in self.lengths_km),
                 "Domain needs three positive lengths_km")
        _require(_number(self.top_elevation_m) <= 4000, "Domain top is above Air_dep=-4 km")
        if self.horizontal_offset_km is not None:
            _require(len(self.horizontal_offset_km) == 2, "horizontal_offset_km needs two finite offsets")
            object.__setattr__(self, "horizontal_offset_km", tuple(map(_number, self.horizontal_offset_km)))

    @property
    def effective_horizontal_offset_km(self):
        """Local x/y of the geographic reference, not necessarily the box center."""
        if self.horizontal_offset_km is None:
            return (self.lengths_km[0] / 2, self.lengths_km[1] / 2)
        return self.horizontal_offset_km

    def _east_north(self, x, y):
        offset_x, offset_y = self.effective_horizontal_offset_km
        x, y = x - offset_x, y - offset_y
        angle = math.radians(self.azimuth_degrees)
        return (x * math.sin(angle) + y * math.cos(angle),
                x * math.cos(angle) - y * math.sin(angle))

    @property
    def lon_scale(self):
        return 111.320 * math.cos(math.radians(self.center_latitude))

    def local(self, latitude, longitude, depth_km):
        lat, lon, depth = map(_number, (latitude, longitude, depth_km))
        _require(-90 <= lat <= 90 and -180 <= lon <= 180, "Invalid geographic coordinates")
        north = (lat - self.center_latitude) * 110.574
        east = (lon - self.center_longitude) * self.lon_scale
        angle = math.radians(self.azimuth_degrees)
        offset_x, offset_y = self.effective_horizontal_offset_km
        return np.array([north * math.cos(angle) + east * math.sin(angle) + offset_x,
                         -north * math.sin(angle) + east * math.cos(angle) + offset_y,
                         depth + self.top_elevation_m / 1000])

    def geographic(self, xyz):
        xyz = np.asarray(xyz, dtype=float)
        _require(xyz.shape == (3,) and np.isfinite(xyz).all(), "Invalid local xyz")
        x, y, z = xyz
        east, north = self._east_north(x, y)
        return (self.center_latitude + north / 110.574,
                self.center_longitude + east / self.lon_scale,
                z - self.top_elevation_m / 1000)

    def check_inside(self, xyz):
        xyz = np.asarray(xyz, dtype=float)
        _require(xyz.shape[-1:] == (3,) and np.isfinite(xyz).all(), "Invalid local xyz")
        _require(np.all(xyz >= -1e-9) and np.all(xyz <= np.asarray(self.lengths_km) + 1e-9),
                 "Coordinates outside the physical domain; do not silently clip")


@dataclass(frozen=True)
class LayerModel:
    boundaries_km: tuple[float, ...]
    vp_km_s: tuple[float, ...]
    vp_vs: float = 1.73

    def __post_init__(self):
        _require(len(self.vp_km_s) == len(self.boundaries_km) + 1,
                 "Layer model needs one more velocity than boundaries")
        _require(all(_number(v) > 0 for v in self.vp_km_s), "Vp must be positive km/s")
        boundaries = [_number(v) for v in self.boundaries_km]
        _require(all(a < b for a, b in zip(boundaries, boundaries[1:])), "Unsorted boundaries")
        _require(_number(self.vp_vs) > 1, "Vp/Vs must exceed one")


@dataclass(frozen=True)
class GradientModel:
    """Vp nodes at local depths; linear Vp, constant continuation outside nodes."""

    depths_km: tuple[float, ...]
    vp_km_s: tuple[float, ...]
    vp_vs: float = 1.73

    def __post_init__(self):
        _require(len(self.depths_km) == len(self.vp_km_s) >= 2, "Gradient needs >=2 depth/velocity nodes")
        depths = [_number(v) for v in self.depths_km]
        _require(all(a < b for a, b in zip(depths, depths[1:])), "Unsorted gradient depths")
        _require(all(_number(v) > 0 for v in self.vp_km_s), "Vp must be positive km/s")
        _require(_number(self.vp_vs) > 1, "Vp/Vs must exceed one")


@dataclass(frozen=True)
class Bundle:
    files: Mapping[str, str]
    manifest: dict


def _perturb(xyz, lengths, sigma, seed):
    """Match old coordinate-wise Gaussian rejection, with a finite safety bound."""
    sigma = _number(sigma)
    _require(sigma >= 0, "sigma_km must be nonnegative")
    _require(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0,
             "seed must be an explicit nonnegative integer")
    if sigma == 0:
        return xyz.copy()
    rng = np.random.default_rng(seed)
    offsets = rng.normal(0, sigma, size=xyz.shape)
    for _ in range(10000):
        initial = xyz + offsets
        invalid = (initial < 0) | (initial > lengths)
        if not invalid.any():
            return initial
        offsets[invalid] = rng.normal(0, sigma, size=int(invalid.sum()))
    raise ValueError("Coordinate rejection did not converge; check sigma/domain")


def _pairs(xyz, picks_by_event):
    """Distance-sorted greedy degree<=4 graph; zero-common-pick edges cost no degree."""
    candidates = []
    for first in range(len(xyz)):
        distances = np.linalg.norm(xyz[first + 1:] - xyz[first], axis=1)
        for offset in np.flatnonzero(distances <= 3.0):
            second = first + 1 + int(offset)
            if picks_by_event[first].keys() & picks_by_event[second].keys():
                candidates.append((float(distances[offset]), first, second))
    degree = [0] * len(xyz)
    selected = []
    for _, first, second in sorted(candidates):
        if degree[first] < 4 and degree[second] < 4:
            selected.append((first, second))
            degree[first] += 1
            degree[second] += 1
    return selected, degree


def _model_files(domain, model):
    if isinstance(model, LayerModel):
        _require(all(0 < b < domain.lengths_km[2] for b in model.boundaries_km),
                 "Layer boundaries must be inside local domain depth")
    angle = math.radians(domain.azimuth_degrees)
    lx, ly, lz = domain.lengths_km
    east_width = abs(math.sin(angle)) * lx + abs(math.cos(angle)) * ly
    north_width = abs(math.cos(angle)) * lx + abs(math.sin(angle)) * ly
    widths = tuple(map(_number, (east_width, north_width, lz)))
    counts = [int(math.ceil(v - 1e-10)) for v in widths]
    _require(all(n >= 3 for n in counts), "Need at least three 1-km main nodes per axis")
    _require(all(n + 2 <= limit for n, limit in zip(counts, (90, 80, 35))),
             "MOD exceeds corrected build's maxnx/maxny/maxnz")
    # Widths alone are insufficient: an explicit offset translates the box
    # relative to the reference. Use the same inverse rotation as event export.
    centers = domain._east_north(lx / 2, ly / 2)
    _require(all(math.isfinite(value) for value in centers), "Nonfinite domain bounding box")
    axes = []
    for index, count in enumerate(counts):
        main = np.arange(count, dtype=float) + 0.5
        if index < 2:
            main += centers[index] - count / 2
        else:
            main -= domain.top_elevation_m / 1000
        axes.append(np.r_[main[0] - 12, main, main[-1] + 12])
    east, north, depths = axes
    lon = domain.center_longitude + east / domain.lon_scale
    lat = domain.center_latitude + north / 110.574
    _require(np.all(abs(lon) < 180) and np.all(abs(lat) < 90), "MOD crosses geographic boundary")
    # The solver stores MOD axes in REAL*4. Extreme datums can erase 1-km
    # spacing even when the Python float64 axes appear ordered.
    for axis in (lon, lat, depths):
        _require(np.isfinite(axis).all() and np.all(np.abs(axis) <= np.finfo(np.float32).max),
                 "MOD axis cannot be represented by native REAL*4")
        native_axis = axis.astype(np.float32)
        _require(np.all(np.diff(native_axis) > 0), "MOD nodes collapse in native REAL*4")
    bld = 0.001 if min(np.diff(lon).min(), np.diff(lat).min()) < 0.01 else 0.01
    _require(all(np.ptp(a) / bld < 199999 for a in (lon, lat, depths)), "MOD lookup capacity exceeded")
    nx, ny, nz = map(len, (lon, lat, depths))
    lines = [f"{bld:g} {nx} {ny} {nz}"]
    lines.extend(" ".join(f"{v:.8f}" for v in axis) for axis in (lon, lat, depths))
    for depth in depths:
        local_depth = depth + domain.top_elevation_m / 1000
        if isinstance(model, GradientModel):
            vp = np.interp(local_depth, model.depths_km, model.vp_km_s)
        else:
            layer = np.searchsorted(model.boundaries_km, local_depth, side="right")
            vp = model.vp_km_s[layer]
        lines.extend([" ".join([f"{vp:.6f}"] * nx)] * ny)
    lines.extend([" ".join([f"{model.vp_vs:.6f}"] * nx)] * (ny * nz))
    grid = (f"{nx} {ny} {nz}\n0.1 0.1\n{domain.center_latitude} {domain.center_longitude}\n"
            + " ".join(f"{v:.8f}" for v in depths) + "\n")
    return {"MOD": "\n".join(lines) + "\n", "grid.dat": grid}, [nx, ny, nz]


def _control(domain):
    # Native ordering and blank source-parameter filename from runner._input_text.
    text = f"""* tomoDD-SE real training picks; corrected dd1-d40-s5 build required
* cross correlation differential times:
dt.cc
* catalog differential times:
dt.ct
* event file:
event.dat
* station file:
station.dat
* original locations:
tomoDD.loc
* relocations:
tomoDD.reloc
* station information:
tomoDD.sta
* residual information:
tomoDD.res
* source parameter information:

* Velocity file
tomoDD.vel
* Vp Model
Vp_model.dat
* Vs Model
Vs_model.dat
* Absolute file
absolute.dat
* IDAT IPHA DIST
2 1 1000
* OBSCC OBSCT Air_dep
0 0 -4.0
* ISTART ISOLV NSET
2 2 1
* iuses iuseq invdel stepl
1 0 0 0.05
* wlat wlon rota
{domain.center_latitude:.8f} {domain.center_longitude:.8f} 0
* weight1 weight2 weight3 CC_format
5 5 5 1
* NITER WTCCP WTCCS WRCC WDCC WTCTP WTCTS WRCT WDCT WTDD DAMP JOINT THRE
{SCHEDULE}
* CID
0
* ID
"""
    _require(all(len(line) <= 80 for line in text.splitlines() if not line.startswith("*")),
             "Control data exceeds line*80")
    return text


def build_inputs(training_picks: Iterable[Mapping], events: Iterable[Mapping],
                 stations: Iterable[Mapping], *, domain: Domain, model: LayerModel | GradientModel,
                 initial_xyz_km: Mapping | None = None, sigma_km: float | None = None,
                 seed: int | None = None, origin_shift_s: float = 0.0) -> Bundle:
    """Validate/render only. All lengths km; no noise default and no pick synthesis.

    Isolated events remain in absolute.dat and are reported, not silently dropped.
    A sparse real dataset need not have the complete synthetic graph's connectivity.
    Noise draws and tie breaks are deterministic in ascending numeric event-ID order.
    ``origin_shift_s`` is a nonnegative, common backward shift in whole centiseconds.
    """
    _require((initial_xyz_km is not None and sigma_km is None and seed is None)
             or (initial_xyz_km is None and sigma_km is not None and seed is not None),
             "Supply either initial_xyz_km OR explicit sigma_km and seed (no defaults)")
    shift = _number(origin_shift_s)
    _require(shift >= 0 and abs(shift * 100 - round(shift * 100)) < 1e-8,
             "origin_shift_s must be nonnegative whole centiseconds")
    event_rows, station_rows = {}, {}
    for row in events:
        key = _event_id(row["event_id"])
        _require(key not in event_rows, "Duplicate catalog event_id")
        event_rows[key] = row
    for row in stations:
        key = _station_key(row)
        _require(key not in station_rows, "Duplicate station coordinates")
        station_rows[key] = row
    picks = {}
    for row in training_picks:
        event, station = _event_id(row["event_id"]), _station_key(row)
        _require(row["phase"] == "P", "Only explicit P training picks are supported")
        _require(event in event_rows and station in station_rows, "Unknown event/station in training picks")
        key = (event, station)
        _require(key not in picks, "Duplicate training event/station/P pick; resolve channels explicitly")
        weight = _number(row.get("weight", 1.0))
        _require(0 < weight <= 1, "Explicit weight must be in (0,1]; raw weight_code is not a weight")
        picks[key] = (_utc(row["pick_time_utc"]), weight)
    _require(bool(picks), "No training picks")
    ids = sorted({key[0] for key in picks})
    station_keys = sorted({key[1] for key in picks})
    _require(len(ids) <= 4000 and len(station_keys) <= 150, "Solver event/station capacity exceeded")
    xyz = np.array([domain.local(event_rows[e]["latitude"], event_rows[e]["longitude"],
                                 event_rows[e]["depth_km"]) for e in ids])
    domain.check_inside(xyz)
    if initial_xyz_km is None:
        initial = _perturb(xyz, np.asarray(domain.lengths_km), sigma_km, seed)
        noise = {"sigma_km": float(sigma_km), "seed": seed, "method": "coordinate-wise rejection"}
    else:
        initial_map = {}
        for key, value in initial_xyz_km.items():
            event = _event_id(key)
            _require(event not in initial_map, "Duplicate normalized initial event_id")
            initial_map[event] = value
        _require(set(initial_map) == set(ids), "initial_xyz_km must cover exactly training event IDs")
        initial = np.asarray([initial_map[e] for e in ids], dtype=float)
        _require(initial.shape == xyz.shape, "initial_xyz_km needs one xyz triple per event")
        noise = {"method": "caller-supplied xyz", "sigma_km": None, "seed": None}
    domain.check_inside(initial)
    aliases = {key: f"S{i + 1:06d}" for i, key in enumerate(station_keys)}
    station_lines = []
    for key in station_keys:
        row = station_rows[key]
        elevation = _number(row["elevation_m"])
        local = domain.local(row["latitude"], row["longitude"], -elevation / 1000)
        domain.check_inside(local)
        station_lines.append(f"{aliases[key]} {float(row['latitude']):.8f} {float(row['longitude']):.8f} {elevation:.6f}")
    grouped = [{} for _ in ids]
    index = {event: i for i, event in enumerate(ids)}
    for (event, station), value in picks.items():
        grouped[index[event]][station] = value
    _require(max(map(len, grouped)) <= 60, "More than MAXOBS=60 training picks per event")
    selected, degree = _pairs(initial, grouped)
    files, shape = _model_files(domain, model)
    phase, absolute, event_lines, event_metadata, pick_metadata = [], [], [], [], []
    relative = {}
    for i, event in enumerate(ids):
        row = event_rows[event]
        origin = _utc(row["origin_time_utc"])
        shifted = origin - timedelta(seconds=shift)
        exported = shifted.replace(microsecond=(shifted.microsecond // 10000) * 10000)
        lat, lon, depth = domain.geographic(initial[i])
        mag, herr, zerr, rms = [_number(row.get(name, 0)) for name in
                              ("magnitude", "horizontal_error_km", "vertical_error_km", "rms_s")]
        _require(min(herr, zerr, rms) >= 0, "Catalog errors/RMS cannot be negative")
        geo = f"{lat:.8f} {lon:.8f} {depth:.8f} {mag:.4f} {herr:.4f} {zerr:.4f} {rms:.4f} {event}"
        seconds = exported.second + exported.microsecond / 1e6
        phase.append(f"# {exported.year} {exported.month} {exported.day} {exported.hour} {exported.minute} {seconds:.2f} {geo}")
        time_code = exported.strftime("%H%M%S") + f"{exported.microsecond // 10000:02d}"
        event_lines.append(f"{exported:%Y%m%d} {time_code} {geo} 0")
        absolute.append(f"# {event}")
        for station, (picked, weight) in sorted(grouped[i].items()):
            seconds = (picked - exported).total_seconds()
            relative[event, station] = seconds
            line = f"{aliases[station]} {seconds:.6f} {weight:.9g} P"
            phase.append(line)
            absolute.append(line)
            pick_metadata.append({"event_id": event, "station_alias": aliases[station],
                                  "pick_time_utc": _iso(picked), "weight": weight,
                                  "catalog_relative_s": (picked - origin).total_seconds()})
        event_metadata.append({"event_id": event, "catalog_origin_time_utc": _iso(origin),
                               "export_origin_time_utc": _iso(exported),
                               "effective_backward_shift_s": (origin - exported).total_seconds(),
                               "catalog_geographic_lat_lon_depth": [float(row[name]) for name in
                                                                     ("latitude", "longitude", "depth_km")],
                               "catalog_xyz_km": xyz[i].tolist(), "initial_xyz_km": initial[i].tolist()})
    differential, pair_metadata = [], []
    count = 0
    for first, second in selected:
        a, b = ids[first], ids[second]
        common = sorted(grouped[first].keys() & grouped[second].keys())
        differential.append(f"# {a} {b}")
        for station in common:
            weight = (grouped[first][station][1] + grouped[second][station][1]) / 2
            differential.append(f"{aliases[station]} {relative[a, station]:.6f} {relative[b, station]:.6f} {weight:.9g} P")
        count += len(common)
        pair_metadata.append({"event_ids": [a, b], "common_training_picks": len(common)})
    _require(len(picks) + count <= 1100000, "MAXDATA exceeded")
    for name, lines in (("phase.dat", phase), ("event.dat", event_lines),
                        ("station.dat", station_lines), ("absolute.dat", absolute), ("dt.ct", differential)):
        limit = 180 if name == "phase.dat" else 200
        _require(all(len(line) <= limit for line in lines), f"{name} exceeds native line*{limit}")
        files[name] = "\n".join(lines) + ("\n" if lines else "")
    files["dt.cc"] = ""
    files["tomoDD-SE.inp"] = _control(domain)
    files["ph2dt.inp"] = ("* Interoperability only; ph2dt graph differs from strict degree-4 graph\n"
                          "station.dat\nphase.dat\n"
                          "* MINWGHT MAXDIST MAXSEP MAXNGH MINLNK MINOBS MAXOBS MINSEP\n"
                          "0 1000 3 4 1 1 60 0\n")
    manifest = {
        "schema_version": 1, "recipe": "dd1-d40-s5", "solver_launched": False,
        "requires_corrected_horizontal_guard": True, "domain": asdict(domain), "model": asdict(model),
        "noise": noise, "common_origin_shift_s": shift, "events": event_metadata,
        "stations": [{"alias": aliases[key], "network": key[0], "station": key[1]} for key in station_keys],
        "training_picks": pick_metadata, "event_pairs": pair_metadata,
        "isolated_event_ids": [event for event, d in zip(ids, degree) if d == 0],
        "absolute_observations": len(picks), "differential_observations": count,
        "mod_shape_nx_ny_nz": shape, "pair_max_distance_km": 3.0, "pair_max_degree": 4,
        "missing_support_files": list(SUPPORT_FILES),
        "file_sha256": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in files.items()},
    }
    return Bundle(files, manifest)


def write_bundle(bundle: Bundle, output: str | Path, *, support_files: Mapping[str, bytes] | None = None):
    """Write only a new directory; caller may explicitly supply the two support blobs.

    Parent must exist. No source files are read. A partial failure is retained for
    inspection rather than deleting history. No solver is called, even with support.
    """
    output = Path(output)
    support = {} if support_files is None else dict(support_files)
    _require(not support or set(support) == set(SUPPORT_FILES), "Supply both native support files or neither")
    _require(all(isinstance(data, bytes) and data for data in support.values()), "Empty/invalid support bytes")
    names = set(bundle.files) | set(support)
    _require(all(Path(name).name == name and name not in (".", "..", "manifest.json") for name in names),
             "Unsafe bundle filename")
    _require(not set(bundle.files) & set(support), "Support filename collision")
    manifest = dict(bundle.manifest)
    manifest["missing_support_files"] = [name for name in SUPPORT_FILES if name not in support]
    manifest["file_sha256"] = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in bundle.files.items()}
    manifest["file_sha256"].update({name: hashlib.sha256(data).hexdigest() for name, data in support.items()})
    output.mkdir(exist_ok=False)
    for name, text in bundle.files.items():
        (output / name).write_bytes(text.encode())
    for name, data in support.items():
        (output / name).write_bytes(data)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export caller-prepared training-only JSON; never run tomoDD")
    parser.add_argument("--input-json", type=Path, required=True,
                        help="Object: training_picks, events, stations, domain, model, and explicit noise/xyz")
    parser.add_argument("--output", type=Path, required=True, help="New directory; parent must exist")
    args = parser.parse_args(argv)
    request = json.loads(args.input_json.read_text())
    request["domain"] = Domain(**request["domain"])
    model_type = GradientModel if "depths_km" in request["model"] else LayerModel
    request["model"] = model_type(**request["model"])
    bundle = build_inputs(**request)
    write_bundle(bundle, args.output)
    print(json.dumps({"output": str(args.output), "solver_launched": False,
                      "missing_support_files": list(SUPPORT_FILES)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
