#!/usr/bin/env python3
"""
Tomography log viewer — HTTP backend.

Install:  pip install flask numpy
Run:      python server.py [--runs-dir /runs] [--host 0.0.0.0] [--port 5050]
Open:     http://localhost:5050
"""

import argparse
import csv
import hashlib
import json
import math
import re
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np
from flask import Flask, abort, jsonify, request, send_from_directory

app = Flask(__name__)
VIEWER_DIR = Path(__file__).resolve().parent
RUNS_DIR = VIEWER_DIR / "runs"
EXPERIMENTS_ROOT = VIEWER_DIR.parent / "forward_modeling" / "experiments"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_MAX_AXIS = 512
_MAX_PIXELS = 65536


# ─── helpers ──────────────────────────────────────────────────────────────────

def _contained(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def _file_signature(path: Path) -> tuple:
    stat = path.stat()
    return (str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_size,
            stat.st_mtime_ns, stat.st_ctime_ns)


@lru_cache(maxsize=128)
def _file_sha256(path: Path, signature: tuple) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if _file_signature(path) != signature:
        raise ValueError("Source changed during validation")
    return digest.hexdigest()


def _source_model(rd: Path, meta: dict) -> Path | None:
    source = meta.get("source_experiment")
    if not isinstance(source, dict):
        return None
    identifier, expected = source.get("id"), source.get("model_sha256")
    if not isinstance(identifier, str) or not _ID.fullmatch(identifier) or identifier in (".", ".."):
        return None
    if not isinstance(expected, str) or not _SHA.fullmatch(expected):
        return None
    input_root = EXPERIMENTS_ROOT / "input"
    directory = input_root / identifier
    path = directory / "model.npz"
    if directory.is_symlink() or not _contained(directory, input_root) or not _contained(path, directory) or not path.is_file():
        return None
    try:
        digest = _file_sha256(path, _file_signature(path))
    except (OSError, ValueError):
        return None
    return path if digest == expected else None


def _reference_points(path: Path, id_column: str, extent: np.ndarray) -> tuple:
    # Match forward input parsing: preserve string IDs and CSV row order.
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != [id_column, "x_m", "y_m", "z_m"]:
            raise ValueError("Invalid reference CSV header")
        ids, coordinates = [], []
        for row in reader:
            if len(row) != 4 or not row[0].strip():
                raise ValueError("Invalid reference CSV row")
            ids.append(row[0])
            coordinates.append(tuple(float(v) for v in row[1:]))
    points = np.asarray(coordinates, dtype=float)
    if (not ids or len(set(ids)) != len(ids) or not np.all(np.isfinite(points))
            or np.any(points < 0) or np.any(points > extent)):
        raise ValueError("Invalid reference IDs or coordinates")
    return tuple(ids), tuple(coordinates)


def _check_reference_order(metadata: dict, event_ids: tuple, station_ids: tuple):
    for name, ids in (("event", event_ids), ("station", station_ids)):
        if name + "_ids" in metadata and metadata[name + "_ids"] != list(ids):
            raise ValueError("Reference ID order mismatch")
        count = "n_" + name + "s"
        if count in metadata and (type(metadata[count]) is not int or metadata[count] != len(ids)):
            raise ValueError("Reference count mismatch")


@lru_cache(maxsize=32)
def _validated_reference_inputs(paths: tuple, signatures: tuple) -> tuple:
    """Cache only validated, immutable geometry; arrivals are not viewer inputs."""
    model_path, stations_path, events_path, *rest = paths
    metadata = json.loads(rest[-1].read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Invalid forward metadata")
    units = {"coordinates": "m", "cell_size": "m", "velocity": "m/s",
             "arrival_time": "s", "elapsed_time": "s"}
    if (metadata.get("time_reference") != "earliest_station_arrival_per_event"
            or not isinstance(metadata.get("units"), dict)
            or any(metadata["units"].get(k) != v for k, v in units.items())):
        raise ValueError("Unsupported forward conventions")
    hashes = metadata.get("input_sha256")
    if not isinstance(hashes, dict) or set(hashes) != {p.name for p in paths[:-1]}:
        raise ValueError("Invalid forward input hash list")
    for path, signature in zip(paths[:-1], signatures[:-1]):
        if hashes[path.name] != _file_sha256(path, signature):
            raise ValueError("Forward input hash mismatch")
    with np.load(model_path, allow_pickle=False) as model:
        if set(model.files) != {"velocity", "cell_size_m", "origin_m"}:
            raise ValueError("Invalid reference model fields")
        velocity = np.asarray(model["velocity"], dtype=float)
        cell_size = model["cell_size_m"]
        origin = model["origin_m"]
        if (velocity.ndim != 3 or any(n == 0 for n in velocity.shape)
                or not np.all(np.isfinite(velocity)) or np.any(velocity <= 0)
                or cell_size.shape != () or not np.isfinite(cell_size) or cell_size <= 0
                or origin.shape != (3,) or not np.all(origin == 0)):
            raise ValueError("Invalid reference model geometry")
        extent = np.asarray(velocity.shape) * float(cell_size)
        if not np.all(np.isfinite(extent)):
            raise ValueError("Invalid reference model bounds")
    station_ids, stations = _reference_points(stations_path, "station_id", extent)
    event_ids, events = _reference_points(events_path, "event_id", extent)
    if len(station_ids) < 2:
        raise ValueError("At least two stations required")
    _check_reference_order(metadata, event_ids, station_ids)
    if tuple(_file_signature(path) for path in paths) != signatures:
        raise ValueError("Source changed during validation")
    return event_ids, events, station_ids, stations


def _reference_events(rd: Path, meta: dict) -> tuple:
    """Viewer-only truth in input row order, matching experiment_data's indexing."""
    if _source_model(rd, meta) is None:
        return (), ()
    identifier = meta["source_experiment"]["id"]
    try:
        paths = []
        for section, names in (
            ("input", ("model.npz", "stations.csv", "events.csv", "generation.json")),
            ("output", ("metadata.json",)),
        ):
            root = EXPERIMENTS_ROOT / section
            directory = root / identifier
            if directory.is_symlink() or not _contained(directory, root):
                return (), ()
            for name in names:
                path = directory / name
                if not _contained(path, directory):
                    return (), ()
                if name == "generation.json" and not path.exists():
                    continue
                paths.append(path)
        signatures = tuple(_file_signature(path) for path in paths)
        event_ids, events, station_ids, stations = _validated_reference_inputs(tuple(paths), signatures)
        params = meta.get("run_params", {})
        if type(params.get("n_events")) is not int or len(event_ids) != params["n_events"]:
            return (), ()
        _check_reference_order(meta, event_ids, station_ids)
        _check_reference_order(params, event_ids, station_ids)
        if not np.array_equal(np.asarray(meta.get("station_locs", []), dtype=float), stations):
            return (), ()
        return event_ids, events
    except (OSError, ValueError, TypeError, KeyError, csv.Error):
        return (), ()


def _eligible(rd: Path) -> bool:
    if not rd.is_dir() or not _contained(rd, RUNS_DIR):
        return False
    meta_path = rd / "meta.json"
    if not _contained(meta_path, rd):
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError):
        return False
    return isinstance(meta, dict) and _source_model(rd, meta) is not None


def _rd(run_id: str) -> Path:
    if not _ID.fullmatch(run_id) or run_id in (".", ".."):
        abort(404)
    d = RUNS_DIR / run_id
    if not _eligible(d):
        abort(404, description=f"Run not found: {run_id}")
    return d


def _child(rd: Path, *parts: str) -> Path:
    path = rd.joinpath(*parts)
    if not _contained(path, rd):
        abort(404)
    return path


def _index(name: str, default: int = 0) -> int:
    raw = request.args.get(name, str(default))
    if not re.fullmatch(r"\d+", raw):
        abort(400, description=f"Invalid {name}")
    return int(raw)


def _y_index(y_km: float, shape: tuple[int, ...], cell_size: float) -> int:
    extent = shape[1] * cell_size / 1000
    if not math.isfinite(y_km) or y_km < 0 or y_km > extent:
        abort(400, description="y_km outside model bounds")
    return min(int(y_km * 1000 / cell_size), shape[1] - 1)


def _requested_y(shape: tuple[int, ...], cell_size: float) -> tuple[float, int]:
    raw = request.args.get("y_km")
    if raw is None:
        # Default to the first cell centre; y_km is always a physical coordinate.
        y_km = cell_size / 2000
    else:
        try:
            y_km = float(raw)
        except ValueError:
            abort(400, description="Invalid y_km")
    return y_km, _y_index(y_km, shape, cell_size)


def _axes(response: dict, cell_size: float, y_km: float, extents_m=None) -> dict:
    nx, _, nz = response["full_shape"]
    sx, _, sz = extents_m if extents_m is not None else (nx * cell_size, 0, nz * cell_size)
    response.update({
        "origin_m": [0, 0, 0], "y_km": y_km,
        "x_km": ((np.arange(nx) + 0.5) * sx / nx / 1000).tolist(),
        "z_km": ((np.arange(nz) + 0.5) * sz / nz / 1000).tolist(),
        "x_edges_km": (np.arange(nx + 1) * sx / nx / 1000).tolist(),
        "z_edges_km": (np.arange(nz + 1) * sz / nz / 1000).tolist(),
    })
    return response


def _npy(path: Path):
    return np.load(path, mmap_mode="r") if path.exists() else None



def _sparse_plane(shape, indices, values, y: int, dtype=np.float64) -> np.ndarray:
    """Scatter only the requested X-Z plane, never a dense 3-D ray/weight grid."""
    result = np.zeros((int(shape[0]), int(shape[2])), dtype=dtype)
    indices = np.asarray(indices, dtype=np.intp)
    if len(indices):
        selected = indices[:, 1] == y
        coords = indices[selected]
        result[coords[:, 0], coords[:, 2]] = np.asarray(values)[selected]
    return result


def _load_weights(path: Path, y: int | None = None) -> np.ndarray | None:
    """Load compact weights; the optional plane keeps the utility API compatible."""
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        if "weight_shape" not in data or "weight_indices" not in data:
            return None
        shape = tuple(int(value) for value in data["weight_shape"])
        indices = np.asarray(data["weight_indices"], dtype=np.intp)
        values = (
            np.asarray(data["weight_values"], dtype=np.float64)
            if "weight_values" in data
            else np.ones(len(indices), dtype=np.float64)
        )
        if y is not None:
            return _sparse_plane(shape, indices, values, y)
        result = np.zeros(shape, dtype=np.float64)
        if len(indices):
            result[tuple(indices.T)] = values
        return result


def _load_G_station(path_stem: Path, y: int) -> np.ndarray | None:
    """Load one station's requested plane from the compact sparse event log."""
    sparse_path = path_stem.parent / "G_stations_sparse.npz"
    if sparse_path.exists():
        station = int(path_stem.name.rsplit("_", 1)[1])
        with np.load(sparse_path, allow_pickle=False) as data:
            offsets = data["offsets"]
            if station < 0 or station + 1 >= len(offsets):
                return None
            start, stop = int(offsets[station]), int(offsets[station + 1])
            return _sparse_plane(data["shape"], data["coords"][start:stop],
                                 data["values"][start:stop], y, np.float32)

    loaded = _legacy_G_plane(path_stem, lambda shape: int(np.clip(y, 0, shape[1] - 1)))
    return loaded[0] if loaded is not None else None


def _legacy_G_plane(path_stem: Path, select_y) -> tuple[np.ndarray, tuple] | None:
    """Read dense standalone G via mmap or streamed NPZ rows, without a volume."""
    for suffix in (".npy", ".npz"):
        path = path_stem.with_suffix(suffix)
        if not path.is_file() or not _contained(path, path_stem.parent):
            continue
        if suffix == ".npy":
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.ndim != 3:
                return None
            return array[:, select_y(array.shape), :], array.shape
        with ZipFile(path) as archive:
            names = archive.namelist()
            if {"shape.npy", "coords.npy", "values.npy"}.issubset(names):
                with np.load(path, allow_pickle=False) as data:
                    shape = tuple(int(n) for n in data["shape"])
                    return _sparse_plane(shape, data["coords"], data["values"], select_y(shape), np.float32), shape
            member = next((name for name in ("G.npy", "arr_0.npy") if name in names), None)
            if member is None:
                return None
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                header_reader = {(1, 0): np.lib.format.read_array_header_1_0,
                                 (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                if header_reader is None:
                    return None
                shape, fortran, dtype = header_reader(stream)
                if len(shape) != 3 or dtype.hasobject:
                    return None
                y = select_y(shape)
                start = stream.tell()
                nx, ny, nz = shape
                plane = np.empty((nx, nz), dtype=dtype)
                # ZipExtFile.seek skips/decompresses forward in bounded chunks;
                # only one contiguous row and the requested plane are retained.
                for index in range(nz if fortran else nx):
                    count = nx if fortran else nz
                    offset = (index * ny + y) * count * dtype.itemsize
                    stream.seek(start + offset)
                    row = np.frombuffer(stream.read(count * dtype.itemsize), dtype=dtype, count=count)
                    if fortran:
                        plane[:, index] = row
                    else:
                        plane[index, :] = row
                return plane, shape
    return None


def _slice_y(arr: np.ndarray, y: int) -> np.ndarray:
    """arr[:, y, :] → (nx, nz), y clamped."""
    y = int(np.clip(y, 0, arr.shape[1] - 1))
    return arr[:, y, :]


def _run_sort_key(run_dir: Path) -> tuple:
    """Sort runs by their timestamp suffix, not by descriptive name."""
    matches = re.findall(r"(\d{8})_(\d{6})(?:$|_)", run_dir.name)
    timestamp = "".join(matches[-1]) if matches else ""
    return (timestamp, run_dir.stat().st_mtime, run_dir.name)


def _model_slice(
    arr: np.ndarray,
    y: int,
    meta: dict,
    target_shape: tuple[int, int, int] | None = None,
) -> tuple[np.ndarray, list[int], float]:
    """Return a model slice sampled like the cell-centred ray-tracing grid."""
    params = meta.get("run_params") or {}
    grid = meta.get("grid_info") or {}
    subdivision = max(1, int(params.get("subdivision") or 1))
    mode = params.get("slowness_interpolation", "nearest")
    full_shape = list(target_shape or tuple(int(size) * subdivision for size in arr.shape))
    y = int(np.clip(y, 0, full_shape[1] - 1))

    def source_coords(source_size: int, target_size: int) -> np.ndarray:
        indices = np.arange(target_size, dtype=np.float64)
        return np.clip(
            (indices + 0.5) * source_size / target_size - 0.5,
            0.0,
            source_size - 1.0,
        )

    if full_shape == list(arr.shape):
        result = arr[:, y, :]
    elif mode == "trilinear":
        def axis_weights(size: int, target_size: int):
            coord = source_coords(size, target_size)
            lower = np.floor(coord).astype(np.intp)
            upper = np.minimum(lower + 1, size - 1)
            return lower, upper, 1.0 - (coord - lower), coord - lower

        ix0, ix1, wx0, wx1 = axis_weights(arr.shape[0], full_shape[0])
        iy0, iy1, wy0, wy1 = axis_weights(arr.shape[1], full_shape[1])
        iz0, iz1, wz0, wz1 = axis_weights(arr.shape[2], full_shape[2])
        positive = bool(np.all(arr > 0))
        source = 1.0 / np.asarray(arr, dtype=np.float64) if positive else np.asarray(arr, dtype=np.float64)
        result = np.zeros((full_shape[0], full_shape[2]), dtype=np.float64)
        for ix, wx in ((ix0, wx0), (ix1, wx1)):
            for iy, wy in ((iy0[y], wy0[y]), (iy1[y], wy1[y])):
                for iz, wz in ((iz0, wz0), (iz1, wz1)):
                    result += (
                        source[ix[:, None], iy, iz[None, :]]
                        * wx[:, None]
                        * wy
                        * wz[None, :]
                    )
        if positive:
            result = 1.0 / result
    else:
        ix = np.floor(
            (np.arange(full_shape[0], dtype=np.float64) + 0.5)
            * arr.shape[0]
            / full_shape[0]
        ).astype(np.intp)
        iy = min(arr.shape[1] - 1, int((y + 0.5) * arr.shape[1] / full_shape[1]))
        iz = np.floor(
            (np.arange(full_shape[2], dtype=np.float64) + 0.5)
            * arr.shape[2]
            / full_shape[2]
        ).astype(np.intp)
        result = arr[ix[:, None], iy, iz[None, :]]

    coarse_side = grid.get("coarse_side_m") if grid else None
    side_x = coarse_side[0] if isinstance(coarse_side, (list, tuple)) else None
    cell_size = (
        float(side_x) / full_shape[0]
        if side_x
        else float(grid.get("fine_cell_size") or 1.0)
    )
    # GeoGrid stores interpolated velocity as float32; matching that precision
    # also prevents color scales from amplifying float64 round-off around constants.
    return np.asarray(result, dtype=np.float32), full_shape, cell_size


def _target_shape() -> tuple[int, int, int] | None:
    names = ("nx", "ny", "nz")
    if not any(name in request.args for name in names):
        return None
    if not all(name in request.args for name in names):
        abort(400, description="nx, ny, nz must be specified together")
    values = tuple(_index(name) for name in names)
    if any(value < 1 or value > _MAX_AXIS for value in values) or values[0] * values[2] > _MAX_PIXELS:
        abort(400, description="Requested model grid too large")
    return values


def _model_grid_step(meta: dict, source_shape, target_shape) -> list[int]:
    mode = (meta.get("run_params") or {}).get("slowness_interpolation", "nearest")
    if mode != "nearest":
        return [1, 1]
    steps = []
    for axis in (0, 2):
        ratio = target_shape[axis] / source_shape[axis]
        steps.append(int(round(ratio)) if ratio >= 1 and float(ratio).is_integer() else 1)
    return steps


def _arr_resp(arr, y: int):
    """Turn 3-D array into a slice response dict, or a null response."""
    if arr is None or arr.ndim != 3:
        return {"slice": None, "shape": None, "full_shape": None, "vmin": 0, "vmax": 1}
    s2d = _slice_y(arr, y)
    fin = s2d[np.isfinite(s2d)]
    return {
        "slice":      s2d.tolist(),
        "shape":      list(s2d.shape),
        "full_shape": list(arr.shape),
        "vmin": float(fin.min()) if fin.size else 0.0,
        "vmax": float(fin.max()) if fin.size else 1.0,
    }


@lru_cache(maxsize=32)
def _sum_ray_counts(file_signature: tuple[tuple[str, int, int], ...]):
    total = None
    for filename, _mtime_ns, _size in file_signature:
        ray_count = np.load(filename).astype(np.float32)
        total = ray_count if total is None else total + ray_count
    return total


def _cached_ray_count(iter_dir: Path):
    aggregate = iter_dir / "ray_count.npy"
    if aggregate.exists():
        return np.load(aggregate, mmap_mode="r")
    files = sorted(iter_dir.glob("event_*/weight_*/ray_count.npy"))
    signature = tuple(
        (str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in files
    )
    return _sum_ray_counts(signature)


def _slice_resp(
    s2d: np.ndarray,
    full_shape: list[int],
    cell_size: float,
    grid_step: list[int] | None = None,
) -> dict:
    fin = s2d[np.isfinite(s2d)]
    return {
        "slice": s2d.tolist(),
        "shape": list(s2d.shape),
        "full_shape": full_shape,
        "cell_size": cell_size,
        "grid_step": grid_step or [1, 1],
        "vmin": float(fin.min()) if fin.size else 0.0,
        "vmax": float(fin.max()) if fin.size else 1.0,
    }


# ─── static ───────────────────────────────────────────────────────────────────

@app.route("/")
def root():
    return send_from_directory(str(VIEWER_DIR), "viewer.html")


# ─── run list ─────────────────────────────────────────────────────────────────

@app.route("/api/runs")
def api_runs():
    if not RUNS_DIR.exists():
        return jsonify([])
    run_dirs = sorted(
        (d for d in RUNS_DIR.iterdir() if _ID.fullmatch(d.name) and _eligible(d)),
        key=_run_sort_key,
        reverse=True,
    )
    return jsonify([d.name for d in run_dirs])


def _jsonl_rows(path: Path, metric: str) -> dict[int, dict]:
    """Ignore malformed records and ambiguous (duplicate) iteration indices."""
    if not path.is_file():
        return {}
    rows, seen = {}, set()
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
            iteration = row["iter"]
            value = row[metric]
            if (type(iteration) is not int or iteration < 0
                    or isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                continue
        except (ValueError, TypeError, KeyError):
            continue
        if iteration in seen:
            rows.pop(iteration, None)
        else:
            rows[iteration] = row
        seen.add(iteration)
    return rows


def _publication_stat(path: Path) -> tuple | None:
    """Cheap identity/change token; no recursive walk or content hashing."""
    try:
        stat = path.stat()
        return (stat.st_dev, stat.st_ino, stat.st_mode, stat.st_size,
                stat.st_mtime_ns, stat.st_ctime_ns)
    except OSError:
        return None


@lru_cache(maxsize=8192)
def _readable_saved_file(filename: str, signature: tuple) -> bool:
    try:
        data = np.load(filename, mmap_mode="r", allow_pickle=False)
        if isinstance(data, np.lib.npyio.NpzFile):
            data.close()
        return True
    except (OSError, ValueError, EOFError, BadZipFile):
        return False


def _saved_file(path: Path, rd: Path) -> bool:
    if not _contained(path, rd) or not path.is_file():
        return False
    signature = _publication_stat(path)
    return signature is not None and _readable_saved_file(str(path), signature)


def _legacy_features(rd: Path, meta: dict, directories: list[Path]) -> dict:
    """Infer the writer layout once per run/publication change, not per cycle."""
    params = meta.get("run_params") or {}
    saved = meta.get("viewer_saved_artifacts") or {}
    terminal_proof = "coverage_damping_power" in params
    diagnostics = terminal_proof or any(
        (directory / name).exists() for directory in directories
        for name in ("sensitivity_diagonal.npy", "coverage_confidence.npy"))
    timefields = saved.get("save_timefields", params.get("save_timefields", False))
    misfit = saved.get("save_misfit", params.get("save_misfit", False))
    if not terminal_proof:
        timefields = timefields or any((d / "station_fields.npy").exists() for d in directories)
        misfit = misfit or any(rd.glob("iter_*/event_*/misfit.npy"))
    return {
        "terminal_proof": terminal_proof, "diagnostics": diagnostics,
        "timefields": bool(timefields), "misfit": bool(misfit),
        "quality": params.get("viewer_quality_expected", bool(meta.get("source_experiment"))
                              or (rd / "true_model.npy").exists() or (rd / "quality.jsonl").exists()),
        "g": params.get("log_g_per_weight", params.get("log_G_per_weight", False)),
    }


def _legacy_g_files(weight: Path, n_stations: int) -> list[Path]:
    """Compact multi-station logs and the older standalone station layouts."""
    compact = weight / "G_stations_sparse.npz"
    if compact.exists():
        return [compact]
    if n_stations:
        return [next((p for p in (weight / f"G_station_{station}.npy",
                                  weight / f"G_station_{station}.npz") if p.exists()),
                     weight / f"G_station_{station}.npy") for station in range(n_stations)]
    return list(weight.glob("G_station_*.np[yz]")) or [compact]


def _legacy_iteration_ready(rd: Path, iteration: int, meta: dict, quality: dict,
                            features: dict, pending: set[Path]) -> bool:
    """End saves prove the event phase finished for the known diagnostics writer."""
    directory = rd / f"iter_{iteration}"
    params = meta.get("run_params") or {}
    required = [directory / "model.npy", directory / "delta_s.npy"]
    if features["diagnostics"]:
        required.extend(directory / name for name in ("sensitivity_diagonal.npy", "coverage_confidence.npy"))
    if features["timefields"]:
        required.append(directory / "station_fields.npy")
    if features["quality"] and iteration not in quality:
        return False
    failed = [path for path in required if not _saved_file(path, rd)]
    if failed:
        pending.update(failed)
        return False
    if features["terminal_proof"]:
        # This writer appends timing only after all events and the velocity update,
        # then saves delta, both diagnostics, and conditional quality. Reopening
        # every preceding event archive adds no publication evidence.
        return True
    n_events = params.get("n_events")
    events = ([directory / f"event_{event}" for event in range(n_events)]
              if type(n_events) is int and n_events >= 0
              else list(directory.glob("event_*")))
    for event in events:
        weights_path = event / "weights.npz"
        required.extend((weights_path, event / "residuals.npy"))
        if features["misfit"]:
            required.append(event / "misfit.npy")
        if not _saved_file(weights_path, rd):
            pending.add(weights_path)
            return False
        try:
            with np.load(weights_path, allow_pickle=False) as data:
                indices = data["positions"] if "positions" in data else data["weight_indices"]
                values = data["weight_values"] if "weight_values" in data else np.ones(len(indices))
                if values.shape != (len(indices),) or not np.all(np.isfinite(values)) or np.any(values < 0):
                    pending.add(weights_path)
                    return False
                weight_indices = set(np.flatnonzero(values > 0).tolist())
        except (OSError, ValueError, KeyError, EOFError, BadZipFile):
            pending.add(weights_path)
            return False
        weight_indices.update(int(w.name[7:]) for w in event.glob("weight_*")
                              if w.is_dir() and re.fullmatch(r"weight_\d+", w.name))
        for index in weight_indices:
            weight = event / f"weight_{index}"
            required.append(weight / "ray_count.npy")
            if features["g"]:
                g_files = _legacy_g_files(weight, len(meta.get("station_locs", [])))
                required.extend(g_files)
                if not all(_saved_file(path, rd) for path in g_files):
                    pending.add(weight)  # Notice .npz/compact publication after a missing .npy.
    failed = [path for path in required if not _saved_file(path, rd)]
    pending.update(failed)
    return not failed


_completion_cache: OrderedDict[str, dict] = OrderedDict()


def _completed_iterations(rd: Path, meta: dict | None = None) -> list[int]:
    meta = _read_meta(rd) if meta is None else meta
    directories = sorted([d for d in rd.iterdir() if d.is_dir()
                          and re.fullmatch(r"iter_\d+", d.name)
                          and d.name == f"iter_{int(d.name[5:])}" and _contained(d, rd)],
                         key=lambda d: int(d.name[5:]))
    cache_id = str(rd.resolve())
    previous = _completion_cache.get(cache_id)
    roots = ("meta.json", "timing.jsonl", "quality.jsonl", "true_model.npy")
    terminal = ("complete.json",) if "viewer_completion_protocol" in meta else (
        "model.npy", "delta_s.npy", "sensitivity_diagonal.npy", "coverage_confidence.npy", "station_fields.npy")
    signature = (
        tuple(_publication_stat(rd / name) for name in roots),
        tuple((d.name, _publication_stat(d), tuple(_publication_stat(d / name) for name in terminal))
              for d in directories),
        tuple((path, _publication_stat(path)) for path in previous["pending"]) if previous else (),
    )
    if previous and signature == previous["signature"]:
        _completion_cache.move_to_end(cache_id)
        return list(previous["iterations"])
    pending: set[Path] = set()
    features = None
    feature_key = None
    if "viewer_completion_protocol" in meta:
        if type(meta["viewer_completion_protocol"]) is not int or meta["viewer_completion_protocol"] != 1:
            return []
        completed = set()
        for directory in directories:
            iteration = int(directory.name[5:])
            marker = directory / "complete.json"
            if not _contained(marker, rd):
                continue
            try:
                row = json.loads(marker.read_text())
                if (isinstance(row, dict) and type(row.get("iter")) is int
                        and row["iter"] == iteration and type(row.get("viewer_completion_protocol")) is int
                        and row["viewer_completion_protocol"] == 1):
                    completed.add(iteration)
            except (OSError, ValueError):
                continue
        iterations = sorted(completed)
    elif not all(_contained(rd / name, rd) for name in ("timing.jsonl", "quality.jsonl")):
        iterations = []
    else:
        # Known writer features do not depend on event history or timing appends.
        # Older, underspecified layouts may discover new optional files while pending.
        feature_key = (signature[0][0], signature[0][2] is not None, signature[0][3] is not None,
                       None if "coverage_damping_power" in (meta.get("run_params") or {}) else signature[1])
        if previous and previous.get("feature_key") == feature_key:
            features = previous["features"]
        else:
            features = _legacy_features(rd, meta, directories)
        timing = _jsonl_rows(rd / "timing.jsonl", "elapsed_s")
        quality = _jsonl_rows(rd / "quality.jsonl", "avg_abs_pct_dev")
        iterations = sorted(i for i in timing if _legacy_iteration_ready(rd, i, meta, quality, features, pending))
    # Only unresolved old-layout paths are watched below the iteration root.
    # Successfully published event outputs are immutable; end saves remain watched.
    pending = tuple(sorted(pending))
    signature = (*signature[:2], tuple((path, _publication_stat(path)) for path in pending))
    entry = {"signature": signature, "iterations": tuple(iterations), "pending": pending}
    if features is not None:
        entry.update(features=features, feature_key=feature_key)
    _completion_cache[cache_id] = entry
    _completion_cache.move_to_end(cache_id)
    if len(_completion_cache) > 256:
        _completion_cache.popitem(last=False)
    return iterations


def _run_summary(rd: Path, meta: dict | None = None) -> dict:
    meta = _read_meta(rd) if meta is None else meta
    iterations = _completed_iterations(rd, meta)
    planned = (meta.get("run_params") or {}).get("n_cycles")
    run_name = meta.get("run_name") or (meta.get("run_params") or {}).get("run_name")
    return {"id": rd.name, "run_name": run_name if isinstance(run_name, str) and run_name else rd.name,
            "completed_iterations": len(iterations),
            "planned_iterations": planned if type(planned) is int and planned >= 0 else None,
            "iterations": iterations}


@app.route("/api/runs/summary")
def api_runs_summary():
    if not RUNS_DIR.exists():
        return jsonify([])
    runs = sorted((d for d in RUNS_DIR.iterdir() if _ID.fullmatch(d.name) and _eligible(d)),
                  key=_run_sort_key, reverse=True)
    return jsonify([_run_summary(rd) for rd in runs])


# ─── meta + info ──────────────────────────────────────────────────────────────

@app.route("/api/runs/<rid>/meta")
def api_meta(rid):
    rd = _rd(rid)
    p = rd / "meta.json"
    if not p.exists():
        return jsonify({})
    meta = json.loads(p.read_text())

    event_ids, coordinates = _reference_events(rd, meta)
    meta["reference_event_ids"] = event_ids
    meta["reference_event_coordinates_m"] = coordinates
    meta["has_reference_events"] = bool(coordinates)
    # Compatibility alias for existing viewer overlays, not inversion metadata.
    meta["event_locs"] = coordinates

    gi = meta.get("grid_info", {})
    meta["coarse_ny"] = gi.get("coarse_shape", [1, 1, 1])[1] if "coarse_shape" in gi else 1
    meta["fine_ny"]   = gi.get("fine_shape",   [1, 1, 1])[1] if "fine_shape"   in gi else 1

    return jsonify(meta)


@app.route("/api/runs/<rid>/info")
def api_info(rid):
    rd = _rd(rid)

    meta = _read_meta(rd)
    summary = _run_summary(rd, meta)
    iters = summary["iterations"]

    n_stations = 0
    # Try to infer station count from G files or weights
    for i in iters:
        iter_d = rd / f"iter_{i}"
        # check first event's weight_0 for G files
        ev0 = iter_d / "event_0" / "weight_0"
        if ev0.exists():
            sparse_g = ev0 / "G_stations_sparse.npz"
            if sparse_g.exists():
                with np.load(sparse_g) as data:
                    n_stations = max(0, len(data["offsets"]) - 1)
                break
            g_files = list(ev0.glob("G_station_*.np*"))
            if g_files:
                n_stations = len(g_files)
                break
        # fallback: station_fields
        p = iter_d / "station_fields.npy"
        if p.exists():
            n_stations = int(np.load(p, mmap_mode="r").shape[0])
            break

    # also try meta
    if n_stations == 0:
        meta_path = rd / "meta.json"
        if meta_path.exists():
            m = json.loads(meta_path.read_text())
            n_stations = len(m.get("station_locs", []))

    meta_path = rd / "meta.json"
    n_events = 0
    if meta_path.exists():
        m = json.loads(meta_path.read_text())
        n_events = m.get("run_params", {}).get("n_events", len(m.get("event_locs", [])))

    return jsonify({
        **summary,
        "n_stations":     n_stations,
        "n_events":       n_events,
        "has_true_model": _source_model(rd, meta) is not None,
    })


# ─── timing ───────────────────────────────────────────────────────────────────

@app.route("/api/runs/<rid>/timing")
def api_timing(rid):
    rd = _rd(rid)
    completed = _completed_iterations(rd)
    rows = _jsonl_rows(_child(rd, "timing.jsonl"), "elapsed_s")
    return jsonify([rows[i] for i in completed if i in rows])


@app.route("/api/runs/<rid>/quality")
def api_quality(rid):
    rd = _rd(rid)
    completed = _completed_iterations(rd)
    rows = _jsonl_rows(_child(rd, "quality.jsonl"), "avg_abs_pct_dev")
    return jsonify([rows[i] for i in completed if i in rows])


# ─── hypocenter quality ───────────────────────────────────────────────────────

def _sorted_event_dirs(iter_dir: Path):
    return sorted(
        [d for d in iter_dir.iterdir() if d.is_dir() and re.fullmatch(r"event_\d+", d.name)
         and _contained(d, iter_dir)],
        key=lambda d: int(d.name.split("_")[1]),
    )


def _read_meta(rd: Path) -> dict:
    p = rd / "meta.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _fine_cell_size(meta: dict) -> float:
    gi = meta.get("grid_info") or {}
    return float(gi.get("fine_cell_size") or gi.get("coarse_cell_size") or 1.0)


def _rms_from_residuals(rfile: Path) -> float | None:
    if not rfile.exists():
        return None
    try:
        R = np.load(rfile, mmap_mode="r")
        idx = np.triu_indices(R.shape[0], k=1)
        vals = R[idx]
        return float(np.sqrt(np.mean(vals ** 2))) if vals.size > 0 else 0.0
    except Exception:
        return None


def _dist_to_true_hypo(ev_dir: Path, true_loc, cell_size: float) -> float | None:
    """Euclidean distance (m) from the best refined hypothesis to truth."""
    wp = ev_dir / "weights.npz"
    if not wp.exists() or not _contained(wp, ev_dir) or true_loc is None:
        return None
    try:
        with np.load(wp) as data:
            if "positions" in data and len(data["positions"]):
                values = (
                    data["weight_values"]
                    if "weight_values" in data
                    else np.ones(len(data["positions"]))
                )
                if values.shape != (len(data["positions"]),) or not np.all(np.isfinite(values)) or np.any(values < 0) or not np.any(values > 0):
                    return None
                coord = np.asarray(
                    data["positions"][int(np.argmax(values))], dtype=np.float64
                )
            else:
                if "weight_indices" not in data or not len(data["weight_indices"]):
                    return None
                indices = data["weight_indices"]
                values = data["weight_values"] if "weight_values" in data else np.ones(len(indices))
                if values.shape != (len(indices),) or not np.all(np.isfinite(values)) or np.any(values < 0) or not np.any(values > 0):
                    return None
                # The former dense argmax chose the first cell in C-order on ties.
                best = np.flatnonzero(values == values.max())
                order = np.lexsort(indices[best].T[::-1])
                coord = np.asarray(indices[best[order[0]]], dtype=np.float64)
        est = (coord + 0.5) * cell_size
        true = np.asarray(true_loc, dtype=np.float64)
        if est.shape != (3,) or true.shape != (3,) or not np.all(np.isfinite(est)) or not np.all(np.isfinite(true)):
            return None
        return float(np.linalg.norm(est - true))
    except Exception:
        return None


def _aggregate(vals: list[float]) -> dict | None:
    if not vals:
        return None
    arr = np.array(vals, dtype=np.float64)
    return {
        "mean":   float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10":    float(np.percentile(arr, 10)),
        "p90":    float(np.percentile(arr, 90)),
        "n":      len(vals),
    }


def _hypo_signature(rd: Path, iterations: tuple[int, ...]) -> tuple:
    signature = []
    paths = [rd / "meta.json"]
    for iteration in iterations:
        for pattern in ("event_*/weights.npz", "event_*/residuals.npy"):
            paths.extend(sorted((rd / f"iter_{iteration}").glob(pattern)))
    for path in paths:
        if _contained(path, rd) and path.is_file():
            stat = path.stat()
            signature.append((str(path.relative_to(rd)), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    return tuple(signature)


@lru_cache(maxsize=16)
def _collect_hypo_dataset(rd_name: str, _signature: tuple, event_locs: tuple, iterations: tuple[int, ...]) -> dict:
    """Read all expensive per-event metrics once for an unchanged run."""
    rd = Path(rd_name)
    meta = _read_meta(rd)

    cell_size = _fine_cell_size(meta)
    residual_by_iter: dict[int, list[dict]] = {}
    distance_by_iter: dict[int, list[dict]] = {}

    iter_dirs = [rd / f"iter_{iteration}" for iteration in iterations]
    for iter_dir in iter_dirs:
        iteration = int(iter_dir.name.split("_")[1])
        for event_dir in _sorted_event_dirs(iter_dir):
            event = int(event_dir.name.split("_")[1])
            residual_file = event_dir / "residuals.npy"
            rms = _rms_from_residuals(residual_file) if _contained(residual_file, event_dir) else None
            if rms is not None:
                residual_by_iter.setdefault(iteration, []).append(
                    {"event": event, "rms": rms}
                )
            true_loc = event_locs[event] if event < len(event_locs) else None
            distance = _dist_to_true_hypo(event_dir, true_loc, cell_size)
            if distance is not None:
                distance_by_iter.setdefault(iteration, []).append(
                    {"event": event, "dist_m": distance}
                )

    def summary_rows(by_iter: dict[int, list[dict]], value_key: str) -> list[dict]:
        rows = []
        for iteration in sorted(by_iter):
            aggregate = _aggregate([row[value_key] for row in by_iter[iteration]])
            if aggregate:
                rows.append({
                    "iter": iteration,
                    "mean": aggregate["mean"],
                    "median": aggregate["median"],
                    "p10": aggregate["p10"],
                    "p90": aggregate["p90"],
                    "n_events": aggregate["n"],
                })
        return rows

    return {
        "residual_summary": summary_rows(residual_by_iter, "rms"),
        "distance_summary": summary_rows(distance_by_iter, "dist_m"),
        "residual_by_iter": residual_by_iter,
        "distance_by_iter": distance_by_iter,
    }


def _collect_all_hypo_metrics(rd: Path, meta: dict, current_iter: int) -> dict:
    _, event_locs = _reference_events(rd, meta)
    iterations = tuple(_completed_iterations(rd, meta))
    dataset = _collect_hypo_dataset(str(rd.resolve()), _hypo_signature(rd, iterations), event_locs, iterations)
    return {
        "residual_summary": dataset["residual_summary"],
        "distance_summary": dataset["distance_summary"],
        "residual_iter": dataset["residual_by_iter"].get(current_iter, []),
        "distance_iter": dataset["distance_by_iter"].get(current_iter, []),
    }


@app.route("/api/runs/latest")
def api_runs_latest():
    """Newest eligible run, including one which has not completed its first cycle."""
    if not RUNS_DIR.exists():
        return jsonify({"run_id": None, "max_iter": None})
    runs = sorted(
        (d for d in RUNS_DIR.iterdir() if _ID.fullmatch(d.name) and _eligible(d)),
        key=_run_sort_key,
        reverse=True,
    )
    if not runs:
        return jsonify({"run_id": None, "max_iter": None})
    rd = runs[0]
    iterations = _completed_iterations(rd)
    return jsonify({"run_id": rd.name, "max_iter": max(iterations) if iterations else None})


@app.route("/api/runs/<rid>/hypo_metrics")
def api_hypo_metrics(rid):
    """Combined hypo residual + distance metrics for one iteration and all iters."""
    rd = _rd(rid)
    it = _index("iter")
    meta = _read_meta(rd)
    return jsonify(_collect_all_hypo_metrics(rd, meta, it))


# ─── iter / event / weight lists ──────────────────────────────────────────────

@app.route("/api/runs/<rid>/iters_list")
def api_iters_list(rid):
    rd = _rd(rid)
    iters = [f"iter_{i}" for i in _completed_iterations(rd)]
    return jsonify(iters)


@app.route("/api/runs/<rid>/events_list")
def api_events_list(rid):
    rd  = _rd(rid)
    itr = _index("iter")
    if itr not in _completed_iterations(rd):
        return jsonify([])
    d   = _child(rd, f"iter_{itr}")
    if not d.exists():
        return jsonify([])
    evs = sorted(
        [x.name for x in d.iterdir() if x.is_dir() and re.fullmatch(r"event_\d+", x.name) and _contained(x, rd)],
        key=lambda x: int(x.split("_")[1]),
    )
    return jsonify(evs)


@app.route("/api/runs/<rid>/weights_list")
def api_weights_list(rid):
    rd  = _rd(rid)
    itr = _index("iter")
    ev  = _index("event")
    if itr not in _completed_iterations(rd):
        return jsonify([])
    d   = _child(rd, f"iter_{itr}", f"event_{ev}")
    if not d.exists():
        return jsonify([])
    ws = sorted(
        [x.name for x in d.iterdir() if x.is_dir() and re.fullmatch(r"weight_\d+", x.name) and _contained(x, rd)],
        key=lambda x: int(x.split("_")[1]),
    )
    return jsonify(ws)


def _sparse_slice_response(data, meta: dict, station: int | None = None) -> dict:
    is_weights = station is None
    shape_key = "weight_shape" if is_weights else "shape"
    if shape_key not in data:
        return _arr_resp(None, 0)
    shape = tuple(int(v) for v in data[shape_key])
    grid = meta.get("grid_info") or {}
    extents = grid.get("coarse_side_m") or [n * float(grid.get("coarse_cell_size", 1)) for n in shape]
    cell_size = float(extents[0]) / shape[0]
    y_km, y = _requested_y(shape, float(extents[1]) / shape[1])
    if is_weights:
        if "weight_indices" not in data:
            return _arr_resp(None, 0)
        indices = data["weight_indices"]
        values = data["weight_values"] if "weight_values" in data else np.ones(len(indices))
        plane = _sparse_plane(shape, indices, values, y)
    else:
        offsets = data["offsets"]
        if station + 1 >= len(offsets):
            return _arr_resp(None, 0)
        start, stop = int(offsets[station]), int(offsets[station + 1])
        plane = _sparse_plane(shape, data["coords"][start:stop], data["values"][start:stop], y, np.float32)
    return _axes(_slice_resp(plane, list(shape), cell_size), cell_size, y_km, extents)


def _legacy_G_slice_response(path_stem: Path, meta: dict) -> dict:
    geometry = {}

    def select_y(shape):
        grid = meta.get("grid_info") or {}
        extents = grid.get("coarse_side_m") or [n * float(grid.get("coarse_cell_size", 1)) for n in shape]
        cell_size = float(extents[0]) / shape[0]
        y_km, y = _requested_y(shape, float(extents[1]) / shape[1])
        geometry.update(cell_size=cell_size, y_km=y_km, extents=extents)
        return y

    loaded = _legacy_G_plane(path_stem, select_y)
    if loaded is None:
        return _arr_resp(None, 0)
    plane, shape = loaded
    return _axes(_slice_resp(plane, list(shape), geometry["cell_size"]),
                 geometry["cell_size"], geometry["y_km"], geometry["extents"])


# ─── main slice endpoint ───────────────────────────────────────────────────────

@app.route("/api/runs/<rid>/slice")
def api_slice(rid):
    """
    Universal 2-D y-slice endpoint.

    Query params
    ────────────
    type        model | weights | G | delta_s | ray_count | sensitivity_diagonal | coverage_confidence
    y_km        float physical y coordinate in kilometres
    iter        int   iteration number

    type=model      model_type: initial | true | iter
    type=weights    event: int
    type=G          event: int, weight: int, station: int   (fine grid via npz)
    type=ray_count  event: int, weight: int                 (coarse grid)
    type=delta_s    (no extra params)
    """
    rd    = _rd(rid)
    dtype = request.args.get("type", "model")
    it    = _index("iter")
    meta = _read_meta(rd)
    grid = meta.get("grid_info") or {}
    if dtype not in ("model", "weights", "G", "delta_s", "ray_count", "sensitivity_diagonal", "coverage_confidence"):
        abort(400, description="Invalid slice type")
    completed = _completed_iterations(rd, meta)
    mt = request.args.get("model_type", "iter" if completed else "initial")
    if dtype == "model" and mt not in ("initial", "true", "iter"):
        abort(400, description="Invalid model_type")
    if (dtype != "model" or mt == "iter") and it not in completed:
        abort(404, description=f"Iteration {it} is not complete")

    arr = None
    response_extra = {}

    if dtype == "model":
        if mt == "true":
            if any(name in request.args for name in ("nx", "ny", "nz")):
                abort(400, description="Truth is only available on its native grid")
            source_path = _source_model(rd, meta)
            if source_path is None:
                abort(404, description="Reference model unavailable")
            try:
                with np.load(source_path, allow_pickle=False) as data:
                    arr = data["velocity"]
                    cell_size = float(data["cell_size_m"])
                    origin = data["origin_m"]
            except (OSError, ValueError, KeyError, EOFError, BadZipFile):
                abort(404, description="Reference model unavailable")
            if arr.ndim != 3 or not np.array_equal(origin, [0, 0, 0]) or not math.isfinite(cell_size) or cell_size <= 0:
                abort(404)
            y_km, y = _requested_y(arr.shape, cell_size)
            return jsonify(_axes(_slice_resp(arr[:, y, :], list(arr.shape), cell_size), cell_size, y_km))
        if mt not in ("initial", "iter"):
            abort(400, description="Invalid model_type")
        path = _child(rd, "initial_model.npy") if mt == "initial" else _child(rd, f"iter_{it}", "model.npy")
        arr = _npy(path)
        if arr is not None and arr.ndim == 3:
            target_shape = _target_shape()
            full_shape = target_shape or tuple(int(size) * max(1, int(meta.get("run_params", {}).get("subdivision") or 1)) for size in arr.shape)
            if any(size < 1 or size > _MAX_AXIS for size in full_shape) or full_shape[0] * full_shape[2] > _MAX_PIXELS:
                abort(400, description="Model grid too large; specify a smaller nx, ny, nz")
            extent_x = float(grid.get("coarse_side_m", [arr.shape[0] * float(grid.get("coarse_cell_size", 1))])[0])
            extents = grid.get("coarse_side_m") or [arr.shape[axis] * float(grid.get("coarse_cell_size", 1)) for axis in range(3)]
            cell_size = extent_x / full_shape[0]
            # The inversion domain is unchanged by resampling on each axis.
            y_km, y = _requested_y(full_shape, float(extents[1]) / full_shape[1])
            s2d, full_shape, cell_size = _model_slice(arr, y, meta, target_shape)
            grid_step = _model_grid_step(meta, arr.shape, full_shape)
            return jsonify(_axes(_slice_resp(s2d, full_shape, cell_size, grid_step), cell_size, y_km, extents))

    elif dtype in ("delta_s", "sensitivity_diagonal", "coverage_confidence"):
        arr = _npy(_child(rd, f"iter_{it}", f"{dtype}.npy"))

    elif dtype == "weights":
        ev = _index("event")
        path = _child(rd, f"iter_{it}", f"event_{ev}", "weights.npz")
        response = _arr_resp(None, 0)
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                response = _sparse_slice_response(data, meta)
                if "positions" in data:
                    values = (
                        data["weight_values"]
                        if "weight_values" in data
                        else np.ones(len(data["positions"]))
                    )
                    response_extra["hypocenters"] = [
                        {"coord": position.tolist(), "weight": float(value)}
                        for position, value in zip(data["positions"], values)
                    ]
        response.update(response_extra)
        return jsonify(response)

    elif dtype == "G":
        ev = _index("event")
        wt = _index("weight")
        sta = _index("station")
        path = _child(rd, f"iter_{it}", f"event_{ev}", f"weight_{wt}", "G_stations_sparse.npz")
        response = _arr_resp(None, 0)
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                response = _sparse_slice_response(data, meta, sta)
        else:
            stem = _child(rd, f"iter_{it}", f"event_{ev}", f"weight_{wt}", f"G_station_{sta}")
            response = _legacy_G_slice_response(stem, meta)
        weights_path = _child(rd, f"iter_{it}", f"event_{ev}", "weights.npz")
        if weights_path.exists():
            with np.load(weights_path) as data:
                if "positions" in data and wt < len(data["positions"]):
                    response_extra["hypocenters"] = [{
                        "coord": data["positions"][wt].tolist(),
                        "weight": float(data["weight_values"][wt]) if "weight_values" in data else 1.0,
                    }]
        response.update(response_extra)
        return jsonify(response)

    elif dtype == "ray_count":
        arr = _cached_ray_count(_child(rd, f"iter_{it}"))
    else:
        abort(400, description="Invalid slice type")

    if arr is None or arr.ndim != 3:
        response = _arr_resp(arr, 0)
    else:
        extent_x = float(grid.get("coarse_side_m", [arr.shape[0] * float(grid.get("coarse_cell_size", 1))])[0])
        extents = grid.get("coarse_side_m") or [arr.shape[axis] * float(grid.get("coarse_cell_size", 1)) for axis in range(3)]
        cell_size = extent_x / arr.shape[0]
        y_km, y = _requested_y(arr.shape, float(extents[1]) / arr.shape[1])
        response = _axes(_slice_resp(arr[:, y, :], list(arr.shape), cell_size), cell_size, y_km, extents)
    response.update(response_extra)
    return jsonify(response)


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Tomography viewer server")
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    ap.add_argument("--experiments-root", default=str(EXPERIMENTS_ROOT))
    ap.add_argument("--host",     default="0.0.0.0")
    ap.add_argument("--port",     type=int, default=5050)
    args = ap.parse_args()

    RUNS_DIR = Path(args.runs_dir)
    EXPERIMENTS_ROOT = Path(args.experiments_root)
    print(f"  Runs dir : {RUNS_DIR.resolve()}")
    print(f"  Viewer   : http://localhost:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=False)