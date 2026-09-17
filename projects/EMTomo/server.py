#!/usr/bin/env python3
"""
Tomography log viewer — HTTP backend.

Install:  pip install flask numpy
Run:      python server.py [--runs-dir /runs] [--host 0.0.0.0] [--port 5050]
Open:     http://localhost:5050
"""

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from flask import Flask, abort, jsonify, request, send_from_directory

app = Flask(__name__)
RUNS_DIR   = Path("runs")
VIEWER_DIR = Path(__file__).parent


# ─── helpers ──────────────────────────────────────────────────────────────────

def _rd(run_id: str) -> Path:
    d = RUNS_DIR / run_id
    if not d.is_dir():
        abort(404, description=f"Run not found: {run_id}")
    return d


def _npy(path: Path):
    return np.load(path, mmap_mode="r") if path.exists() else None


def _npz(path: Path, key: str):
    if not path.exists():
        return None
    with np.load(path) as data:
        return data[key]


def _load_weights(path: Path) -> np.ndarray | None:
    """Load legacy dense or compact sparse event weights."""
    if not path.exists():
        return None
    with np.load(path) as data:
        if "weights" in data:
            return data["weights"]
        if "weight_shape" not in data or "weight_indices" not in data:
            return None
        shape = tuple(int(value) for value in data["weight_shape"])
        indices = np.asarray(data["weight_indices"], dtype=np.intp)
        values = (
            np.asarray(data["weight_values"], dtype=np.float64)
            if "weight_values" in data
            else np.ones(len(indices), dtype=np.float64)
        )
        result = np.zeros(shape, dtype=np.float64)
        if len(indices):
            result[tuple(indices.T)] = values
        return result


def _load_G_station(path_stem: Path) -> np.ndarray | None:
    """Load one station G from compact sparse, NPZ, or legacy NPY storage."""
    sparse_path = path_stem.parent / "G_stations_sparse.npz"
    if sparse_path.exists():
        station = int(path_stem.name.rsplit("_", 1)[1])
        with np.load(sparse_path) as data:
            offsets = data["offsets"]
            if station < 0 or station + 1 >= len(offsets):
                return None
            start, stop = int(offsets[station]), int(offsets[station + 1])
            result = np.zeros(tuple(data["shape"]), dtype=np.float32)
            coords = data["coords"][start:stop]
            result[tuple(coords.T)] = data["values"][start:stop]
            return result

    npz_path = path_stem.with_suffix(".npz")
    npy_path = path_stem.with_suffix(".npy")
    if npz_path.exists():
        with np.load(npz_path) as data:
            return data["G"]
    if npy_path.exists():
        return np.load(npy_path)
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
    values = tuple(request.args.get(name, type=int) for name in ("nx", "ny", "nz"))
    return values if all(value is not None and value > 0 for value in values) else None


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
        (d for d in RUNS_DIR.iterdir() if d.is_dir() and (d / "meta.json").exists()),
        key=_run_sort_key,
        reverse=True,
    )
    return jsonify([d.name for d in run_dirs])


# ─── meta + info ──────────────────────────────────────────────────────────────

@app.route("/api/runs/<rid>/meta")
def api_meta(rid):
    p = _rd(rid) / "meta.json"
    if not p.exists():
        return jsonify({})
    meta = json.loads(p.read_text())

    gi = meta.get("grid_info", {})
    meta["coarse_ny"] = gi.get("coarse_shape", [1, 1, 1])[1] if "coarse_shape" in gi else 1
    meta["fine_ny"]   = gi.get("fine_shape",   [1, 1, 1])[1] if "fine_shape"   in gi else 1

    return jsonify(meta)


@app.route("/api/runs/<rid>/info")
def api_info(rid):
    rd = _rd(rid)

    iters = sorted(
        int(d.name[5:]) for d in rd.iterdir()
        if d.is_dir() and d.name.startswith("iter_")
    )

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
        n_events = len(json.loads(meta_path.read_text()).get("event_locs", []))

    return jsonify({
        "iterations":     iters,
        "n_stations":     n_stations,
        "n_events":       n_events,
        "has_true_model":      (rd / "true_model.npy").exists(),
        "has_true_model_fine": (rd / "true_model_fine.npy").exists(),
    })


# ─── timing ───────────────────────────────────────────────────────────────────

@app.route("/api/runs/<rid>/timing")
def api_timing(rid):
    p = _rd(rid) / "timing.jsonl"
    if not p.exists():
        return jsonify([])
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return jsonify(rows)


@app.route("/api/runs/<rid>/quality")
def api_quality(rid):
    p = _rd(rid) / "quality.jsonl"
    if not p.exists():
        return jsonify([])
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return jsonify(rows)


# ─── hypocenter quality ───────────────────────────────────────────────────────

def _sorted_event_dirs(iter_dir: Path):
    return sorted(
        [d for d in iter_dir.iterdir() if d.is_dir() and d.name.startswith("event_")],
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
    if not wp.exists() or not true_loc:
        return None
    try:
        with np.load(wp) as data:
            if "positions" in data and len(data["positions"]):
                values = (
                    data["weight_values"]
                    if "weight_values" in data
                    else np.ones(len(data["positions"]))
                )
                coord = np.asarray(
                    data["positions"][int(np.argmax(values))], dtype=np.float64
                )
            else:
                weights = _load_weights(wp)
                if weights is None:
                    return None
                coord = np.asarray(
                    np.unravel_index(int(np.argmax(weights)), weights.shape),
                    dtype=np.float64,
                )
        est = (coord + 0.5) * cell_size
        true = np.asarray(true_loc, dtype=np.float64)
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


def _hypo_signature(rd: Path) -> tuple:
    signature = []
    for iter_dir in sorted(rd.glob("iter_*")):
        event_dirs = list(iter_dir.glob("event_*"))
        newest_event = max(
            (event_dir.stat().st_mtime_ns for event_dir in event_dirs), default=0
        )
        signature.append((iter_dir.name, len(event_dirs), newest_event))
    timing = rd / "timing.jsonl"
    signature.append(("timing", timing.stat().st_size if timing.exists() else 0, 0))
    return tuple(signature)


@lru_cache(maxsize=16)
def _collect_hypo_dataset(rd_name: str, _signature: tuple) -> dict:
    """Read all expensive per-event metrics once for an unchanged run."""
    rd = Path(rd_name)
    meta = _read_meta(rd)
    event_locs = meta.get("event_locs") or []
    cell_size = _fine_cell_size(meta)
    residual_by_iter: dict[int, list[dict]] = {}
    distance_by_iter: dict[int, list[dict]] = {}

    iter_dirs = sorted(
        [d for d in rd.iterdir() if d.is_dir() and d.name.startswith("iter_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    for iter_dir in iter_dirs:
        iteration = int(iter_dir.name.split("_")[1])
        for event_dir in _sorted_event_dirs(iter_dir):
            event = int(event_dir.name.split("_")[1])
            rms = _rms_from_residuals(event_dir / "residuals.npy")
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
    del meta  # Metadata is loaded inside the cache for a stable cache key.
    dataset = _collect_hypo_dataset(str(rd.resolve()), _hypo_signature(rd))
    return {
        "residual_summary": dataset["residual_summary"],
        "distance_summary": dataset["distance_summary"],
        "residual_iter": dataset["residual_by_iter"].get(current_iter, []),
        "distance_iter": dataset["distance_by_iter"].get(current_iter, []),
    }


@app.route("/api/runs/latest")
def api_runs_latest():
    """Most recent run id and its max iteration."""
    if not RUNS_DIR.exists():
        return jsonify({"run_id": None, "max_iter": 0})
    runs = sorted(
        (d for d in RUNS_DIR.iterdir() if d.is_dir() and (d / "meta.json").exists()),
        key=_run_sort_key,
        reverse=True,
    )
    if not runs:
        return jsonify({"run_id": None, "max_iter": 0})
    rd = runs[0]
    rid = rd.name
    iters = [
        int(d.name[5:]) for d in rd.iterdir()
        if d.is_dir() and d.name.startswith("iter_")
    ]
    return jsonify({"run_id": rid, "max_iter": max(iters) if iters else 0})


@app.route("/api/runs/<rid>/hypo_metrics")
def api_hypo_metrics(rid):
    """Combined hypo residual + distance metrics for one iteration and all iters."""
    rd = _rd(rid)
    it = request.args.get("iter", 0, type=int)
    meta = _read_meta(rd)
    return jsonify(_collect_all_hypo_metrics(rd, meta, it))


# ─── iter / event / weight lists ──────────────────────────────────────────────

@app.route("/api/runs/<rid>/iters_list")
def api_iters_list(rid):
    rd = _rd(rid)
    iters = sorted(
        [d.name for d in rd.iterdir() if d.is_dir() and d.name.startswith("iter_")],
        key=lambda x: int(x.split("_")[1]),
    )
    return jsonify(iters)


@app.route("/api/runs/<rid>/events_list")
def api_events_list(rid):
    rd  = _rd(rid)
    itr = request.args.get("iter", "0")
    d   = rd / f"iter_{itr}"
    if not d.exists():
        return jsonify([])
    evs = sorted(
        [x.name for x in d.iterdir() if x.is_dir() and x.name.startswith("event_")],
        key=lambda x: int(x.split("_")[1]),
    )
    return jsonify(evs)


@app.route("/api/runs/<rid>/weights_list")
def api_weights_list(rid):
    rd  = _rd(rid)
    itr = request.args.get("iter",  "0")
    ev  = request.args.get("event", "0")
    d   = rd / f"iter_{itr}" / f"event_{ev}"
    if not d.exists():
        return jsonify([])
    ws = sorted(
        [x.name for x in d.iterdir() if x.is_dir() and x.name.startswith("weight_")],
        key=lambda x: int(x.split("_")[1]),
    )
    return jsonify(ws)


# ─── main slice endpoint ───────────────────────────────────────────────────────

@app.route("/api/runs/<rid>/slice")
def api_slice(rid):
    """
    Universal 2-D y-slice endpoint.

    Query params
    ────────────
    type        model | true_model | station_field | weights | G | delta_s | ray_count
    y           int   y-slice index
    iter        int   iteration number

    type=model      model_type: initial | true | iter
    type=weights    event: int
    type=G          event: int, weight: int, station: int   (fine grid via npz)
    type=ray_count  event: int, weight: int                 (coarse grid)
    type=delta_s    (no extra params)
    """
    rd    = _rd(rid)
    dtype = request.args.get("type", "model")
    y     = request.args.get("y",    0, type=int)
    it    = request.args.get("iter", 0, type=int)

    arr = None
    response_extra = {}

    if dtype == "model":
        mt = request.args.get("model_type", "iter")
        fine_true_path = rd / "true_model_fine.npy"
        paths = {
            "initial": rd / "initial_model.npy",
            "true": rd / "true_model.npy",
            "true_fine": (
                fine_true_path if fine_true_path.exists() else rd / "true_model.npy"
            ),
            "iter": rd / f"iter_{it}" / "model.npy",
        }
        arr = _npy(paths.get(mt, Path("__none__")))
        if arr is not None and arr.ndim == 3:
            meta = _read_meta(rd)
            target_shape = _target_shape()
            native = request.args.get("native", 0, type=int) == 1
            if native or (mt == "true_fine" and fine_true_path.exists()):
                target_shape = target_shape or tuple(arr.shape)
            s2d, full_shape, cell_size = _model_slice(
                arr, y, meta, target_shape
            )
            grid_step = _model_grid_step(meta, arr.shape, full_shape)
            return jsonify(_slice_resp(s2d, full_shape, cell_size, grid_step))

    elif dtype == "true_model":
        arr = _npy(rd / "true_model.npy")
        if arr is not None and arr.ndim == 3:
            meta = _read_meta(rd)
            s2d, full_shape, cell_size = _model_slice(
                arr, y, meta, _target_shape()
            )
            grid_step = _model_grid_step(meta, arr.shape, full_shape)
            return jsonify(_slice_resp(s2d, full_shape, cell_size, grid_step))

    elif dtype == "delta_s":
        arr = _npy(rd / f"iter_{it}" / "delta_s.npy")

    elif dtype == "weights":
        ev = request.args.get("event", 0, type=int)
        path = rd / f"iter_{it}" / f"event_{ev}" / "weights.npz"
        arr = _load_weights(path)
        if path.exists():
            with np.load(path) as data:
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

    elif dtype == "G":
        ev = request.args.get("event", 0, type=int)
        wt = request.args.get("weight", 0, type=int)
        sta = request.args.get("station", 0, type=int)
        # Try new npz path, fall back to old npy path.
        stem = rd / f"iter_{it}" / f"event_{ev}" / f"weight_{wt}" / f"G_station_{sta}"
        arr = _load_G_station(stem)
        weights_path = rd / f"iter_{it}" / f"event_{ev}" / "weights.npz"
        if weights_path.exists():
            with np.load(weights_path) as data:
                if "positions" in data and wt < len(data["positions"]):
                    response_extra["hypocenters"] = [{
                        "coord": data["positions"][wt].tolist(),
                        "weight": float(data["weight_values"][wt]) if "weight_values" in data else 1.0,
                    }]

    elif dtype == "ray_count":
        arr = _cached_ray_count(rd / f"iter_{it}")

    response = _arr_resp(arr, y)
    response.update(response_extra)
    return jsonify(response)


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Tomography viewer server")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--host",     default="0.0.0.0")
    ap.add_argument("--port",     type=int, default=5050)
    args = ap.parse_args()

    RUNS_DIR = Path(args.runs_dir)
    print(f"  Runs dir : {RUNS_DIR.resolve()}")
    print(f"  Viewer   : http://localhost:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=False)