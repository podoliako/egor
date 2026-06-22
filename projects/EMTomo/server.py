#!/usr/bin/env python3
"""
Tomography log viewer — HTTP backend.

Install:  pip install flask numpy
Run:      python server.py [--runs-dir /runs] [--host 0.0.0.0] [--port 5050]
Open:     http://localhost:5050
"""

import argparse
import json
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
    return np.load(path) if path.exists() else None


def _npz(path: Path, key: str):
    return np.load(path)[key] if path.exists() else None


def _load_G_station(path_stem: Path) -> np.ndarray | None:
    """Load G_station_<si> from either .npz (new) or .npy (old) file."""
    npz_path = path_stem.with_suffix(".npz")
    npy_path = path_stem.with_suffix(".npy")
    if npz_path.exists():
        return np.load(npz_path)["G"]
    if npy_path.exists():
        return np.load(npy_path)
    return None


def _slice_y(arr: np.ndarray, y: int) -> np.ndarray:
    """arr[:, y, :] → (nx, nz), y clamped."""
    y = int(np.clip(y, 0, arr.shape[1] - 1))
    return arr[:, y, :]


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


# ─── static ───────────────────────────────────────────────────────────────────

@app.route("/")
def root():
    return send_from_directory(str(VIEWER_DIR), "viewer.html")


# ─── run list ─────────────────────────────────────────────────────────────────

@app.route("/api/runs")
def api_runs():
    if not RUNS_DIR.exists():
        return jsonify([])
    runs = sorted(
        (d.name for d in RUNS_DIR.iterdir() if d.is_dir()),
        reverse=True,
    )
    return jsonify(runs)


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
        "has_true_model": (rd / "true_model.npy").exists(),
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
    """Euclidean distance (m) from argmax weights cell to true event location."""
    wp = ev_dir / "weights.npz"
    if not wp.exists() or not true_loc:
        return None
    try:
        w = np.load(wp)["weights"]
        flat = int(np.argmax(w))
        ix, iy, iz = np.unravel_index(flat, w.shape)
        est = np.array(
            [(ix + 0.5) * cell_size, (iy + 0.5) * cell_size, (iz + 0.5) * cell_size],
            dtype=np.float64,
        )
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


def _collect_all_hypo_metrics(rd: Path, meta: dict, current_iter: int) -> dict:
    """Single pass over all iterations — summaries + current-iter rows."""
    event_locs = meta.get("event_locs") or []
    cell_size = _fine_cell_size(meta)
    rms_by_iter: dict[int, list[float]] = {}
    dist_by_iter: dict[int, list[float]] = {}
    residual_iter, distance_iter = [], []

    iter_dirs = sorted(
        [d for d in rd.iterdir() if d.is_dir() and d.name.startswith("iter_")],
        key=lambda d: int(d.name.split("_")[1]),
    )

    for it_dir in iter_dirs:
        it = int(it_dir.name.split("_")[1])
        for ev_dir in _sorted_event_dirs(it_dir):
            ev = int(ev_dir.name.split("_")[1])
            rms = _rms_from_residuals(ev_dir / "residuals.npy")
            if rms is not None:
                rms_by_iter.setdefault(it, []).append(rms)
                if it == current_iter:
                    residual_iter.append({"event": ev, "rms": rms})
            true_loc = event_locs[ev] if ev < len(event_locs) else None
            dist = _dist_to_true_hypo(ev_dir, true_loc, cell_size)
            if dist is not None:
                dist_by_iter.setdefault(it, []).append(dist)
                if it == current_iter:
                    distance_iter.append({"event": ev, "dist_m": dist})

    def _summary_rows(by_iter: dict[int, list[float]]) -> list[dict]:
        rows = []
        for it in sorted(by_iter):
            agg = _aggregate(by_iter[it])
            if agg:
                rows.append({
                    "iter": it, "mean": agg["mean"], "median": agg["median"],
                    "p10": agg["p10"], "p90": agg["p90"], "n_events": agg["n"],
                })
        return rows

    return {
        "residual_summary": _summary_rows(rms_by_iter),
        "distance_summary": _summary_rows(dist_by_iter),
        "residual_iter": residual_iter,
        "distance_iter": distance_iter,
    }


@app.route("/api/runs/latest")
def api_runs_latest():
    """Most recent run id and its max iteration."""
    if not RUNS_DIR.exists():
        return jsonify({"run_id": None, "max_iter": 0})
    runs = sorted(
        (d.name for d in RUNS_DIR.iterdir() if d.is_dir()),
        reverse=True,
    )
    if not runs:
        return jsonify({"run_id": None, "max_iter": 0})
    rid = runs[0]
    rd = RUNS_DIR / rid
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
    type=G          event: int, weight: int, station: int   (coarse grid via npz)
    type=ray_count  event: int, weight: int                 (coarse grid)
    type=delta_s    (no extra params)
    """
    rd    = _rd(rid)
    dtype = request.args.get("type", "model")
    y     = request.args.get("y",    0, type=int)
    it    = request.args.get("iter", 0, type=int)

    arr = None

    if dtype == "model":
        mt = request.args.get("model_type", "iter")
        paths = {
            "initial": rd / "initial_model.npy",
            "true":    rd / "true_model.npy",
            "iter":    rd / f"iter_{it}" / "model.npy",
        }
        arr = _npy(paths.get(mt, Path("__none__")))

    elif dtype == "true_model":
        arr = _npy(rd / "true_model.npy")

    elif dtype == "delta_s":
        arr = _npy(rd / f"iter_{it}" / "delta_s.npy")

    elif dtype == "weights":
        ev  = request.args.get("event", 0, type=int)
        arr = _npz(rd / f"iter_{it}" / f"event_{ev}" / "weights.npz", "weights")

    elif dtype == "G":
        ev     = request.args.get("event",   0, type=int)
        wt     = request.args.get("weight",  0, type=int)
        sta    = request.args.get("station", 0, type=int)
        # Try new npz path, fall back to old npy path
        stem = rd / f"iter_{it}" / f"event_{ev}" / f"weight_{wt}" / f"G_station_{sta}"
        arr  = _load_G_station(stem)

    elif dtype == "ray_count":
        iter_dir = rd / f"iter_{it}"
        arr = None
        
        # Проходим по всем событиям и весам текущей итерации
        for rc_file in iter_dir.glob("event_*/weight_*/ray_count.npy"):
            rc = np.load(rc_file).astype(np.float32)
            if arr is None:
                arr = rc
            else:
                arr += rc

    return jsonify(_arr_resp(arr, y))


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